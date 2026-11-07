"""Compare mask quality and synchronized single-sample synthesis latency."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version
import torch

from baselines import MultiPlaneGS
from evaluate import evaluate, load_grayscale_png
from fresnel import DISTANCES_MM
from model import PhaseUNet
from metrics import normalize_amplitude
from optics_torch import AmplitudeObjective
from train import CachedTargets, normalized_batch


METHOD_LABELS = {
    "gs_10": "Герчберг–Сэкстон, 10 итераций",
    "gs_100": "Герчберг–Сэкстон, 100 итераций",
    "neural": "Нейросеть с 6 оптическими шагами",
    "hybrid": "Нейросеть и ещё 10 итераций GS",
}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def encode_phase(phase):
    return torch.round((phase + torch.pi) * (255 / (2 * torch.pi))).clamp(0, 255).to(torch.uint8)


def summarize(records):
    summary = {}
    for method in METHOD_LABELS:
        selected = [record for record in records if record["method"] == method]
        latency = np.array([np.median(record["latency_ms"]) for record in selected])
        correlation = np.array([record["pearson"] for record in selected])
        ssim = np.array([record["ssim"] for record in selected])
        correct = np.array([record["correct_depth"] for record in selected])
        summary[method] = {
            "samples": len(selected),
            "pearson": float(correlation.mean()),
            "ssim": float(ssim.mean()),
            "pearson_std_across_quadruplet_means": float(correlation.mean(1).std(ddof=1)),
            "ssim_std_across_quadruplet_means": float(ssim.mean(1).std(ddof=1)),
            "depth_identification_accuracy": float(correct.mean()),
            "latency_ms_median": float(np.median(latency)),
            "latency_ms_mean": float(latency.mean()),
            "latency_ms_p10": float(np.percentile(latency, 10)),
            "latency_ms_p90": float(np.percentile(latency, 90)),
            "per_plane": [
                {"distance_mm": z, "pearson": float(correlation[:, i].mean()),
                 "ssim": float(ssim[:, i].mean()),
                 "depth_identification_accuracy": float(correct[:, i].mean())}
                for i, z in enumerate(DISTANCES_MM)
            ],
        }
    return summary


def save_comparison(output, sample, data_root, example_scores):
    font = ImageFont.load_default(size=17)
    # Arial supplies Cyrillic on the task's Windows machine.
    arial = Path("C:/Windows/Fonts/arial.ttf")
    if arial.exists():
        font = ImageFont.truetype(str(arial), 18)
    canvas = Image.new("RGB", (868, 1500), "#f6f7fa")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 14), "Сравнение методов ДОЭ — первая тестовая четвёрка", font=font, fill="#17253d")
    rows = [("Исходные изображения", None), *[(METHOD_LABELS[method], method) for method in METHOD_LABELS]]
    for row, (label, method) in enumerate(rows):
        top = 54 + row * 288
        draw.text((20, top), label, font=font, fill="#17253d")
        for plane, distance in enumerate(DISTANCES_MM):
            left = 20 + plane * 212
            path = data_root / sample["images"][plane]
            caption = f"{distance} мм"
            if method is not None:
                score = example_scores[method][plane]
                path = output / "example" / method / score["reconstruction"]
                caption += f"  КК {score['pearson']:.3f}"
            with Image.open(path) as image:
                if method is None:
                    pixels = np.rint(255 * normalize_amplitude(np.asarray(image))).astype(np.uint8)
                    image = Image.fromarray(pixels)
                canvas.paste(image.resize((192, 192), Image.Resampling.NEAREST).convert("RGB"),
                             (left, top + 28))
            draw.text((left, top + 224), caption, font=font, fill="#43536b")
            if method is not None:
                draw.text((left, top + 244), f"SSIM {score['ssim']:.3f}", font=font, fill="#43536b")
    canvas.save(output / "comparison.png")


def write_report(output, summary, config):
    lines = [
        "# Сравнение методов синтеза ДОЭ", "",
        f"Проверены {config['samples']} тестовых четвёрок на {config['device_name']}.",
        "Все оценки получены из сохранённых 8-битных PNG фазовых масок.",
        "Амплитуды каждой исходной и восстановленной картинки нормированы по собственному максимуму.",
        "", "| Метод | Средний КК | Средний SSIM | Своя глубина | Медиана времени, мс | P10–P90, мс |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, score in summary.items():
        lines.append(
            f"| {METHOD_LABELS[method]} | {score['pearson']:.4f} | {score['ssim']:.4f} | "
            f"{100 * score['depth_identification_accuracy']:.1f}% | {score['latency_ms_median']:.3f} | "
            f"{score['latency_ms_p10']:.3f}–{score['latency_ms_p90']:.3f} |"
        )
    lines += [
        "", "Время измерено для одной четвёрки за вызов, после прогрева.",
        f"На каждую четвёрку выполнено {config['timing_repeats']} измерения каждого метода.",
        "В таблице приведена медиана индивидуальных медиан по всем четвёркам.",
        "Порядок методов чередовался. CPU и GPU синхронизировались перед и после замеров.",
        "В измерение входят синтез и перевод фазы в 8-битный массив на устройстве.",
        "Загрузка модели, чтение входных PNG, перенос входов на GPU, сохранение файлов",
        "и вычисление метрик выполнялись за пределами измеряемого участка.",
        "",
        "Нейросетевой вариант включает все шесть встроенных оптических шагов.",
        "Гибрид включает полный вызов сети и десять дополнительных классических итераций.",
        "Классический метод одновременно ограничивает амплитуды всех четырёх плоскостей",
        "и усредняет обратно распространённые комплексные поля перед выделением фазы.",
        "Начальная маска строится обратным распространением с фиксированными случайными",
        "фазами, совпадающими с инициализацией нейросети. Используется один старт с seed 42.",
        "",
        "Классические ограничения амплитуды внутри GS используют одинаковую энергию полей",
        "через нормировку по среднеквадратичному значению. Оценки КК и SSIM используют",
        "нормировку по максимуму, как предусмотрено условиями задачи.",
        "",
        "Метрика своей глубины показывает долю восстановлений, у которых максимальная",
        "корреляция среди четырёх целей достигается со своей целевой картинкой.",
        "Подробные оценки по глубинам находятся в summary.json и per_plane.csv.",
        "Все отдельные замеры и контрольные суммы масок находятся в samples.jsonl.",
        "В comparison.png показана первая тестовая четвёрка.",
        "",
        "Методика замеров GPU описана в",
        "[документации PyTorch](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution).",
        "Многоплоскостные итеративные методы рассмотрены в работе",
        "[Realistic Defocus Blur for Multiplane Computer-Generated Holography](https://arxiv.org/abs/2205.07030).",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (output / "per_plane.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["method", "distance_mm", "pearson", "ssim",
                                                 "depth_identification_accuracy"])
        writer.writeheader()
        for method, score in summary.items():
            writer.writerows({"method": method, **plane} for plane in score["per_plane"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/div2k"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/phase_unet/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/benchmark"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if min(args.timing_repeats, args.warmup) < 1:
        parser.error("Repeat and warmup counts must be positive.")
    if args.output.exists():
        parser.error("Choose a new output directory.")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if state.get("refinement_steps") != 6:
        parser.error("This comparison is defined for the six-step trained model.")
    model = PhaseUNet(state["width"], state["refinement_steps"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    gs = MultiPlaneGS().to(device)
    objective = AmplitudeObjective().to(device)
    torch.testing.assert_close(gs.target_phasor, model.target_phasor, atol=0, rtol=0)
    dataset = CachedTargets(args.data, "test")
    samples = [json.loads(line) for line in (args.data / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    methods = {
        "gs_10": lambda target: gs(target, 10),
        "gs_100": lambda target: gs(target, 100),
        "neural": model,
        "hybrid": lambda target: gs(target, 10, initial_phase=model(target)),
    }
    args.output.mkdir(parents=True)
    for method in methods:
        (args.output / "masks" / method).mkdir(parents=True)
    config = {
        "status": "running", "samples": len(dataset), "split": "test",
        "manifest_sha256": dataset.fingerprint,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_epoch": state["epoch"],
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": np.__version__, "Pillow": pillow_version, "cuda_runtime": torch.version.cuda,
        "batch_size": 1, "dtype": "float32", "seed": 42,
        "torch_cpu_threads": torch.get_num_threads(),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "timing_repeats": args.timing_repeats, "warmup_calls_per_method": args.warmup,
        "timing": "Synchronized wall time for synthesis and uint8 encoding on device",
        "methods": {"gs_10": {"gs_iterations": 10}, "gs_100": {"gs_iterations": 100},
                    "neural": {"internal_optical_steps": 6},
                    "hybrid": {"internal_optical_steps": 6, "extra_gs_iterations": 10}},
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    records = []
    with torch.inference_mode(), (args.output / "samples.jsonl").open("w", encoding="utf-8") as log:
        warmup_target = normalized_batch(dataset[0][None], device)
        for method in methods.values():
            for _ in range(args.warmup):
                encode_phase(method(warmup_target))
        synchronize(device)
        names = list(methods)
        for index, sample in enumerate(samples):
            target = normalized_batch(dataset[index][None], device)
            timings = {name: [] for name in names}
            encoded = {}
            for repetition in range(args.timing_repeats):
                offset = (index + repetition) % len(names)
                for name in names[offset:] + names[:offset]:
                    synchronize(device)
                    start = time.perf_counter()
                    pixels = encode_phase(methods[name](target))
                    synchronize(device)
                    timings[name].append((time.perf_counter() - start) * 1000)
                    encoded[name] = pixels[0].cpu().numpy()
            for name in names:
                mask_path = args.output / "masks" / name / f"{sample['id']}.png"
                Image.fromarray(encoded[name]).save(mask_path)
                # Metrics start from the actual saved artifact, not the pre-export tensor.
                saved_pixels = load_grayscale_png(mask_path)
                phase = torch.from_numpy(saved_pixels.astype(np.float32)).to(device)
                phase = phase * (2 * torch.pi / 255) - torch.pi
                reconstructed = model.optics(phase[None])
                ssim, correlation = objective.metrics(reconstructed, target)
                _, correct = objective.depth_separation(reconstructed, target)
                record = {
                    "sample": sample["id"], "method": name, "images": sample["images"],
                    "mask": mask_path.relative_to(args.output).as_posix(),
                    "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                    "latency_ms": timings[name],
                    "ssim": ssim[0].cpu().tolist(), "pearson": correlation[0].cpu().tolist(),
                    "correct_depth": correct[0].cpu().tolist(),
                }
                records.append(record)
                log.write(json.dumps(record) + "\n")
            log.flush()
            if (index + 1) % 25 == 0:
                print(f"Compared {index + 1}/{len(samples)} quadruplets", flush=True)

    summary = summarize(records)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    example_scores = {}
    first = samples[0]
    for name in methods:
        example_scores[name] = evaluate(
            args.output / "masks" / name / f"{first['id']}.png",
            [args.data / path for path in first["images"]], args.output / "example" / name,
        )
        reference = next(record for record in records if record["sample"] == first["id"] and record["method"] == name)
        np.testing.assert_allclose([score["ssim"] for score in example_scores[name]], reference["ssim"], atol=2e-5)
        np.testing.assert_allclose([score["pearson"] for score in example_scores[name]], reference["pearson"], atol=2e-5)
    save_comparison(args.output, first, args.data, example_scores)
    write_report(args.output, summary, config)
    config["status"] = "complete"
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
