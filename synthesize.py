"""Generate a DOE with the trained network and evaluate its saved PNG."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from evaluate import evaluate, load_grayscale_png
from metrics import normalize_amplitude
from model import PhaseUNet
from baselines import MultiPlaneGS


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "weights" / "model.pt"


def mean_metric(values):
    if any(value is None for value in values):
        return None
    return float(np.mean(values))


def generate_mask(targets, model, gs, args):
    target_tensor = torch.from_numpy(targets.astype(np.float32))[None].to(args.device)
    if target_tensor.is_cuda:
        torch.cuda.synchronize(target_tensor.device)
    start = time.perf_counter()
    with torch.inference_mode():
        if args.method == "gs":
            phase = gs(target_tensor, args.gs_iterations)
        else:
            phase = model(target_tensor)
            if args.method == "hybrid":
                phase = gs(target_tensor, args.refine_iterations, initial_phase=phase)
        phase = phase[0].cpu().numpy()
    pixels = np.rint((phase + np.pi) * (255 / (2 * np.pi))).clip(0, 255).astype(np.uint8)
    return pixels, (time.perf_counter() - start) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--targets", type=Path, nargs=4,
                        help="Four PNGs for 24.1, 25.3, 26.5, 27.7 mm, in this order")
    inputs.add_argument("--sets", type=Path,
                        help="Directory of sets, each containing four PNGs in filename order")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--method", choices=("neural", "gs", "hybrid"), default="neural")
    parser.add_argument("--gs-iterations", type=int, default=100)
    parser.add_argument("--refine-iterations", type=int, default=10)
    args = parser.parse_args()
    if min(args.gs_iterations, args.refine_iterations) < 0:
        parser.error("Iteration counts must be nonnegative.")
    if args.output.exists():
        parser.error("Choose a new output directory, for example /output/run_01.")
    started = time.perf_counter()
    torch.set_num_threads(4)
    try:
        cases = [(None, args.targets)]
        if args.sets is not None:
            cases = [
                (folder.name, sorted(path for path in folder.iterdir() if path.suffix.lower() == ".png"))
                for folder in sorted(args.sets.iterdir()) if folder.is_dir()
            ]
        if not cases:
            raise ValueError("The sets directory contains no image sets.")
        prepared = []
        for name, paths in cases:
            if len(paths) != 4:
                raise ValueError(f"{name}: expected exactly four PNGs, found {len(paths)}.")
            targets = np.stack([normalize_amplitude(load_grayscale_png(path)) for path in paths])
            prepared.append((name, paths, targets))
    except (OSError, ValueError) as error:
        parser.error(str(error))

    gs = MultiPlaneGS().to(args.device)
    model = None
    load_started = time.perf_counter()
    if args.method != "gs":
        state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        model = PhaseUNet(state["width"], state.get("refinement_steps", 0)).to(args.device)
        state["model"].setdefault("refinement_logits", model.refinement_logits.detach())
        model.load_state_dict(state["model"])
        model.eval()
    load_ms = (time.perf_counter() - load_started) * 1000
    results = []
    for name, paths, targets in prepared:
        output = args.output / name if name is not None else args.output
        pixels, elapsed_ms = generate_mask(targets, model, gs, args)
        output.mkdir(parents=True)
        phase_path = output / "doe.png"
        Image.fromarray(pixels).save(phase_path)
        scores = evaluate(phase_path, paths, output / "reconstruction")
        results.append({
            "set": name or output.name, "doe": phase_path.relative_to(args.output).as_posix(),
            "synthesis_ms": elapsed_ms,
            "ssim": mean_metric([score["ssim"] for score in scores]),
            "pearson": mean_metric([score["pearson"] for score in scores]),
            "per_plane": scores,
        })
    report = {
        "set_count": len(results), "method": args.method, "device": args.device,
        "checkpoint": str(args.checkpoint) if model is not None else None,
        "model_load_ms": load_ms, "total_seconds": time.perf_counter() - started,
        "ssim": mean_metric([result["ssim"] for result in results]),
        "pearson": mean_metric([result["pearson"] for result in results]),
        "mean_synthesis_ms": mean_metric([result["synthesis_ms"] for result in results]),
        "sets": results,
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report if args.sets is not None else results[0]["per_plane"], indent=2))


if __name__ == "__main__":
    main()
