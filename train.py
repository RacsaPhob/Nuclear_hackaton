"""Train a physics-informed phase U-Net on prepared DIV2K quadruplets."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from evaluate import load_grayscale_png
from fresnel import DISTANCES_MM
from model import PhaseUNet
from optics_torch import AmplitudeObjective, quantize_phase


class CachedTargets(Dataset):
    def __init__(self, root, split):
        manifest = root / f"{split}.jsonl"
        self.fingerprint = hashlib.sha256(manifest.read_bytes()).hexdigest()
        records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        paths = sorted({path for record in records for path in record["images"]})
        index = {path: i for i, path in enumerate(paths)}
        self.indices = torch.tensor([[index[path] for path in record["images"]] for record in records])
        pixels = np.empty((len(paths), 128, 128), dtype=np.uint8)
        for i, path in enumerate(paths):
            pixels[i] = load_grayscale_png(root / path)
            if (i + 1) % 2000 == 0:
                print(f"Loading {split}: {i + 1}/{len(paths)} images", flush=True)
        self.pixels = torch.from_numpy(pixels)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.pixels[self.indices[index]]


def normalized_batch(pixels, device):
    values = pixels.to(device=device, dtype=torch.float32)
    return values / values.amax((-2, -1), keepdim=True).clamp_min(1)


@torch.no_grad()
def validate(model, loader, objective, device):
    model.eval()
    total = torch.zeros(5, 4, device=device)
    count = 0
    for pixels in loader:
        target = normalized_batch(pixels, device)
        phase = quantize_phase(model(target))
        reconstruction = model.optics(phase)
        ssim, correlation = objective.metrics(reconstruction, target)
        mse = (reconstruction - target).square().mean((-2, -1))
        separation, correct = objective.depth_separation(reconstruction, target)
        total += torch.stack([mse.sum(0), ssim.sum(0), correlation.sum(0),
                              separation.sum(0), correct.sum(0)])
        count += len(target)
    mse, ssim, correlation, separation, correct = (total / count).cpu().tolist()
    return {
        "samples": count,
        "mse": float(np.mean(mse)),
        "ssim": float(np.mean(ssim)),
        "pearson": float(np.mean(correlation)),
        "depth_identification_accuracy": float(np.mean(correct)),
        "crosstalk_penalty": float(np.mean(separation)),
        "loss": float(np.mean(mse) + 0.5 * (1 - np.mean(ssim))
                      + 0.25 * (1 - np.mean(correlation)) + 2 * np.mean(separation)),
        "per_plane": [
            {"distance_mm": z, "mse": m, "ssim": s, "pearson": c,
             "depth_identification_accuracy": accuracy}
            for z, m, s, c, accuracy in zip(DISTANCES_MM, mse, ssim, correlation, correct)
        ],
        "phase_quantized_to_8bit": True,
    }


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/div2k"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase_unet"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--refinement-steps", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.width) < 1 or args.learning_rate <= 0 or args.refinement_steps < 0:
        parser.error("Epochs, batch size, width and learning rate must be positive.")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = PhaseUNet(args.width, args.refinement_steps).to(device)
    objective = AmplitudeObjective().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    epoch_start, best_loss = 0, float("inf")
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.output / "last.pt", map_location=device, weights_only=True)
        if checkpoint.get("objective_version", 1) != 2:
            parser.error("This checkpoint used an earlier loss. Start a new run for the depth-separation objective.")
        if (checkpoint["width"] != args.width or checkpoint["seed"] != args.seed
                or checkpoint.get("refinement_steps", 0) != args.refinement_steps):
            parser.error("Resume requires the same model width, refinement steps, and seed.")
        checkpoint["model"].setdefault("refinement_logits", model.refinement_logits.detach())
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch_start, best_loss = checkpoint["epoch"], checkpoint["best_loss"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
    else:
        args.output.mkdir(parents=True)

    train_data = CachedTargets(args.data, "train")
    validation_data = CachedTargets(args.data, "validation")
    fingerprints = {"train": train_data.fingerprint, "validation": validation_data.fingerprint}
    if checkpoint is not None and checkpoint["data_fingerprints"] != fingerprints:
        raise ValueError("The dataset manifests differ from the resumed run.")
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size)
    if checkpoint is None:
        initial = validate(model, validation_loader, objective, device)
        (args.output / "initial_validation.json").write_text(json.dumps(initial, indent=2), encoding="utf-8")
        config = {**vars(args), "data": str(args.data.resolve()), "output": str(args.output.resolve()),
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "torch_version": str(torch.__version__), "data_fingerprints": fingerprints,
                  "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  "objective": "MSE + 0.5*(1-SSIM) + 0.25*(1-Pearson) + 2*depth_separation",
                  "depth_separation_margin": 0.25}
        (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(f"Initial validation: {json.dumps(initial)}", flush=True)
    start = time.monotonic()
    for epoch in range(epoch_start, args.epochs):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=generator)
        model.train()
        train_loss, count = 0.0, 0
        for batch_index, pixels in enumerate(loader):
            target = normalized_batch(pixels, device)
            phase = quantize_phase(model(target), straight_through=True)
            restored = model.optics(phase)
            loss, _, _ = objective(restored, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item() * len(target)
            count += len(target)
            if (batch_index + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{args.epochs}, batch {batch_index + 1}/{len(loader)}, loss {train_loss / count:.5f}", flush=True)
        scores = validate(model, validation_loader, objective, device)
        scheduler.step(scores["loss"])
        improved = scores["loss"] < best_loss
        best_loss = min(best_loss, scores["loss"])
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch + 1, "best_loss": best_loss,
            "width": args.width, "seed": args.seed, "data_fingerprints": fingerprints,
            "refinement_steps": args.refinement_steps,
            "objective_version": 2,
            "validation": scores,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }
        save_checkpoint(args.output / "last.pt", state)
        if improved:
            save_checkpoint(args.output / "best.pt", state)
            (args.output / "best_validation.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
        record = {"epoch": epoch + 1, "train_loss": train_loss / count,
                  "validation": scores, "learning_rate": optimizer.param_groups[0]["lr"],
                  "elapsed_seconds_this_session": time.monotonic() - start}
        with (args.output / "history.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(record) + "\n")
        print(f"Epoch {epoch + 1}: validation SSIM {scores['ssim']:.5f}, Pearson {scores['pearson']:.5f}, loss {scores['loss']:.5f}", flush=True)


if __name__ == "__main__":
    main()
