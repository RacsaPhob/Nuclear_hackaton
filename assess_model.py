"""Evaluate a saved neural DOE model on a prepared dataset split."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import PhaseUNet
from optics_torch import AmplitudeObjective
from train import CachedTargets, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/phase_unet/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/div2k"))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model = PhaseUNet(state["width"], state.get("refinement_steps", 0)).to(args.device)
    state["model"].setdefault("refinement_logits", model.refinement_logits.detach())
    model.load_state_dict(state["model"])
    dataset = CachedTargets(args.data, args.split)
    scores = validate(model, DataLoader(dataset, batch_size=32),
                      AmplitudeObjective().to(args.device), torch.device(args.device))
    scores.update({"split": args.split, "checkpoint": str(args.checkpoint),
                   "epoch": state["epoch"], "manifest_sha256": dataset.fingerprint})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
