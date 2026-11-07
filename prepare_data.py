"""Prepare source-disjoint DIV2K crops and four-plane image manifests."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
import zipfile

import numpy as np
from PIL import Image, ImageOps, __version__ as pillow_version

from download_div2k import ARCHIVES, BASE_URL
from fresnel import DISTANCES_MM, IMAGE_SIZE


CROPS_PER_SOURCE = 16
MIN_STANDARD_DEVIATION = 0.05
MIN_MEAN_GRADIENT = 0.01
SEED = 42


def split_source_ids(seed: int) -> dict[str, list[int]]:
    training_ids = list(range(1, 801))
    random.Random(seed).shuffle(training_ids)
    return {
        "train": sorted(training_ids[:700]),
        "validation": sorted(training_ids[700:]),
        "test": list(range(801, 901)),
    }


def content_statistics(pixels: np.ndarray) -> dict[str, float]:
    values = pixels.astype(np.float64) / 255
    gradient = (
        np.abs(np.diff(values, axis=0)).mean()
        + np.abs(np.diff(values, axis=1)).mean()
    ) / 2
    return {"standard_deviation": float(values.std()), "mean_gradient": float(gradient)}


def make_crop(image: Image.Image, rng: random.Random, index: int, augment: bool):
    width, height = image.size
    shortest = min(width, height)
    for attempt in range(100):
        if index == 0 and attempt == 0:
            side = shortest
            left, top = (width - side) // 2, (height - side) // 2
        else:
            side = rng.randint(max(IMAGE_SIZE, round(0.35 * shortest)), shortest)
            left = rng.randint(0, width - side)
            top = rng.randint(0, height - side)
        box = (left, top, left + side, top + side)
        crop = image.crop(box).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        stats = content_statistics(np.asarray(crop))
        if (
            stats["standard_deviation"] >= MIN_STANDARD_DEVIATION
            and stats["mean_gradient"] >= MIN_MEAN_GRADIENT
        ):
            flipped = augment and rng.random() < 0.5
            turns = rng.randrange(4) if augment else 0
            if flipped:
                crop = ImageOps.mirror(crop)
            if turns:
                crop = crop.rotate(90 * turns)
            return crop, {"box": box, "mirror": flipped, "quarter_turns": turns, **stats}
    raise ValueError("No sufficiently detailed crop found after 100 attempts.")


def make_quadruplets(source_ids: list[int], split: str, count: int, seed: int) -> list[dict]:
    """Cover each crop once, then add distinct ordered training combinations."""
    rng = random.Random(seed)
    combinations = []
    for crop_index in range(CROPS_PER_SOURCE):
        shuffled = source_ids.copy()
        rng.shuffle(shuffled)
        for start in range(0, len(shuffled), 4):
            combinations.append(tuple((source_id, crop_index) for source_id in shuffled[start:start + 4]))
    if count < len(combinations):
        raise ValueError(f"{split} needs at least {len(combinations)} quadruplets to cover its crops.")
    seen = set(combinations)
    while len(combinations) < count:
        combination = tuple(
            (source_id, rng.randrange(CROPS_PER_SOURCE))
            for source_id in rng.sample(source_ids, 4)
        )
        if combination not in seen:
            combinations.append(combination)
            seen.add(combination)
    rng.shuffle(combinations)
    return [
        {
            "id": f"{split}_{index:06d}",
            "images": [f"images/{split}/{source_id:04d}_{crop_index:02d}.png"
                       for source_id, crop_index in combination],
        }
        for index, combination in enumerate(combinations)
    ]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_dataset(archives: Path, output: Path, seed: int, train_quadruplets: int) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    splits = split_source_ids(seed)
    source_splits = {source_id: split for split, ids in splits.items() for source_id in ids}
    counts = {"train": train_quadruplets, "validation": 400, "test": 400}
    quadruplets = {
        split: make_quadruplets(ids, split, counts[split], seed + index)
        for index, (split, ids) in enumerate(splits.items())
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        for split in splits:
            (staging / "images" / split).mkdir(parents=True)

        source_records, crop_records = [], []
        source_hashes, crop_hashes = {}, {}
        for archive_name, source_ids in zip(ARCHIVES, (range(1, 801), range(801, 901))):
            archive_path = archives / archive_name
            with zipfile.ZipFile(archive_path) as archive:
                for source_id in source_ids:
                    member = f"{archive_name[:-4]}/{source_id:04d}.png"
                    encoded = archive.read(member)  # ZipFile verifies the member's CRC.
                    with Image.open(BytesIO(encoded)) as original:
                        image = ImageOps.exif_transpose(original).convert("L")
                    split = source_splits[source_id]
                    pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
                    if pixel_hash in source_hashes and source_hashes[pixel_hash] != split:
                        raise ValueError(f"Duplicate source image across splits: {source_id}")
                    source_hashes[pixel_hash] = split
                    source_records.append({
                        "source_id": source_id, "split": split, "archive": archive_name,
                        "member": member, "size": image.size,
                        "file_sha256": hashlib.sha256(encoded).hexdigest(),
                        "grayscale_sha256": pixel_hash,
                    })
                    rng = random.Random(f"{seed}:{source_id}")
                    for index in range(CROPS_PER_SOURCE):
                        crop, details = make_crop(image, rng, index, augment=split == "train")
                        crop_hash = hashlib.sha256(crop.tobytes()).hexdigest()
                        if crop_hash in crop_hashes:
                            raise ValueError(f"Duplicate crop encountered for source {source_id}, crop {index}.")
                        crop_hashes[crop_hash] = split
                        relative = f"images/{split}/{source_id:04d}_{index:02d}.png"
                        crop.save(staging / relative)
                        crop_records.append({
                            "image": relative, "source_id": source_id, "split": split,
                            "pixel_sha256": crop_hash, **details,
                        })
                    if source_id % 50 == 0:
                        print(f"Prepared {source_id}/900 sources", flush=True)

        for split, records in quadruplets.items():
            write_jsonl(staging / f"{split}.jsonl", records)
        write_jsonl(staging / "sources.jsonl", source_records)
        write_jsonl(staging / "crops.jsonl", crop_records)
        metadata = {
            "dataset": "DIV2K HR", "seed": seed,
            "versions": {"python": sys.version, "numpy": np.__version__, "Pillow": pillow_version},
            "source_urls": [BASE_URL + name for name in ARCHIVES],
            "terms_url": BASE_URL, "usage": "Academic research only; copyright belongs to image owners.",
            "split_policy": "700 official training images for train, 100 for validation; official validation for test.",
            "image_size": [IMAGE_SIZE, IMAGE_SIZE], "mode": "L", "bit_depth": 8,
            "crops_per_source": CROPS_PER_SOURCE,
            "crop_side_fraction": [0.35, 1.0],
            "resize": "Pillow LANCZOS",
            "train_augmentation": "Random mirror and 0/90/180/270-degree rotation",
            "min_standard_deviation": MIN_STANDARD_DEVIATION,
            "min_mean_gradient": MIN_MEAN_GRADIENT,
            "normalization_at_loading": "Divide each grayscale image by its own maximum.",
            "distances_mm": DISTANCES_MM,
            "counts": {
                split: {"sources": len(ids), "images": len(ids) * CROPS_PER_SOURCE,
                        "quadruplets": len(quadruplets[split])}
                for split, ids in splits.items()
            },
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        staging.rename(output)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/div2k"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-quadruplets", type=int, default=20000)
    args = parser.parse_args()
    metadata = prepare_dataset(args.archives, args.output, args.seed, args.train_quadruplets)
    print(json.dumps(metadata["counts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
