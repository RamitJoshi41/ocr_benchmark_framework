#!/usr/bin/env python3
"""
Prepare SROIE and FUNSD datasets for benchmarking.

Requires manual download of datasets due to licensing restrictions.

Usage:
    # Default: 25 samples per dataset (matches the original benchmark)
    python scripts/prepare_datasets.py

    # Full datasets (all available samples)
    python scripts/prepare_datasets.py --full

    # Custom limit
    python scripts/prepare_datasets.py --limit 100

    # Per-dataset limits
    python scripts/prepare_datasets.py --sroie-limit 200 --funsd-limit 50

    # Force re-prepare even if samples exist
    python scripts/prepare_datasets.py --full --force
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_benchmark_2025.config import settings


def extract_sroie_gt(gt_path: Path) -> str:
    """Extract text from SROIE format (bbox + text)."""
    lines = []
    with gt_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 9:
                text = ",".join(parts[8:])
                lines.append(text)
    return "\n".join(lines)


def extract_funsd_gt(json_path: Path) -> str:
    """Extract text from FUNSD JSON format.

    Groups words into lines by sorting semantic entities by vertical position (y1).
    """
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    entities = []
    for item in data.get("form", []):
        text = item.get("text", "").strip()
        box = item.get("box", [])
        if text and len(box) == 4:
            entities.append({"text": text, "y1": box[1], "x1": box[0]})

    # Sort by Y position (line by line), then X position (left to right)
    entities.sort(key=lambda e: (e["y1"] // 5, e["x1"]))

    return "\n".join([e["text"] for e in entities])


def prepare_sroie(limit: int | None = None, force: bool = False):
    """Prepare SROIE dataset.

    Args:
        limit: Max number of samples to copy. None = all available.
        force: If True, re-process even if samples already exist.
    """
    label = f"{limit} samples" if limit else "ALL samples"
    print(f"\n=== Preparing SROIE ({label}) ===")

    settings.SROIE_SAMPLES_PATH.mkdir(parents=True, exist_ok=True)
    existing = {p.stem.replace("_gt", "") for p in settings.SROIE_SAMPLES_PATH.glob("*_gt.txt")}

    if not force and limit is not None and len(existing) >= limit:
        print(f"Already have {len(existing)} samples (>= limit {limit}).")
        return sorted(existing)[:limit]

    local_img_dir = settings.SROIE_FULL_PATH / "images"
    if not local_img_dir.exists() or not any(local_img_dir.iterdir()):
        print(f"Error: Local SROIE dataset not found at {settings.SROIE_FULL_PATH}")
        print("Action Required: Manual Download")
        print("1. Go to https://rrc.cvc.uab.es/?ch=13 (Task 1)")
        print("2. Register/Login and accept the license.")
        print("3. Download the dataset.")
        print(f"4. Extract to: {settings.SROIE_FULL_PATH}")
        print("   Expected structure: datasets/sroie_full/images/000.jpg")
        return []

    print(f"Processing local dataset from {settings.SROIE_FULL_PATH}")
    available = [p.stem for p in local_img_dir.glob("*.jpg")]
    available.sort(
        key=lambda x: int("".join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else x
    )

    added = 0
    skipped = 0
    for raw_id in available:
        sample_id = raw_id if raw_id.startswith("sroie_") else f"sroie_{raw_id}"

        if not force and sample_id in existing:
            skipped += 1
            continue
        if limit is not None and added >= limit:
            break

        src_img = settings.SROIE_FULL_PATH / f"images/{raw_id}.jpg"
        dst_img = settings.SROIE_SAMPLES_PATH / f"{sample_id}.jpg"

        if not src_img.exists():
            continue

        shutil.copy(src_img, dst_img)

        gt_path = settings.SROIE_FULL_PATH / f"ground_truth/{raw_id}.txt"
        if gt_path.exists():
            gt_text = extract_sroie_gt(gt_path)
            sample_gt_path = settings.SROIE_SAMPLES_PATH / f"{sample_id}_gt.txt"
            sample_gt_path.write_text(gt_text, encoding="utf-8")
            # Print sparingly when going full-scale
            if added < 10 or added % 50 == 0:
                print(f"  Added {sample_id} ({len(gt_text)} chars)")
            added += 1

    final_samples = sorted(
        [p.stem.replace("_gt", "") for p in settings.SROIE_SAMPLES_PATH.glob("*_gt.txt")]
    )
    print(f"Added {added} new samples, skipped {skipped} existing.")
    print(f"Total SROIE samples on disk: {len(final_samples)}")
    return final_samples[:limit] if limit else final_samples


def prepare_funsd(limit: int | None = None, force: bool = False):
    """Prepare FUNSD dataset.

    Args:
        limit: Max number of samples to copy. None = all available.
        force: If True, re-process even if samples already exist.
    """
    label = f"{limit} samples" if limit else "ALL samples"
    print(f"\n=== Preparing FUNSD ({label}) ===")

    settings.FUNSD_SAMPLES_PATH.mkdir(parents=True, exist_ok=True)
    existing = {p.stem.replace("_gt", "") for p in settings.FUNSD_SAMPLES_PATH.glob("*_gt.txt")}

    if not force and limit is not None and len(existing) >= limit:
        print(f"Already have {len(existing)} samples (>= limit {limit}).")
        return sorted(existing)[:limit]

    if not settings.FUNSD_FULL_PATH.exists():
        print(f"Error: FUNSD dataset not found at {settings.FUNSD_FULL_PATH}")
        print("Action Required: Manual Download")
        print("1. Go to https://guillaumejaume.github.io/FUNSD/download/")
        print("2. Read and agree to the license.")
        print("3. Download the dataset.")
        print(f"4. Extract to: {settings.FUNSD_FULL_PATH}")
        return []

    available = [p.stem for p in (settings.FUNSD_FULL_PATH / "images").glob("*.png")]
    available.sort(key=lambda x: (int(x), "") if x.isdigit() else (float("inf"), x))

    added = 0
    skipped = 0
    for sample_id in available:
        if not force and sample_id in existing:
            skipped += 1
            continue
        if limit is not None and added >= limit:
            break

        src_img = settings.FUNSD_FULL_PATH / f"images/{sample_id}.png"
        dst_img = settings.FUNSD_SAMPLES_PATH / f"{sample_id}.png"
        if not src_img.exists():
            continue

        shutil.copy(src_img, dst_img)

        json_path = settings.FUNSD_FULL_PATH / f"annotations/{sample_id}.json"
        if json_path.exists():
            gt_text = extract_funsd_gt(json_path)
            gt_path = settings.FUNSD_SAMPLES_PATH / f"{sample_id}_gt.txt"
            gt_path.write_text(gt_text, encoding="utf-8")
            if added < 10 or added % 25 == 0:
                print(f"  Added {sample_id} ({len(gt_text)} chars)")
            added += 1

    final_samples = sorted(
        [p.stem.replace("_gt", "") for p in settings.FUNSD_SAMPLES_PATH.glob("*_gt.txt")]
    )
    print(f"Added {added} new samples, skipped {skipped} existing.")
    print(f"Total FUNSD samples on disk: {len(final_samples)}")
    return final_samples[:limit] if limit else final_samples


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare SROIE and FUNSD datasets for benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Prepare ALL available samples (overrides --limit)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            f"Number of samples per dataset "
            f"(default: {settings.SROIE_SAMPLE_COUNT} SROIE, "
            f"{settings.FUNSD_SAMPLE_COUNT} FUNSD)"
        ),
    )
    parser.add_argument(
        "--sroie-limit",
        type=int,
        default=None,
        help="Override limit for SROIE only",
    )
    parser.add_argument(
        "--funsd-limit",
        type=int,
        default=None,
        help="Override limit for FUNSD only",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process and overwrite existing samples",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Resolve effective limits
    if args.full:
        sroie_limit = None
        funsd_limit = None
    else:
        default_sroie = args.limit if args.limit is not None else settings.SROIE_SAMPLE_COUNT
        default_funsd = args.limit if args.limit is not None else settings.FUNSD_SAMPLE_COUNT
        sroie_limit = args.sroie_limit if args.sroie_limit is not None else default_sroie
        funsd_limit = args.funsd_limit if args.funsd_limit is not None else default_funsd

    sroie_samples = prepare_sroie(limit=sroie_limit, force=args.force)
    funsd_samples = prepare_funsd(limit=funsd_limit, force=args.force)

    print("\n=== Summary ===")
    print(f"SROIE: {len(sroie_samples)} samples")
    print(f"FUNSD: {len(funsd_samples)} samples")
    print(f"Total: {len(sroie_samples) + len(funsd_samples)} samples")
