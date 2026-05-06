#!/usr/bin/env python3
"""
Run Google Cloud Vision on benchmark samples.

Usage:
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
    python scripts/run_google.py sroie
    python scripts/run_google.py funsd
    python scripts/run_google.py all
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_benchmark_2025.datasets import DATASETS

try:
    from google.cloud import vision
except ImportError:
    print("Error: google-cloud-vision not installed.")
    print("Run: pip install google-cloud-vision")
    sys.exit(1)

BASE = Path(__file__).parent.parent


def ocr_file(file_path: Path, client: vision.ImageAnnotatorClient) -> str:
    """Call Google Cloud Vision API on a file."""
    with file_path.open("rb") as f:
        content = f.read()

    image = vision.Image(content=content)

    # Use document_text_detection for dense text/documents
    response = client.document_text_detection(image=image)

    if response.error.message:
        raise Exception(f"{response.error.message}")

    return response.full_text_annotation.text


def run_ocr(dataset: str, samples: list[str], client: vision.ImageAnnotatorClient):
    """Run Google Cloud Vision on specified samples."""
    samples_dir = BASE / f"datasets/{dataset}/samples"
    results_dir = BASE / f"results/google_cloud/{dataset}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Get existing results
    existing = {p.stem for p in results_dir.glob("*.txt")}
    to_process = [s for s in samples if s not in existing]

    print(f"\n{dataset.upper()}: {len(to_process)} samples to process ({len(existing)} existing)")

    for sample_id in to_process:
        # Find image file
        img_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            candidate = samples_dir / f"{sample_id}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if not img_path:
            print(f"  Image not found for {sample_id}")
            continue

        print(f"  Processing {sample_id}...")
        try:
            text = ocr_file(img_path, client)
            out_path = results_dir / f"{sample_id}.txt"
            out_path.write_text(text, encoding="utf-8")
            print(f"    Wrote {len(text)} chars")
        except Exception as e:
            print(f"    Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Google Cloud Vision on benchmark samples",
        epilog="Set GOOGLE_APPLICATION_CREDENTIALS to your JSON key path",
    )
    parser.add_argument(
        "dataset",
        choices=["sroie", "funsd", "all"],
        nargs="?",
        default="sroie",
        help="Dataset to process (default: sroie)",
    )
    args = parser.parse_args()

    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        print("Error: Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
        sys.exit(1)

    client = vision.ImageAnnotatorClient()

    if args.dataset == "all":
        for ds in DATASETS:
            run_ocr(ds, DATASETS[ds], client)
    else:
        run_ocr(args.dataset, DATASETS[args.dataset], client)

    print("\nDone!")


if __name__ == "__main__":
    main()
