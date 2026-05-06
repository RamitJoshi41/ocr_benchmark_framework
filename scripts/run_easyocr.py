#!/usr/bin/env python3
"""
Run EasyOCR on benchmark samples.

Usage:
    python scripts/run_easyocr.py --dataset sroie
    python scripts/run_easyocr.py --dataset funsd
    python scripts/run_easyocr.py --dataset sroie sroie_000 sroie_001  # Specific samples
"""

import argparse
import sys
from pathlib import Path

try:
    import easyocr
except ImportError:
    print("EasyOCR not installed. Install with: pip install easyocr")
    sys.exit(1)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_benchmark_2025.config import settings
from ocr_benchmark_2025.datasets import DATASETS, discover_samples


def run_ocr(dataset: str, samples: list[str]):
    """Run EasyOCR on specified samples."""
    samples_dir = settings.DATA_DIR / f"{dataset}/samples"
    results_dir = settings.RESULTS_DIR / f"easyocr/{dataset}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading EasyOCR model (this takes 30-60 seconds)...")
    reader = easyocr.Reader(["en"], gpu=True)
    print("Model loaded!")

    for sample_id in samples:
        # Try common extensions
        img_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = samples_dir / f"{sample_id}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if not img_path:
            print(f"Image not found for {sample_id}")
            continue

        print(f"Processing {sample_id}...")
        try:
            results = reader.readtext(str(img_path))
            lines = [text for _, text, _ in results]

            output_path = results_dir / f"{sample_id}.txt"
            output_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"  Wrote {len(lines)} lines to {output_path}")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EasyOCR on benchmark samples")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        required=True,
        help="Dataset to process",
    )
    parser.add_argument(
    	"--all",
    	action="store_true",
    	help="Run on ALL prepared samples (not just the 25-sample default)",
    )
    parser.add_argument("samples", nargs="*", help="Specific sample IDs (optional)")
    args = parser.parse_args()

    if args.samples:
        samples = args.samples
    elif args.all:
        samples = discover_samples(args.dataset)
    else:
        samples = DATASETS[args.dataset]
    run_ocr(args.dataset, samples)

    print("\nDone!")
