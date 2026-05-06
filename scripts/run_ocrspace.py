#!/usr/bin/env python3
"""
Run OCR.space on benchmark samples.

Usage:
    export OCR_SPACE_API_KEY=your_api_key

    # Default 25-sample subset
    python scripts/run_ocrspace.py sroie
    python scripts/run_ocrspace.py funsd
    python scripts/run_ocrspace.py all

    # Full prepared dataset
    python scripts/run_ocrspace.py sroie --all
    python scripts/run_ocrspace.py all --all

    # Specific samples
    python scripts/run_ocrspace.py sroie --samples sroie_X00016469670 sroie_X00016469671

Get your API key at: https://ocr.space/ocrapi
"""

import argparse
import sys
import time
from pathlib import Path

import requests

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_benchmark_2025.config import settings
from ocr_benchmark_2025.datasets import DATASETS, discover_samples


# OCR.space free tier hard limit per file
OCR_SPACE_MAX_BYTES = 1024 * 1024  # 1 MB


def ocr_file(file_path: Path, api_key: str, engine: int = 2) -> str:
    """Call OCR.space API on a file.

    Args:
        file_path: Path to image
        api_key: OCR.space API key
        engine: OCR engine (1=fast/free-friendly, 2=better accuracy)
    """
    url = "https://api.ocr.space/parse/image"

    with file_path.open("rb") as f:
        response = requests.post(
            url,
            files={"file": f},
            data={
                "apikey": api_key,
                "language": "eng",
                "OCREngine": engine,
                "scale": "true",
                "isTable": "true",  # better for receipts/forms
            },
            timeout=120,
        )

    result = response.json()
    if result.get("IsErroredOnProcessing"):
        msg = result.get("ErrorMessage", "Unknown error")
        if isinstance(msg, list):
            msg = " | ".join(msg)
        raise RuntimeError(msg)

    parsed = result.get("ParsedResults", [])
    if parsed:
        return parsed[0].get("ParsedText", "")
    return ""


def run_ocr(dataset: str, samples: list[str], api_key: str, sleep_seconds: float = 1.5):
    """Run OCR.space on specified samples."""
    samples_dir = settings.DATA_DIR / f"{dataset}/samples"
    results_dir = settings.RESULTS_DIR / f"ocrspace/{dataset}"
    results_dir.mkdir(parents=True, exist_ok=True)

    existing = {p.stem for p in results_dir.glob("*.txt")}
    to_process = [s for s in samples if s not in existing]

    print(
        f"\n{dataset.upper()}: {len(to_process)} samples to process "
        f"({len(existing)} already done out of {len(samples)} requested)"
    )

    failures = 0
    for i, sample_id in enumerate(to_process, 1):
        img_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            candidate = samples_dir / f"{sample_id}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if not img_path:
            print(f"  [{i}/{len(to_process)}] Image not found for {sample_id}")
            continue

        # Skip files too large for the free tier
        size = img_path.stat().st_size
        if size > OCR_SPACE_MAX_BYTES:
            print(
                f"  [{i}/{len(to_process)}] Skipping {sample_id}: "
                f"{size / 1024:.0f} KB exceeds 1 MB free-tier limit"
            )
            failures += 1
            continue

        print(f"  [{i}/{len(to_process)}] Processing {sample_id} ({size / 1024:.0f} KB)...")
        try:
            text = ocr_file(img_path, api_key)
            out_path = results_dir / f"{sample_id}.txt"
            out_path.write_text(text, encoding="utf-8")
            print(f"    Wrote {len(text)} chars")
        except Exception as e:
            print(f"    Error: {e}")
            failures += 1

        # Rate limit
        time.sleep(sleep_seconds)

    print(f"  Finished {dataset}: {len(to_process) - failures} OK, {failures} failed/skipped")


def main():
    parser = argparse.ArgumentParser(
        description="Run OCR.space on benchmark samples",
        epilog="Get your API key at: https://ocr.space/ocrapi",
    )
    parser.add_argument(
        "dataset",
        choices=["sroie", "funsd", "all"],
        nargs="?",
        default="sroie",
        help="Dataset to process (default: sroie)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="run_all_samples",
        help="Run on ALL prepared samples (not just the 25-sample default)",
    )
    parser.add_argument(
        "--samples",
        "-s",
        nargs="*",
        help="Specific sample IDs to process (overrides --all)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.5,
        help="Seconds to sleep between API calls (default: 1.5)",
    )
    args = parser.parse_args()

    api_key = settings.OCR_SPACE_API_KEY
    if not api_key:
        print("Error: Set OCR_SPACE_API_KEY environment variable (or in .env)")
        print("Get your API key at: https://ocr.space/ocrapi")
        sys.exit(1)

    def resolve_samples(ds: str) -> list[str]:
        if args.samples:
            return args.samples
        if args.run_all_samples:
            return discover_samples(ds)
        return DATASETS[ds]

    if args.dataset == "all":
        if args.samples:
            print("Error: --samples cannot be combined with dataset 'all'")
            sys.exit(1)
        for ds in DATASETS:
            run_ocr(ds, resolve_samples(ds), api_key, sleep_seconds=args.sleep)
    else:
        run_ocr(args.dataset, resolve_samples(args.dataset), api_key, sleep_seconds=args.sleep)

    print("\nDone!")


if __name__ == "__main__":
    main()
