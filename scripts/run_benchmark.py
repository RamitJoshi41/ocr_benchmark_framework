#!/usr/bin/env python3
"""
OCR Benchmark Tool - Calculate CER/WER/BoW for OCR results against ground truth.

Usage:
    python scripts/run_benchmark.py --dataset sroie
    python scripts/run_benchmark.py --dataset funsd --tokenize
    python scripts/run_benchmark.py --dataset sroie --output custom_results.json
    python scripts/run_benchmark.py --help

Metrics:
    CER (Character Error Rate): Levenshtein distance on characters / total chars
    WER (Word Error Rate): Levenshtein distance on words / total words
    BoW (Bag of Words Error): Order-independent word frequency comparison

Lower scores = better accuracy
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_benchmark_2025.config import settings
from ocr_benchmark_2025.metrics import (
    calculate_bow_error,
    calculate_cer,
    calculate_wer,
    normalize_text,
    tokenize_text,
)


def load_text(path: Path) -> str:
    """Load and normalize text from file."""
    return normalize_text(path.read_text(encoding="utf-8"))


def find_results(results_dir: Path, document_name: str, dataset: str | None = None) -> dict:
    """Find OCR result files for a document across different tools."""
    tools = {}

    tool_names = {
        "ocrspace": "OCR.space",
        "easyocr": "EasyOCR",
        "paddleocr": "PaddleOCR",
        "google_cloud": "Google Cloud Vision",
        "tesseract":"Tesseract",
        "doctr":"docTR",
    }

    for subdir, display_name in tool_names.items():
        if dataset:
            result_path = results_dir / subdir / dataset / f"{document_name}.txt"
        else:
            result_path = results_dir / subdir / f"{document_name}.txt"

        if result_path.exists():
            tools[display_name] = result_path

    return tools


def run_dataset_benchmark(
    dataset: str, results_dir: Path, case_insensitive: bool = True, tokenize: bool = False
) -> dict:
    """Run benchmark on a standard dataset (FUNSD or SROIE)."""
    datasets = {
        "funsd": {
            "samples_dir": settings.DATA_DIR / "funsd/samples",
            "gt_pattern": "*_gt.txt",
            "name": "FUNSD Forms",
        },
        "sroie": {
            "samples_dir": settings.DATA_DIR / "sroie/samples",
            "gt_pattern": "*_gt.txt",
            "name": "SROIE Receipts",
        },
    }

    if dataset not in datasets:
        print(f"Error: Unknown dataset '{dataset}'. Available: {list(datasets.keys())}")
        sys.exit(1)

    config = datasets[dataset]
    samples_dir = config["samples_dir"]

    gt_files = sorted(samples_dir.glob(config["gt_pattern"]))
    if not gt_files:
        print(f"Error: No ground truth files found in {samples_dir}")
        print("Did you run scripts/prepare_datasets.py first?")
        sys.exit(1)

    mode_str = "tokenized" if tokenize else "case-insensitive" if case_insensitive else "raw"
    print("=" * 70)
    print(f"{config['name'].upper()} BENCHMARK ({len(gt_files)} samples, {mode_str})")
    print("=" * 70)

    tool_scores: dict = {}

    for gt_path in gt_files:
        sample_name = gt_path.stem.replace("_gt", "")
        gt_text = load_text(gt_path)
        tool_results = find_results(results_dir, sample_name, dataset=dataset)

        for tool_name, result_path in tool_results.items():
            if tool_name not in tool_scores:
                tool_scores[tool_name] = []

            ocr_text = load_text(result_path)

            if tokenize:
                gt_cmp = tokenize_text(gt_text)
                ocr_cmp = tokenize_text(ocr_text)
            elif case_insensitive:
                gt_cmp = gt_text.lower()
                ocr_cmp = ocr_text.lower()
            else:
                gt_cmp = gt_text
                ocr_cmp = ocr_text

            cer = calculate_cer(gt_cmp, ocr_cmp)
            wer = calculate_wer(gt_cmp, ocr_cmp)
            bow = calculate_bow_error(gt_cmp, ocr_cmp)
            tool_scores[tool_name].append(
                {"sample": sample_name, "cer": cer, "wer": wer, "bow": bow}
            )

    print(f"{'Tool':<28} {'Avg CER':>12} {'Avg WER':>12} {'Avg BoW':>12} {'Samples':>10}")
    print("-" * 80)

    summary = {}
    for tool_name, scores in sorted(tool_scores.items()):
        avg_cer = sum(s["cer"] for s in scores) / len(scores)
        avg_wer = sum(s["wer"] for s in scores) / len(scores)
        avg_bow = sum(s["bow"] for s in scores) / len(scores)
        summary[tool_name] = {
            "avg_cer": avg_cer,
            "avg_wer": avg_wer,
            "avg_bow": avg_bow,
            "samples": len(scores),
        }
        cer_pct = avg_cer * 100
        wer_pct = avg_wer * 100
        bow_pct = avg_bow * 100
        n = len(scores)
        print(f"{tool_name:<28} {cer_pct:>11.1f}% {wer_pct:>11.1f}% {bow_pct:>11.1f}% {n:>10}")

    print("-" * 80)
    print(f"Lower = Better | CER: chars | WER: words | BoW: order-independent (mode: {mode_str})")
    print()

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="OCR Benchmark Tool - Calculate CER/WER metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--results",
        "-r",
        type=Path,
        default=settings.RESULTS_DIR,
        help="Path to results directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["funsd", "sroie"],
        required=True,
        help="Dataset to benchmark",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json"],
        default="table",
        help="Output format",
    )
    parser.add_argument(
        "--tokenize",
        "-t",
        action="store_true",
        help="Normalize to word tokens before comparison (recommended)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=settings.RESULTS_DIR / "benchmark_results.json",
        help="Path to save benchmark results (JSON)",
    )

    args = parser.parse_args()

    summary = run_dataset_benchmark(args.dataset, args.results, tokenize=args.tokenize)

    # Save results to file
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {args.output}")

    if args.format == "json":
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
