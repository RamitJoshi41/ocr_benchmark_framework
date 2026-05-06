#!/usr/bin/env python3
"""
Run docTR OCR on images, PDFs, or benchmark datasets.

Usage:
    # Single file (image or PDF)
    python scripts/run_doctr.py --file document.pdf
    python scripts/run_doctr.py --file image.png --output result.txt

    # Benchmark datasets
    python scripts/run_doctr.py --dataset sroie
    python scripts/run_doctr.py --dataset funsd
    python scripts/run_doctr.py --dataset sroie --samples sroie_X00016469670

Notes:
    docTR has native PDF support — no PyMuPDF conversion needed.
    Uses GPU automatically when available; pass --cpu to force CPU mode.
"""

import argparse
import sys
from pathlib import Path

try:
    import torch
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
except ImportError:
    print("docTR not installed. Install with: uv sync --all-extras")
    sys.exit(1)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_benchmark_2025.config import settings
from ocr_benchmark_2025.datasets import DATASETS, discover_samples

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
PDF_EXTENSIONS = {".pdf"}


def build_predictor(use_gpu: bool):
    """Build a docTR OCR predictor (det + reco)."""
    predictor = ocr_predictor(
        det_arch="db_resnet50",
        reco_arch="crnn_vgg16_bn",
        pretrained=True,
        assume_straight_pages=True,
    )
    if use_gpu and torch.cuda.is_available():
        predictor = predictor.cuda()
    return predictor


def extract_lines(result) -> list[str]:
    """Flatten a docTR Document into reading-order text lines."""
    lines: list[str] = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                words = [w.value for w in line.words]
                if words:
                    lines.append(" ".join(words))
    return lines


def run_doctr_on_file(predictor, file_path: Path) -> list[str]:
    """Run docTR on a single image or PDF, return text lines."""
    suffix = file_path.suffix.lower()
    if suffix in PDF_EXTENSIONS:
        doc = DocumentFile.from_pdf(str(file_path))
    elif suffix in IMAGE_EXTENSIONS:
        doc = DocumentFile.from_images(str(file_path))
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    result = predictor(doc)
    return extract_lines(result)


def run_ocr_on_file(
    file_path: Path,
    output_path: Path | None = None,
    use_gpu: bool = True,
) -> str:
    """Run docTR on a single image or PDF."""
    print(f"Loading docTR predictor (GPU={use_gpu and torch.cuda.is_available()})...")
    predictor = build_predictor(use_gpu=use_gpu)
    print("Models loaded!")

    print(f"Processing {file_path.name}...")
    lines = run_doctr_on_file(predictor, file_path)
    print(f"  Extracted {len(lines)} lines")

    text = "\n".join(lines)
    if output_path:
        output_path.write_text(text, encoding="utf-8")
        print(f"Wrote output to {output_path}")
    else:
        print("\n--- OCR Result ---")
        print(text)
    return text


def run_ocr_on_dataset(dataset: str, samples: list[str], use_gpu: bool = True):
    """Run docTR on benchmark dataset samples."""
    print(f"Loading docTR predictor (GPU={use_gpu and torch.cuda.is_available()})...")
    predictor = build_predictor(use_gpu=use_gpu)
    print("Models loaded!")

    samples_dir = settings.DATA_DIR / f"{dataset}/samples"
    results_dir = settings.RESULTS_DIR / f"doctr/{dataset}"
    results_dir.mkdir(parents=True, exist_ok=True)

    for sample_id in samples:
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
            lines = run_doctr_on_file(predictor, img_path)
            output_path = results_dir / f"{sample_id}.txt"
            output_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"  Wrote {len(lines)} lines to {output_path}")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run docTR OCR on images, PDFs, or benchmark datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--file", "-f", type=Path, help="Single image or PDF file")
    parser.add_argument("--output", "-o", type=Path, help="Output file (default: stdout)")
    parser.add_argument(
    	"--all",
    	action="store_true",
    	help="Run on ALL prepared samples (not just the 25-sample default)",
    )
    parser.add_argument(
        "--dataset", "-d", choices=list(DATASETS.keys()), help="Benchmark dataset"
    )
    parser.add_argument("--samples", "-s", nargs="*", help="Specific sample IDs")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    args = parser.parse_args()

    if args.file and args.dataset:
        parser.error("Cannot use both --file and --dataset")
    if not args.file and not args.dataset:
        parser.error("Must specify either --file or --dataset")

    use_gpu = not args.cpu

    if args.file:
        if not args.file.exists():
            parser.error(f"File not found: {args.file}")
        run_ocr_on_file(args.file, output_path=args.output, use_gpu=use_gpu)
    else:
    	if args.samples:
            samples = args.samples
	elif args.all:
    	    samples = discover_samples(args.dataset)
	else:
            samples = DATASETS[args.dataset]
    	
        
        run_ocr_on_dataset(args.dataset, samples, use_gpu=use_gpu)

    print("\nDone!")
