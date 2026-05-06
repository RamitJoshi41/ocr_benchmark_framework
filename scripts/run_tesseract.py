#!/usr/bin/env python3
"""
Run Tesseract OCR on images, PDFs, or benchmark datasets.

Usage:
    # Single file (image or PDF)
    python scripts/run_tesseract.py --file document.pdf
    python scripts/run_tesseract.py --file image.png --output result.txt
    python scripts/run_tesseract.py --file german_doc.pdf --lang deu

    # Benchmark datasets
    python scripts/run_tesseract.py --dataset sroie
    python scripts/run_tesseract.py --dataset funsd
    python scripts/run_tesseract.py --dataset sroie --samples sroie_X00016469670

Notes:
    Requires the system Tesseract binary:
        sudo apt install tesseract-ocr tesseract-ocr-eng
    Language codes use Tesseract format: eng, deu, fra, spa, etc.
"""

import argparse
import sys
import tempfile
from pathlib import Path

try:
    import pytesseract
    from PIL import Image
except ImportError:
    print("Tesseract deps not installed. Install with: uv sync --all-extras")
    sys.exit(1)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_benchmark_2025.config import settings
from ocr_benchmark_2025.datasets import DATASETS, discover_samples

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
PDF_EXTENSIONS = {".pdf"}


def pdf_to_images(pdf_path: Path, dpi: int = 200) -> list[Path]:
    """Convert PDF pages to temporary PNG images."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    temp_images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        temp_path = Path(tempfile.mktemp(suffix=f"_page{page_num:03d}.png"))
        pix.save(str(temp_path))
        temp_images.append(temp_path)
    doc.close()
    return temp_images


def run_tesseract_on_image(image_path: Path, lang: str = "eng") -> list[str]:
    """Run Tesseract on a single image, return text lines."""
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang=lang)
    # Drop blank lines so the output matches the line-per-detection style
    return [ln for ln in text.splitlines() if ln.strip()]


def run_ocr_on_file(
    file_path: Path,
    output_path: Path | None = None,
    lang: str = "eng",
) -> str:
    """Run Tesseract on a single image or PDF."""
    suffix = file_path.suffix.lower()
    all_lines: list[str] = []
    temp_files: list[Path] = []

    try:
        if suffix in PDF_EXTENSIONS:
            print("Converting PDF to images...")
            temp_files = pdf_to_images(file_path)
            print(f"  {len(temp_files)} page(s) to process")

            for i, img_path in enumerate(temp_files):
                print(f"Processing page {i + 1}/{len(temp_files)}...")
                lines = run_tesseract_on_image(img_path, lang=lang)
                all_lines.extend(lines)
                print(f"  Extracted {len(lines)} lines")

        elif suffix in IMAGE_EXTENSIONS:
            print("Processing image...")
            all_lines = run_tesseract_on_image(file_path, lang=lang)
            print(f"  Extracted {len(all_lines)} lines")

        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    finally:
        for tf in temp_files:
            try:
                tf.unlink()
            except OSError:
                pass

    text = "\n".join(all_lines)

    if output_path:
        output_path.write_text(text, encoding="utf-8")
        print(f"Wrote output to {output_path}")
    else:
        print("\n--- OCR Result ---")
        print(text)

    return text


def run_ocr_on_dataset(dataset: str, samples: list[str], lang: str = "eng"):
    """Run Tesseract on benchmark dataset samples."""
    samples_dir = settings.DATA_DIR / f"{dataset}/samples"
    results_dir = settings.RESULTS_DIR / f"tesseract/{dataset}"
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
            lines = run_tesseract_on_image(img_path, lang=lang)
            output_path = results_dir / f"{sample_id}.txt"
            output_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"  Wrote {len(lines)} lines to {output_path}")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Tesseract OCR on images, PDFs, or benchmark datasets",
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
        "--lang",
        "-l",
        default="eng",
        help="Tesseract language code: eng, deu, fra, spa, etc. (default: eng)",
    )
    parser.add_argument(
        "--dataset", "-d", choices=list(DATASETS.keys()), help="Benchmark dataset"
    )
    parser.add_argument("--samples", "-s", nargs="*", help="Specific sample IDs")
    args = parser.parse_args()

    if args.file and args.dataset:
        parser.error("Cannot use both --file and --dataset")
    if not args.file and not args.dataset:
        parser.error("Must specify either --file or --dataset")

    if args.file:
        if not args.file.exists():
            parser.error(f"File not found: {args.file}")
        run_ocr_on_file(args.file, output_path=args.output, lang=args.lang)
    else:
        if args.samples:
            samples = args.samples
        elif args.all:
            samples = discover_samples(args.dataset)
        else:
            samples = DATASETS[args.dataset]
        run_ocr_on_dataset(args.dataset, samples, lang=args.lang)

    print("\nDone!")
