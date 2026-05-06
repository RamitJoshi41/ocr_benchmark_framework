#!/usr/bin/env python3
"""
Run PaddleOCR on images or PDFs with retry logic and GPU diagnostics.

Usage:
    # Single file (image or PDF)
    python scripts/run_paddleocr.py --file document.pdf
    python scripts/run_paddleocr.py --file image.png
    python scripts/run_paddleocr.py --file document.pdf --output result.txt

    # Benchmark datasets
    python scripts/run_paddleocr.py --dataset sroie
    python scripts/run_paddleocr.py --dataset funsd
    python scripts/run_paddleocr.py --dataset sroie --samples 000 001 002

Features:
    - Native PDF support (converts pages to images automatically)
    - Automatic retry on transient failures (20% failure rate observed)
    - GPU memory diagnostics when failures occur
    - Optional interactive mode to kill GPU processes
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path
from PIL import Image

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ocr_benchmark_2025.datasets import DATASETS, discover_samples

BASE = Path(__file__).parent.parent

# Supported file extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
PDF_EXTENSIONS = {".pdf"}


def get_gpu_memory_usage() -> str:
    """Get GPU memory usage via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return "No GPU processes found"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "nvidia-smi not available"


def is_transient_failure(text: str, min_length: int = 4) -> bool:
    """Detect transient OCR failures (very short output for full documents).

    Transient failures produce output like "n a o t o e e e..." instead of
    actual document text. These are fixed by simply re-running OCR.

    Note: Reduced threshold to 4 to support cropped text datasets.
    """
    if not text.strip():
        return True
    return len(text) < min_length


def run_ocr_with_retry(
    ocr,
    image_path: str,
    max_retries: int = 2,
    interactive: bool = False,
) -> list[str]:
    """Run PaddleOCR with retry logic for transient failures.

    Args:
        ocr: PaddleOCR instance
        image_path: Path to image file
        max_retries: Number of retries on transient failure
        interactive: If True, prompt to kill GPU processes on failure
    """
    import paddle  # Import needed for empty_cache

    for attempt in range(max_retries + 1):
        try:
            result = ocr.predict(image_path)

            # Extract text from PaddleOCR v5 result
            lines = []
            if result and result[0]:
                ocr_result = result[0]
                if "rec_texts" in ocr_result:
                    lines = ocr_result["rec_texts"]
                else:
                    # Fallback for older format
                    for item in ocr_result:
                        if item and len(item) >= 2:
                            text = item[1][0] if isinstance(item[1], tuple) else item[1]
                            lines.append(text)

            text = "\n".join(lines)

            # Check for transient failure
            if is_transient_failure(text):
                if attempt < max_retries:
                    print(
                        f"    Transient failure detected (attempt {attempt + 1}/{max_retries + 1})"
                    )
                    print(f"    GPU memory usage:\n{get_gpu_memory_usage()}")

                    # Try to clear GPU cache to recover from bad state
                    try:
                        paddle.device.cuda.empty_cache()
                        print("    Cleared GPU cache.")
                    except Exception:
                        pass

                    if interactive:
                        response = input("    Kill GPU processes? [y/N]: ").strip().lower()
                        if response == "y":
                            # Could implement process killing here
                            print("    (Process killing not implemented - retry only)")

                    time.sleep(1)
                    continue

            return lines

        except Exception as e:
            print(f"    Error: {e}")
            if attempt < max_retries:
                print(f"    Retrying... (attempt {attempt + 1}/{max_retries + 1})")
                try:
                    paddle.device.cuda.empty_cache()
                except Exception:
                    pass
                time.sleep(1)
            else:
                raise

    return []


def pdf_to_images(pdf_path: Path, dpi: int = 200) -> list[Path]:
    """Convert PDF pages to temporary PNG images.

    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for rendering (default 200 DPI)

    Returns:
        List of paths to temporary PNG files
    """
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    temp_images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)

        # Create temp file that persists until explicitly deleted
        temp_path = Path(tempfile.mktemp(suffix=f"_page{page_num:03d}.png"))
        pix.save(str(temp_path))
        temp_images.append(temp_path)

    doc.close()
    return temp_images


def run_ocr_on_file(
    file_path: Path,
    output_path: Path | None = None,
    interactive: bool = False,
    lang: str = "en",
) -> str:
    """Run PaddleOCR on a single file (image or PDF).

    Args:
        file_path: Path to image or PDF file
        output_path: Optional path to save results (default: print to stdout)
        interactive: If True, prompt to kill GPU processes on failure
        lang: OCR language (default: "en")

    Returns:
        Extracted text as string
    """
    print("Loading PaddleOCR models (this takes 30-60 seconds on first run)...")
    import paddle
    from paddleocr import PaddleOCR

    if paddle.device.is_compiled_with_cuda():
        # Even if get_available_device returns [], set_device("gpu") might work
        try:
            paddle.device.set_device("gpu")
        except Exception:
            pass

    ocr = PaddleOCR(lang=lang)
    print("Models loaded!")

    suffix = file_path.suffix.lower()
    all_lines = []
    temp_files = []

    try:
        if suffix in PDF_EXTENSIONS:
            print("Converting PDF to images...")
            temp_files = pdf_to_images(file_path)
            print(f"  {len(temp_files)} page(s) to process")

            for i, img_path in enumerate(temp_files):
                print(f"Processing page {i + 1}/{len(temp_files)}...")
                lines = run_ocr_with_retry(ocr, str(img_path), interactive=interactive)
                all_lines.extend(lines)
                print(f"  Extracted {len(lines)} lines")

        elif suffix in IMAGE_EXTENSIONS:
            print("Processing image...")
            all_lines = run_ocr_with_retry(ocr, str(file_path), interactive=interactive)
            print(f"  Extracted {len(all_lines)} lines")

        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            try:
                temp_file.unlink()
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


def run_ocr_on_dataset(dataset: str, samples: list[str], interactive: bool = False):
    """Run PaddleOCR on benchmark dataset samples."""
    print("Loading PaddleOCR models (this takes 30-60 seconds)...")
    import paddle
    from paddleocr import PaddleOCR

    if paddle.device.is_compiled_with_cuda():
        try:
            paddle.device.set_device("gpu")
        except Exception:
            pass

    ocr = PaddleOCR(lang="en")
    print("Models loaded!")

    samples_dir = BASE / f"datasets/{dataset}/samples"
    results_dir = BASE / f"results/paddleocr/{dataset}"
    results_dir.mkdir(parents=True, exist_ok=True)

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
        tmp_resized = None
        try:
            # Pre-resize huge images to avoid OOM hangs on 4GB VRAM
            with Image.open(img_path) as im:
                w, h = im.size
                max_side = max(w, h)
                if max_side > 2500:
                    scale = 2500 / max_side
                    new_w, new_h = int(w * scale), int(h * scale)
                    print(f"  Pre-resizing {w}x{h} -> {new_w}x{new_h}")
                    resized = im.convert("RGB").resize((new_w, new_h), Image.LANCZOS)
                    tmp_resized = Path(tempfile.mktemp(suffix=".jpg"))
                    resized.save(tmp_resized, "JPEG", quality=92)
                    ocr_input = str(tmp_resized)
                else:
                    ocr_input = str(img_path)

            lines = run_ocr_with_retry(ocr, ocr_input, interactive=interactive)

            output_path = results_dir / f"{sample_id}.txt"
            output_path.write_text("\n".join(lines), encoding="utf-8")
            print(f"  Wrote {len(lines)} lines to {output_path}")
        except Exception as e:
            print(f"  Failed after retries: {e}")
        finally:
            # Free GPU memory between samples (helps with fragmentation)
            try:
                import paddle
                paddle.device.cuda.empty_cache()
            except Exception:
                pass
            # Clean up temp file
            if tmp_resized and tmp_resized.exists():
                try:
                    tmp_resized.unlink()
                except OSError:
                    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run PaddleOCR on images, PDFs, or benchmark datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single file
    python scripts/run_paddleocr.py --file document.pdf
    python scripts/run_paddleocr.py --file image.png --output result.txt
    python scripts/run_paddleocr.py --file german_doc.pdf --lang german
    
    # Benchmark dataset
    python scripts/run_paddleocr.py --dataset sroie
    python scripts/run_paddleocr.py --dataset funsd --samples 000 001 002
""",
    )

    # File mode
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        help="Single image or PDF file to process",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file for results (default: print to stdout)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run on ALL prepared samples (not just the 25-sample default)",
    )
    parser.add_argument(
        "--lang",
        "-l",
        default="en",
        help="OCR language: en, german, ch, fr, etc. (default: en)",
    )

    # Dataset mode (legacy)
    parser.add_argument(
        "--dataset",
        "-d",
        choices=list(DATASETS.keys()),
        help="Benchmark dataset to process",
    )
    parser.add_argument(
        "--samples",
        "-s",
        nargs="*",
        help="Specific sample IDs for dataset mode",
    )

    # Common options
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Prompt to kill GPU processes on failure",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.file and args.dataset:
        parser.error("Cannot use both --file and --dataset")

    if not args.file and not args.dataset:
        parser.error("Must specify either --file or --dataset")

    if args.file:
        if not args.file.exists():
            parser.error(f"File not found: {args.file}")
        run_ocr_on_file(
            args.file,
            output_path=args.output,
            interactive=args.interactive,
            lang=args.lang,
        )
    else:
        if args.samples:
            samples = args.samples
        elif args.all:
            samples = discover_samples(args.dataset)
        else:
            samples = DATASETS[args.dataset]
        run_ocr_on_dataset(args.dataset, samples, interactive=args.interactive)

    print("\nDone!")
