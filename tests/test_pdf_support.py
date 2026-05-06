"""Tests for PDF support in run_paddleocr.py."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class TestFileExtensions:
    """Test file extension detection."""

    def test_image_extensions_defined(self):
        """Verify all common image extensions are supported."""
        from run_paddleocr import IMAGE_EXTENSIONS

        assert ".png" in IMAGE_EXTENSIONS
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".jpeg" in IMAGE_EXTENSIONS
        assert ".tiff" in IMAGE_EXTENSIONS
        assert ".tif" in IMAGE_EXTENSIONS
        assert ".bmp" in IMAGE_EXTENSIONS
        assert ".webp" in IMAGE_EXTENSIONS

    def test_pdf_extensions_defined(self):
        """Verify PDF extension is supported."""
        from run_paddleocr import PDF_EXTENSIONS

        assert ".pdf" in PDF_EXTENSIONS


class TestTransientFailureDetection:
    """Test transient failure detection logic."""

    def test_short_output_is_failure(self):
        """Short output indicates transient failure."""
        from run_paddleocr import is_transient_failure

        assert is_transient_failure("") is True
        assert is_transient_failure("n a o t") is False  # Updated: 7 chars > min_length 4
        assert is_transient_failure("x" * 3) is True

    def test_normal_output_not_failure(self):
        """Normal length output is not a failure."""
        from run_paddleocr import is_transient_failure

        assert is_transient_failure("x" * 50) is False
        assert (
            is_transient_failure("This is a normal OCR result with more than enough text content.")
            is False
        )

    def test_custom_min_length(self):
        """Custom minimum length threshold works."""
        from run_paddleocr import is_transient_failure

        assert is_transient_failure("short", min_length=10) is True
        assert is_transient_failure("short", min_length=5) is False


class TestPdfToImages:
    """Test PDF to image conversion."""

    @pytest.fixture
    def sample_pdf(self):
        """Create a minimal PDF for testing."""
        pytest.importorskip("fitz")
        import fitz

        # Create a simple 2-page PDF
        doc = fitz.open()
        for i in range(2):
            page = doc.new_page(width=100, height=100)
            page.insert_text((10, 50), f"Page {i + 1}")

        pdf_path = Path(tempfile.mktemp(suffix=".pdf"))
        doc.save(str(pdf_path))
        doc.close()

        yield pdf_path

        # Cleanup
        if pdf_path.exists():
            pdf_path.unlink()

    def test_pdf_converts_to_images(self, sample_pdf):
        """PDF pages are converted to individual images."""
        from run_paddleocr import pdf_to_images

        images = pdf_to_images(sample_pdf)

        try:
            assert len(images) == 2
            for img in images:
                assert img.exists()
                assert img.suffix == ".png"
        finally:
            # Cleanup temp images
            for img in images:
                if img.exists():
                    img.unlink()

    def test_pdf_custom_dpi(self, sample_pdf):
        """Custom DPI setting is applied."""
        from run_paddleocr import pdf_to_images

        # Higher DPI should produce larger images
        images_low = pdf_to_images(sample_pdf, dpi=72)
        images_high = pdf_to_images(sample_pdf, dpi=300)

        try:
            # Both should produce same number of pages
            assert len(images_low) == len(images_high) == 2

            # Higher DPI images should be larger
            size_low = images_low[0].stat().st_size
            size_high = images_high[0].stat().st_size
            assert size_high > size_low
        finally:
            for img in images_low + images_high:
                if img.exists():
                    img.unlink()


class TestArgumentParsing:
    """Test CLI argument parsing."""

    def test_file_mode_accepts_path(self):
        """--file argument accepts a path."""
        import argparse

        from run_paddleocr import DATASETS

        # Create parser manually to test
        parser = argparse.ArgumentParser()
        parser.add_argument("--file", "-f", type=Path)
        parser.add_argument("--output", "-o", type=Path)
        parser.add_argument("--lang", "-l", default="en")
        parser.add_argument("--dataset", "-d", choices=list(DATASETS.keys()))
        parser.add_argument("--samples", "-s", nargs="*")
        parser.add_argument("--interactive", "-i", action="store_true")

        args = parser.parse_args(["--file", "/path/to/doc.pdf"])
        assert args.file == Path("/path/to/doc.pdf")
        assert args.lang == "en"

    def test_dataset_mode_still_works(self):
        """--dataset argument works for benchmark mode."""
        import argparse

        from run_paddleocr import DATASETS

        parser = argparse.ArgumentParser()
        parser.add_argument("--file", "-f", type=Path)
        parser.add_argument("--dataset", "-d", choices=list(DATASETS.keys()))
        parser.add_argument("--samples", "-s", nargs="*")

        args = parser.parse_args(["--dataset", "sroie"])
        assert args.dataset == "sroie"
        assert args.file is None

    def test_language_option(self):
        """--lang option sets OCR language."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--file", "-f", type=Path)
        parser.add_argument("--lang", "-l", default="en")

        args = parser.parse_args(["--file", "doc.pdf", "--lang", "german"])
        assert args.lang == "german"


class TestGpuMemoryUsage:
    """Test GPU memory diagnostics."""

    def test_returns_string_when_nvidia_smi_missing(self):
        """Returns fallback message when nvidia-smi not available."""
        from run_paddleocr import get_gpu_memory_usage

        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = get_gpu_memory_usage()
            assert result == "nvidia-smi not available"

    def test_returns_no_processes_on_empty_output(self):
        """Returns appropriate message when no GPU processes running."""
        from run_paddleocr import get_gpu_memory_usage

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            result = get_gpu_memory_usage()
            assert result == "No GPU processes found"


class TestRunOcrOnFile:
    """Test the main file OCR function (mocked).

    Note: PaddleOCR is imported inside run_ocr_on_file(), so we mock at paddleocr module level.
    """

    def test_unsupported_file_type_raises(self):
        """Unsupported file extensions raise ValueError."""
        from run_paddleocr import run_ocr_on_file

        mock_ocr_class = MagicMock()
        with patch.dict("sys.modules", {"paddleocr": MagicMock(PaddleOCR=mock_ocr_class)}):
            with pytest.raises(ValueError, match="Unsupported file type"):
                run_ocr_on_file(Path("/tmp/test.xyz"))

    def test_pdf_calls_pdf_to_images(self):
        """PDF files trigger PDF-to-image conversion."""
        from run_paddleocr import run_ocr_on_file

        mock_ocr_instance = MagicMock()
        mock_ocr_instance.predict.return_value = [{"rec_texts": ["test"]}]
        mock_ocr_class = MagicMock(return_value=mock_ocr_instance)

        with patch.dict(
            "sys.modules",
            {
                "paddleocr": MagicMock(PaddleOCR=mock_ocr_class),
                "paddle": MagicMock(),
            },
        ):
            with patch("run_paddleocr.pdf_to_images") as mock_pdf:
                mock_pdf.return_value = []

                run_ocr_on_file(Path("/tmp/test.pdf"))

                mock_pdf.assert_called_once()

    def test_image_skips_pdf_conversion(self):
        """Image files don't trigger PDF conversion."""
        from run_paddleocr import run_ocr_on_file

        mock_ocr_instance = MagicMock()
        mock_ocr_instance.predict.return_value = [{"rec_texts": ["test"]}]
        mock_ocr_class = MagicMock(return_value=mock_ocr_instance)

        with patch.dict(
            "sys.modules",
            {
                "paddleocr": MagicMock(PaddleOCR=mock_ocr_class),
                "paddle": MagicMock(),
            },
        ):
            with patch("run_paddleocr.pdf_to_images") as mock_pdf:
                with patch("run_paddleocr.run_ocr_with_retry", return_value=["test"]):
                    # Need to create a temp file since we check suffix
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                        temp_path = Path(f.name)

                    try:
                        run_ocr_on_file(temp_path)
                        mock_pdf.assert_not_called()
                    finally:
                        temp_path.unlink()

    def test_output_written_to_file(self):
        """Output is written to file when --output specified."""
        from run_paddleocr import run_ocr_on_file

        mock_ocr_instance = MagicMock()
        mock_ocr_instance.predict.return_value = [{"rec_texts": ["line1", "line2"]}]
        mock_ocr_class = MagicMock(return_value=mock_ocr_instance)

        with patch.dict(
            "sys.modules",
            {
                "paddleocr": MagicMock(PaddleOCR=mock_ocr_class),
                "paddle": MagicMock(),
            },
        ):
            with patch("run_paddleocr.run_ocr_with_retry", return_value=["line1", "line2"]):
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as img_f:
                    img_path = Path(img_f.name)

                with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as out_f:
                    out_path = Path(out_f.name)

                try:
                    run_ocr_on_file(img_path, output_path=out_path)
                    content = out_path.read_text()
                    assert "line1" in content
                    assert "line2" in content
                finally:
                    img_path.unlink()
                    out_path.unlink()
