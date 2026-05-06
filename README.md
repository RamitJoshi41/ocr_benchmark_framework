# ocr-benchmark-2025

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OCR benchmark toolkit comparing **PaddleOCR**, **EasyOCR**, and **OCR.space** on SROIE and FUNSD datasets.

## Results Summary

| Dataset | PaddleOCR | EasyOCR | OCR.space |
|---------|-----------|---------|-----------|
| SROIE (CER) | **15.2%** | 20.4% | 44.3% |
| SROIE (BoW) | **25.8%** | 81.1% | 73.3% |
| FUNSD (CER) | **20.3%** | 26.4% | 48.2% |
| FUNSD (BoW) | **50.5%** | 102.1% | 84.8% |

*Lower is better. CER = Character Error Rate. BoW = Bag of Words Error (order-independent).*

**Winner: PaddleOCR** — Best on both metrics across all datasets.

## Metrics

- **CER (Character Error Rate)**: Levenshtein distance at character level
- **WER (Word Error Rate)**: Levenshtein distance at word level
- **BoW (Bag of Words Error)**: Order-independent word frequency comparison — robust to layout differences

## Installation

### Using uv (Recommended)

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable package management. `uv` replaces `pip`, `pip-tools`, and `virtualenv` with a single, high-performance tool written in Rust.

Why we use `uv`:
- **Speed**: Installs dependencies 10-100x faster than pip.
- **Reliability**: Uses a lockfile (`uv.lock`) to ensure reproducible environments.
- **Simplicity**: Manages python versions and virtual environments automatically.

```bash
# Clone the repository
git clone https://gitlab.com/agentic.ai.forge/ocr-benchmark-2025.git
cd ocr-benchmark-2025

# Create a virtual environment and install dependencies
# This reads pyproject.toml and installs all required packages
uv sync --all-extras

# Activate the environment
source .venv/bin/activate
```


### Traditional pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install base package
pip install -e .

# Install OCR engines (choose what you need)
pip install -e ".[paddleocr]"  # PaddleOCR
pip install -e ".[easyocr]"    # EasyOCR
pip install -e ".[datasets]"   # Dataset preparation tools
pip install -e ".[all]"        # Everything
```

### PaddleOCR GPU Support (Linux/Windows)

For GPU acceleration, this project is configured to use Baidu's custom package index via `uv`.

#### Reproducible Installation with uv
The `pyproject.toml` includes specific source mapping for `paddlepaddle-gpu`:
```toml
[[tool.uv.index]]
name = "paddlepaddle"
url = "https://www.paddlepaddle.org.cn/packages/stable/cu126/"
explicit = true

[tool.uv.sources]
paddlepaddle-gpu = { index = "paddlepaddle" }
```

Simply run:
```bash
uv sync --all-extras
```

#### GPU Troubleshooting
If PaddlePaddle is installed but doesn't "see" your GPU (e.g., `Available devices: []` or `Compiled with CUDA: False`), follow these steps:

1. **Verify Visibility:**
   PaddlePaddle initialization can be sensitive to environment variables. Explicitly allow the device:
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   uv run python -c "import paddle; paddle.utils.run_check()"
   ```

2. **Check Library Path:**
   If the above fails, ensure `LD_LIBRARY_PATH` includes the `.venv` library directory:
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/.venv/lib
   ```

3. **Memory Management:**
   For cards with limited VRAM (like GTX 1650 Ti 4GB), use sequential processing (default in scripts) and avoid high batch sizes.

## Dataset Setup

Due to licensing restrictions, datasets must be downloaded manually. This ensures compliance with the original licenses (non-commercial/research use).

### SROIE (ICDAR 2019 Competition)

1. Download from: https://rrc.cvc.uab.es/?ch=13
   * Requires registration/login.
   * **Important:** Select **"Task 1 - Scanned Receipt Text Localisation"**. This provides full-page images which are necessary for the benchmark. Do not use Task 2 (cropped images).
2. Extract to: `datasets/sroie_full/`

Expected structure:
```
datasets/sroie_full/
├── images/
│   ├── 000.jpg
│   ├── 001.jpg
│   └── ...
└── ground_truth/
    ├── 000.txt
    ├── 001.txt
    └── ...
```

### FUNSD (Form Understanding in Noisy Scanned Documents)

1. Download from: https://guillaumejaume.github.io/FUNSD/
   * Read and agree to the license.
2. Extract to: `datasets/funsd_full/dataset/`

Expected structure:
```
datasets/funsd_full/dataset/testing_data/
├── images/
│   ├── 82092117.png
│   └── ...
└── annotations/
    ├── 82092117.json
    └── ...
```

### Prepare Samples

After downloading the full datasets, run this script to extract standard samples (25 per dataset) for the benchmark:

```bash
python scripts/prepare_datasets.py
```

This will:
1. Verify the local datasets exist.
2. Copy 25 samples to the `datasets/` folder used by the benchmark.
3. Generate Ground Truth (`_gt.txt`) files.
    * **Note on Preprocessing**: For FUNSD, the script performs **Spatial Sorting**. Raw FUNSD annotations often store text blocks in non-logical order (e.g., starting with form labels like "TO:" regardless of their position). Our script sorts semantic entities by their vertical (Y) and horizontal (X) coordinates to create a Ground Truth that matches the natural reading order (top-to-bottom, left-to-right), ensuring fair comparison with OCR engines.

## Usage

### Run OCR

```bash
# PaddleOCR (recommended)
uv run scripts/run_paddleocr.py --dataset sroie
uv run scripts/run_paddleocr.py --dataset funsd

# EasyOCR
uv run scripts/run_easyocr.py sroie
uv run scripts/run_easyocr.py funsd

# OCR.space (requires API key)
export OCR_SPACE_API_KEY=your_key
uv run scripts/run_ocrspace.py all
```

### Run Benchmark

```bash
# Compare results
uv run scripts/run_benchmark.py --dataset sroie --tokenize
uv run scripts/run_benchmark.py --dataset funsd --tokenize
```

### LLM-as-Judge Evaluation

```bash
export ANTHROPIC_API_KEY=your_key
uv run scripts/llm_judge.py --all
```

## Handling PaddleOCR Transient Failures

PaddleOCR exhibits ~20% transient failure rate during batch processing (outputs single characters instead of text). The `run_paddleocr.py` script includes automatic retry logic:

```python
from scripts.run_paddleocr import run_ocr_with_retry

# Automatic retry with GPU diagnostics
lines = run_ocr_with_retry(ocr, image_path, max_retries=2)
```

Use `--interactive` flag to prompt for GPU process management on failure:

```bash
uv run scripts/run_paddleocr.py --dataset sroie --interactive
```

## Project Structure

```
ocr-benchmark-2025/
├── src/ocr_benchmark_2025/    # Core metrics library
│   ├── __init__.py
│   ├── metrics.py             # CER, WER, BoW calculations
│   └── py.typed
├── scripts/                    # Benchmark runners
│   ├── run_benchmark.py       # Calculate metrics
│   ├── run_paddleocr.py       # PaddleOCR with retry
│   ├── run_easyocr.py         # EasyOCR runner
│   ├── run_ocrspace.py        # OCR.space API
│   ├── prepare_datasets.py    # Extract samples
│   └── llm_judge.py           # LLM evaluation
├── datasets/                   # Downloaded datasets (gitignored)
├── results/                    # OCR outputs (gitignored)
└── tests/                      # Unit tests
```

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Lint and format
uv run ruff check src/ tests/ scripts/
uv run ruff format src/ tests/ scripts/
```

## Troubleshooting

### PaddlePaddle 3.3+ Issues

If you encounter the error `(Unimplemented) ConvertPirAttribute2RuntimeAttribute` or crashes involving `onednn`, specifically when using PaddlePaddle 3.3+:

This is caused by the new **Program Intermediate Representation (PIR)** API which is enabled by default in PaddlePaddle 3.3 but has known regressions with MKLDNN/oneDNN on CPU.

**Recommended Fix:** Downgrade `paddlepaddle` to version 3.2.0:

```bash
uv pip install paddlepaddle==3.2.0
```

**Alternative Workaround:** Disable the PIR API by setting the environment variable:

```bash
export FLAGS_enable_pir_api=0
```

Or add it before your command:

```bash
FLAGS_enable_pir_api=0 uv run scripts/run_paddleocr.py --dataset sroie
```

This reverts PaddlePaddle to the legacy IR system, which is stable for current OCR models.

## License

This project is licensed under the MIT License.

### Dataset Licenses & Attribution

We gratefully acknowledge the creators of the datasets used in this benchmark:

- **SROIE Dataset**: Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  - *Huang, Z., Chen, K., He, J., Bai, X., Karatzas, D., Lu, S., & Jawahar, C. V. (2019). ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction.*
- **FUNSD Dataset**: Provided for **non-commercial research use only**.
  - *Jaume, G., Ekenel, H. K., & Thiran, J. P. (2019). FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents.*

## References

- [SROIE Dataset](https://rrc.cvc.uab.es/?ch=13) - ICDAR 2019 Competition
- [FUNSD Dataset](https://guillaumejaume.github.io/FUNSD/) - Form Understanding
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - Baidu's OCR
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - JaidedAI
- [OCR.space](https://ocr.space/) - Cloud OCR API

## Blog Post

For detailed analysis and methodology, see: [Local OCR Benchmark 2025: Privacy-First Tools That Actually Work](https://agentic.ai.forge/blog/2025-12-29-ocr-benchmark-paddleocr-easyocr-ocrspace)
