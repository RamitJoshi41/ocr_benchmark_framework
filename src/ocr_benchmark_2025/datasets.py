"""
Dataset configurations for OCR benchmarks.

Centralizes sample IDs to avoid duplication across scripts.
"""
from ocr_benchmark_2025.config import settings

# 25 samples per dataset
DATASETS = {
    "sroie": [
        "sroie_X00016469670",
        "sroie_X00016469671",
        "sroie_X51005200931",
        "sroie_X51005230605",
        "sroie_X51005230616",
        "sroie_X51005230621",
        "sroie_X51005230648",
        "sroie_X51005230657",
        "sroie_X51005230659",
        "sroie_X51005268275",
        "sroie_X51005268408",
        "sroie_X51005288570",
        "sroie_X51005301666",
        "sroie_X51005337867",
        "sroie_X51005337877",
        "sroie_X51005361906",
        "sroie_X51005361908",
        "sroie_X51005361912",
        "sroie_X51005361923",
        "sroie_X51005365187",
        "sroie_X51005433518",
        "sroie_X51005433543",
        "sroie_X51005433548",
        "sroie_X51005433556",
        "sroie_X51005442322",
    ],
    "funsd": [
        "82092117",
        "82251504",
        "82254765",
        "82491256",
        "82504862",
        "82562350",
        "82573104",
        "82837252",
        "83443897",
        "83573282",
        "83594639",
        "83624198",
        "83635935",
        "83772145",
        "83823750",
        "83996357",
        "85201976",
        "85240939",
        "85540866",
        "85629964",
        "86220490",
        "86244113",
        "86263525",
        "87086073",
        "87125460",
    ],
}
def discover_samples(dataset: str) -> list[str]:
    """Discover ALL prepared sample IDs in datasets/<dataset>/samples/ on disk.

    Returns the stems of all image files (png/jpg/jpeg) for which a matching
    ground-truth file (_gt.txt) exists. Use this to benchmark the full dataset
    rather than the hardcoded 25-sample subset in DATASETS.
    """
    if dataset == "sroie":
        samples_dir = settings.SROIE_SAMPLES_PATH
    elif dataset == "funsd":
        samples_dir = settings.FUNSD_SAMPLES_PATH
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if not samples_dir.is_dir():
        raise FileNotFoundError(
            f"Samples dir not found: {samples_dir}. "
            f"Run scripts/prepare_datasets.py first."
        )

    extensions = {".png", ".jpg", ".jpeg"}
    image_stems = {
        p.stem for p in samples_dir.iterdir() if p.suffix.lower() in extensions
    }
    gt_stems = {
        p.stem.replace("_gt", "") for p in samples_dir.glob("*_gt.txt")
    }
    # Only include samples that have BOTH an image AND ground truth
    sample_ids = sorted(image_stems & gt_stems)

    if not sample_ids:
        raise FileNotFoundError(f"No prepared samples found in {samples_dir}")
    return sample_ids


