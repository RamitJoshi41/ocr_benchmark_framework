"""
OCR evaluation metrics: CER, WER, and Bag-of-Words error.
"""

import re
from collections import Counter


def normalize_text(text: str) -> str:
    """Normalize text for fair comparison."""
    # Remove page markers
    text = re.sub(r"\[Page \d+\]", "", text)
    # Normalize whitespace
    return " ".join(text.split())


def tokenize_text(text: str) -> str:
    """Convert text to normalized word tokens.

    This is the recommended comparison method for OCR benchmarks as it eliminates
    false CER inflation from format differences (e.g., word-tokenized ground truth
    vs sentence-based OCR output).
    """
    # Collapse whitespace and lowercase
    text = " ".join(text.lower().split())

    # Split into words
    words = text.split()

    # Strip boundary punctuation, keep internal (e.g., "don't" stays)
    cleaned = []
    for word in words:
        stripped = word.strip(".,;:!?()[]{}\"'-*#@&")
        if stripped:
            cleaned.append(stripped)

    return " ".join(cleaned)


def levenshtein_distance(s1: str | list, s2: str | list) -> int:
    """Calculate Levenshtein (edit) distance between two sequences."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate.

    Args:
        reference: Ground truth text
        hypothesis: OCR output text

    Returns:
        CER as a float (0.0 = perfect, can exceed 1.0 if hypothesis is much longer)
    """
    if not reference:
        # If reference is empty, any hypothesis content is pure insertion error
        return float(len(hypothesis)) if hypothesis else 0.0
    distance = levenshtein_distance(reference, hypothesis)
    return distance / len(reference)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate.

    Args:
        reference: Ground truth text
        hypothesis: OCR output text

    Returns:
        WER as a float (0.0 = perfect, can exceed 1.0 if hypothesis is much longer)
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        # If reference is empty, any hypothesis content is pure insertion error
        return float(len(hyp_words)) if hyp_words else 0.0
    distance = levenshtein_distance(ref_words, hyp_words)
    return distance / len(ref_words)


def calculate_bow_error(reference: str, hypothesis: str) -> float:
    """Calculate Bag of Words Error Rate (order-independent).

    This metric compares word frequencies without considering order,
    making it robust to text reordering from layout differences.

    Formula: sum(|GT_count - OCR_count| per word) / total_words

    Args:
        reference: Ground truth text
        hypothesis: OCR output text

    Returns:
        BoW error as a float (0.0 = perfect match, 1.0+ = complete mismatch)
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not ref_words and not hyp_words:
        return 0.0

    ref_counts = Counter(ref_words)
    hyp_counts = Counter(hyp_words)

    # Get all unique words
    all_words = set(ref_counts.keys()) | set(hyp_counts.keys())

    # Sum of absolute differences
    total_diff = sum(abs(ref_counts.get(w, 0) - hyp_counts.get(w, 0)) for w in all_words)

    # Normalize by total word count (GT + OCR) / 2 to handle length differences
    total_words = (len(ref_words) + len(hyp_words)) / 2
    if total_words == 0:
        return 0.0

    return total_diff / total_words
