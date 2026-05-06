"""Tests for OCR evaluation metrics."""

import ocr_benchmark_2025
from ocr_benchmark_2025.metrics import (
    calculate_bow_error,
    calculate_cer,
    calculate_wer,
    levenshtein_distance,
    normalize_text,
    tokenize_text,
)


def test_version():
    """Verify package version is accessible."""
    assert ocr_benchmark_2025.__version__ == "0.1.0"


# =============================================================================
# normalize_text tests
# =============================================================================


def test_normalize_text_whitespace():
    """Test whitespace normalization."""
    assert normalize_text("  hello   world  ") == "hello world"
    assert normalize_text("line1\nline2\tline3") == "line1 line2 line3"


def test_normalize_text_page_markers():
    """Test page marker removal."""
    assert normalize_text("[Page 1] Hello [Page 2]") == "Hello"
    assert normalize_text("[Page 123] Text") == "Text"


# =============================================================================
# tokenize_text tests
# =============================================================================


def test_tokenize_text_basic():
    """Test basic word tokenization."""
    assert tokenize_text("Hello, World!") == "hello world"
    assert tokenize_text("  Multiple   Spaces  ") == "multiple spaces"


def test_tokenize_text_internal_punctuation():
    """Test that internal punctuation is preserved."""
    assert tokenize_text("don't stop") == "don't stop"
    assert tokenize_text("it's, isn't it?") == "it's isn't it"


def test_tokenize_text_boundary_punctuation():
    """Test boundary punctuation stripping."""
    assert tokenize_text("(test)") == "test"
    assert tokenize_text('"quote"') == "quote"
    assert tokenize_text("--dashes--") == "dashes"


# =============================================================================
# levenshtein_distance tests
# =============================================================================


def test_levenshtein_empty_strings():
    """Test Levenshtein with empty strings."""
    assert levenshtein_distance("", "") == 0
    assert levenshtein_distance("abc", "") == 3
    assert levenshtein_distance("", "xyz") == 3


def test_levenshtein_identical():
    """Test Levenshtein with identical strings."""
    assert levenshtein_distance("abc", "abc") == 0
    assert levenshtein_distance("hello world", "hello world") == 0


def test_levenshtein_substitutions():
    """Test Levenshtein with substitutions only."""
    assert levenshtein_distance("abc", "abd") == 1
    assert levenshtein_distance("abc", "xyz") == 3


def test_levenshtein_classic_example():
    """Test classic kitten-sitting example."""
    # kitten -> sitten (s/k) -> sittin (e/i) -> sitting (insert g) = 3
    assert levenshtein_distance("kitten", "sitting") == 3


def test_levenshtein_insertions():
    """Test Levenshtein with insertions."""
    assert levenshtein_distance("abc", "abcd") == 1
    assert levenshtein_distance("abc", "aabbcc") == 3


def test_levenshtein_deletions():
    """Test Levenshtein with deletions."""
    assert levenshtein_distance("abcd", "abc") == 1
    assert levenshtein_distance("aabbcc", "abc") == 3


def test_levenshtein_symmetric():
    """Test that Levenshtein distance is symmetric."""
    assert levenshtein_distance("abc", "xyz") == levenshtein_distance("xyz", "abc")
    assert levenshtein_distance("kitten", "sitting") == levenshtein_distance("sitting", "kitten")


def test_levenshtein_on_lists():
    """Test Levenshtein works on word lists (for WER)."""
    assert levenshtein_distance(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert levenshtein_distance(["a", "b", "c"], ["a", "x", "c"]) == 1
    assert levenshtein_distance(["a", "b"], ["a", "b", "c"]) == 1


# =============================================================================
# calculate_cer tests
# =============================================================================


def test_cer_perfect_match():
    """Test CER with perfect match."""
    assert calculate_cer("hello", "hello") == 0.0
    assert calculate_cer("test string", "test string") == 0.0


def test_cer_empty_both():
    """Test CER when both strings are empty."""
    assert calculate_cer("", "") == 0.0


def test_cer_empty_reference_with_hypothesis():
    """Test CER when reference is empty but hypothesis has content."""
    # This is a key edge case: should return length of hypothesis
    assert calculate_cer("", "garbage") == 7.0
    assert calculate_cer("", "x") == 1.0


def test_cer_empty_hypothesis():
    """Test CER when hypothesis is empty."""
    assert calculate_cer("hello", "") == 1.0  # 5/5 = 1.0


def test_cer_single_error():
    """Test CER with single character error."""
    assert calculate_cer("hello", "hallo") == 0.2  # 1/5


def test_cer_can_exceed_one():
    """Test that CER can exceed 1.0 with insertions."""
    # Reference: "a" (1 char), Hypothesis: "aaaa" (4 chars)
    # Distance: 3 insertions, CER = 3/1 = 3.0
    assert calculate_cer("a", "aaaa") == 3.0


def test_cer_all_different():
    """Test CER when all characters differ."""
    assert calculate_cer("abc", "xyz") == 1.0  # 3 subs / 3 chars


# =============================================================================
# calculate_wer tests
# =============================================================================


def test_wer_perfect_match():
    """Test WER with perfect match."""
    assert calculate_wer("hello world", "hello world") == 0.0


def test_wer_empty_both():
    """Test WER when both strings are empty."""
    assert calculate_wer("", "") == 0.0


def test_wer_empty_reference_with_hypothesis():
    """Test WER when reference is empty but hypothesis has content."""
    assert calculate_wer("", "garbage text") == 2.0  # 2 words


def test_wer_empty_hypothesis():
    """Test WER when hypothesis is empty."""
    assert calculate_wer("hello world", "") == 1.0  # 2/2 = 1.0


def test_wer_missing_word():
    """Test WER with missing word."""
    assert calculate_wer("hello world", "hello") == 0.5  # 1/2


def test_wer_can_exceed_one():
    """Test that WER can exceed 1.0 with insertions."""
    # Reference: 1 word, Hypothesis: 4 words
    # Distance: 3 insertions, WER = 3/1 = 3.0
    assert calculate_wer("hello", "hello extra words here") == 3.0


# =============================================================================
# calculate_bow_error tests
# =============================================================================


def test_bow_perfect_match():
    """Test BoW with perfect match."""
    assert calculate_bow_error("hello world", "hello world") == 0.0


def test_bow_order_independent():
    """Test that BoW is order-independent."""
    assert calculate_bow_error("hello world", "world hello") == 0.0
    assert calculate_bow_error("a b c d", "d c b a") == 0.0


def test_bow_empty_both():
    """Test BoW when both strings are empty."""
    assert calculate_bow_error("", "") == 0.0


def test_bow_missing_words():
    """Test BoW with missing words."""
    # Reference: [a, b], Hypothesis: [a]
    # Diff: b is missing (1)
    # Total words: (2 + 1) / 2 = 1.5
    # BoW = 1 / 1.5 ≈ 0.667
    result = calculate_bow_error("a b", "a")
    assert 0.66 < result < 0.67


def test_bow_extra_words():
    """Test BoW with extra words in hypothesis."""
    # Reference: [a], Hypothesis: [a, b]
    # Diff: b is extra (1)
    # Total words: (1 + 2) / 2 = 1.5
    # BoW = 1 / 1.5 ≈ 0.667
    result = calculate_bow_error("a", "a b")
    assert 0.66 < result < 0.67


def test_bow_case_insensitive():
    """Test that BoW is case-insensitive."""
    assert calculate_bow_error("Hello World", "hello world") == 0.0


def test_bow_duplicate_words():
    """Test BoW handles duplicate words correctly."""
    # Reference: [hello, hello], Hypothesis: [hello]
    # Diff: hello count diff = 1
    # Total words: (2 + 1) / 2 = 1.5
    result = calculate_bow_error("hello hello", "hello")
    assert 0.66 < result < 0.67
