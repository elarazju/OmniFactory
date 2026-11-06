import pytest

from omnifactory.training.asr_metrics import corpus_wer, word_errors, words


def test_corpus_wer_weights_reference_words():
    result = corpus_wer(
        [
            {"reference": "one", "prediction": "wrong"},
            {"reference": "one two three four", "prediction": "one two three four extra"},
        ]
    )
    assert result == {"wer": 0.4, "errors": 2, "reference_words": 5, "examples": 2}


def test_word_errors_and_explicit_normalization():
    assert words("Ｈｅｌｌｏ, DON'T! I’m 42.") == ["hello", "don't", "i'm", "42"]
    assert word_errors("a b c", "a c") == {"errors": 1, "reference_words": 3}
    assert word_errors("a b", "") == {"errors": 2, "reference_words": 2}
    with pytest.raises(ValueError):
        word_errors("...", "anything")
    with pytest.raises(ValueError):
        corpus_wer([])
