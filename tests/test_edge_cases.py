from concurrent.futures import ThreadPoolExecutor

import pytest

from src.api.main import clean_text, compute_linguistic_signals
from src.app.client import MindSightClient


def test_linguistic_signals_division_by_zero_prevention():
    signals = compute_linguistic_signals("")
    assert signals["confidence"] >= 0.0
    assert signals["features"]["word_count"] == 0
    assert signals["features"]["first_person_ratio"] == 0.0

    signals_spaces = compute_linguistic_signals("   \t\n  ")
    assert signals_spaces["features"]["word_count"] == 0


def test_linguistic_signals_extreme_pronoun_density():
    text = "I me my myself mine I me my"
    signals = compute_linguistic_signals(text)
    assert signals["features"]["first_person_ratio"] == 1.0
    assert signals["confidence"] > 0.35


def test_linguistic_signals_extreme_negations():
    text = "not no never cannot can't won't don't"
    signals = compute_linguistic_signals(text)
    assert signals["features"]["negation_count"] == 7


def test_concurrent_screening_requests():
    client = MindSightClient(base_url="http://127.0.0.1:59999")
    client.timeout = 0.05  # Fast fallback for unit testing
    texts = [
        "I am having panic attacks and I cannot breathe.",
        "Today was a great sunny day in the park.",
        "Completely overwhelmed with exams and deadlines.",
        "Had coffee and finished my daily reading list.",
        "Hopeless and crying all night alone."
    ] * 6

    def process_text(t):
        return client.predict(t, model_type="baseline", threshold=0.45)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_text, texts))

    assert len(results) == len(texts)
    for r in results:
        assert r["success"] is True
        assert r["data"]["confidence"] >= 0.0


def test_deterministic_output():
    text = "I feel stressed and anxious every morning."
    res1 = compute_linguistic_signals(text)
    res2 = compute_linguistic_signals(text)
    assert res1["confidence"] == res2["confidence"]
    assert res1["features"] == res2["features"]
    assert len(res1["explanation"]) == len(res2["explanation"])


def test_punctuation_and_numeric_edge_cases():
    """Verify sentences containing only numbers or punctuation do not cause errors."""
    res_num = compute_linguistic_signals("12345 67890 999")
    assert res_num["confidence"] >= 0.0
    assert res_num["features"]["word_count"] == 3

    res_punct = compute_linguistic_signals("... ??? !!! ---")
    assert res_punct["confidence"] >= 0.0


def test_repeated_distress_words_scaling():
    """Repeated crisis keywords increase weight without exceeding maximum capped contribution."""
    text_single = "panic"
    text_multi = "panic panic panic panic panic"

    res_single = compute_linguistic_signals(text_single)
    res_multi = compute_linguistic_signals(text_multi)

    assert res_multi["confidence"] >= res_single["confidence"]
    assert res_multi["confidence"] <= 0.98
