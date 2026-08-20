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
