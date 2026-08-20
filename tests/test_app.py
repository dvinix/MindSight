import pytest
from src.app.client import MindSightClient
from src.app.components import (
    get_preset_samples,
    render_gauge_chart,
    render_feature_bar_chart,
    render_explanation_chips,
    render_crisis_resources
)


def test_preset_samples_integrity():
    presets = get_preset_samples()
    assert isinstance(presets, dict)
    assert len(presets) >= 4
    for title, text in presets.items():
        assert isinstance(title, str) and len(title.strip()) > 0
        assert isinstance(text, str) and len(text.strip()) > 30


def test_mindsight_client_offline_fallback():
    client = MindSightClient(base_url="http://127.0.0.1:59999")
    health = client.check_health()
    assert health["status"] == "offline"

    resp = client.predict("I feel so anxious and overwhelmed.", model_type="baseline", threshold=0.45)
    assert resp["success"] is True
    assert resp["source"] == "local_engine"
    assert resp["data"]["risk"] == "stressed"
    assert resp["data"]["confidence"] >= 0.45


def test_mindsight_client_empty_and_none():
    client = MindSightClient(base_url="http://127.0.0.1:59999")

    resp_empty = client.predict("", model_type="baseline", threshold=0.45)
    assert resp_empty["success"] is True
    assert resp_empty["data"]["risk"] == "not stressed"
    assert resp_empty["data"]["features"]["word_count"] == 0

    resp_none = client.predict(None, model_type="baseline", threshold=0.45)
    assert resp_none["success"] is True
    assert resp_none["data"]["risk"] == "not stressed"


def test_mindsight_client_threshold_clamping():
    client = MindSightClient(base_url="http://127.0.0.1:59999")

    resp_low = client.predict("Standard note", threshold=-0.5)
    assert resp_low["data"]["threshold"] == 0.0

    resp_high = client.predict("Standard note", threshold=1.8)
    assert resp_high["data"]["threshold"] == 1.0


def test_components_gauge_chart():
    fig1 = render_gauge_chart(0.85, 0.45)
    assert fig1 is not None
    assert hasattr(fig1, "data")

    fig2 = render_gauge_chart(0.0, 0.50)
    assert fig2 is not None

    fig3 = render_gauge_chart(1.0, 0.50)
    assert fig3 is not None


def test_components_feature_bar_chart():
    features = {
        "word_count": 45,
        "first_person_ratio": 0.12,
        "negation_count": 3,
        "sentiment_valence": -0.65
    }
    fig = render_feature_bar_chart(features)
    assert fig is not None
    assert len(fig.data) > 0


def test_components_feature_bar_chart_empty():
    fig = render_feature_bar_chart({})
    assert fig is not None
