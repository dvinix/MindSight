import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "operational"
    assert "version" in data
    assert "/predict" in data["endpoints"]


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "mindsight-api"


def test_predict_stressed_standard():
    payload = {
        "text": "I feel hopeless, deeply depressed and completely overwhelmed by daily anxiety.",
        "model_type": "baseline",
        "threshold": 0.45
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["risk"] == "stressed"
    assert data["confidence"] >= 0.45
    assert data["threshold"] == 0.45
    assert data["model_used"] == "baseline"
    assert len(data["explanation"]) > 0
    assert data["features"]["word_count"] > 0
    assert "disclaimer" in data


def test_predict_protective_standard():
    payload = {
        "text": "Today was calm, restful and peaceful. I feel grateful and happy with my progress.",
        "model_type": "baseline",
        "threshold": 0.45
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["risk"] == "not stressed"
    assert data["confidence"] < 0.45


def test_predict_empty_text():
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code in [400, 422]


def test_predict_whitespace_only():
    resp = client.post("/predict", json={"text": "      \n\t  "})
    assert resp.status_code == 400


def test_predict_symbols_only():
    resp = client.post("/predict", json={"text": "@#$%^&*()_+~`"})
    assert resp.status_code == 400


def test_predict_missing_text_field():
    resp = client.post("/predict", json={"model_type": "baseline"})
    assert resp.status_code == 422


def test_predict_invalid_json():
    resp = client.post("/predict", content="invalid payload", headers={"Content-Type": "application/json"})
    assert resp.status_code == 422


def test_predict_boundary_thresholds():
    payload_low = {
        "text": "Regular day at work.",
        "model_type": "baseline",
        "threshold": 0.0
    }
    resp_low = client.post("/predict", json=payload_low)
    assert resp_low.status_code == 200
    assert resp_low.json()["risk"] == "stressed"

    payload_high = {
        "text": "Regular day at work.",
        "model_type": "baseline",
        "threshold": 1.0
    }
    resp_high = client.post("/predict", json=payload_high)
    assert resp_high.status_code == 200
    assert resp_high.json()["risk"] == "not stressed"


def test_predict_out_of_bound_threshold():
    payload_neg = {"text": "I feel stressed", "threshold": -0.2}
    resp_neg = client.post("/predict", json=payload_neg)
    assert resp_neg.status_code == 422

    payload_over = {"text": "I feel stressed", "threshold": 1.5}
    resp_over = client.post("/predict", json=payload_over)
    assert resp_over.status_code == 422


def test_predict_different_model_types():
    for m in ["baseline", "svm", "bilstm", "bert", "custom_unknown"]:
        payload = {
            "text": "I am feeling exhausted and anxious about tomorrow.",
            "model_type": m,
            "threshold": 0.45
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        assert resp.json()["model_used"] == m


def test_explain_endpoint_valid():
    payload = {
        "text": "I feel completely trapped, terrified and overwhelmed by panic.",
        "threshold": 0.45
    }
    resp = client.post("/explain", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "tokens" in data
    assert len(data["tokens"]) > 0
    assert data["risk"] == "stressed"


def test_explain_endpoint_empty():
    resp = client.post("/explain", json={"text": "   "})
    assert resp.status_code == 400


def test_method_not_allowed():
    resp = client.get("/predict")
    assert resp.status_code == 405
