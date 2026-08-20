import os
import re
import html
import requests
from typing import Dict, Any, List, Optional


class MindSightClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or os.environ.get("API_URL", "http://localhost:8000")).rstrip("/")
        self.timeout = 8

    def check_health(self) -> Dict[str, Any]:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=2)
            if resp.status_code == 200:
                return {"status": "connected", "details": resp.json()}
            return {"status": "degraded", "code": resp.status_code}
        except Exception as err:
            return {"status": "offline", "error": str(err)}

    def predict(self, text: str, model_type: str = "baseline", threshold: float = 0.45) -> Dict[str, Any]:
        clamped_threshold = max(0.0, min(1.0, float(threshold)))
        payload = {
            "text": str(text) if text is not None else "",
            "model_type": model_type or "baseline",
            "threshold": clamped_threshold
        }
        try:
            resp = requests.post(f"{self.base_url}/predict", json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                return {"success": True, "source": "api", "data": data}
            return {"success": False, "source": "api", "error": f"API Error {resp.status_code}: {resp.text}"}
        except Exception:
            fallback_data = self._compute_local_heuristic(text, model_type, clamped_threshold)
            return {"success": True, "source": "local_engine", "data": fallback_data}

    def _compute_local_heuristic(self, text: str, model_type: str, threshold: float) -> Dict[str, Any]:
        cleaned = self._clean_text(text)
        words = cleaned.lower().split() if cleaned else []
        total_words = len(words)

        if total_words == 0:
            return {
                "risk": "not stressed",
                "confidence": 0.05,
                "threshold": threshold,
                "model_used": model_type or "baseline",
                "explanation": [],
                "features": {
                    "word_count": 0,
                    "first_person_ratio": 0.0,
                    "negation_count": 0,
                    "sentiment_valence": 0.0
                },
                "disclaimer": "This is an AI conversational screening aid, not a diagnostic or clinical tool."
            }

        high_risk_lexicon = {
            "anxious": 0.38, "panic": 0.42, "depressed": 0.45, "depression": 0.45,
            "hopeless": 0.48, "suicidal": 0.65, "suicide": 0.65, "kill": 0.45,
            "overwhelmed": 0.35, "stress": 0.32, "stressed": 0.35, "crying": 0.30,
            "exhausted": 0.28, "alone": 0.25, "lonely": 0.30, "hate": 0.22,
            "scared": 0.26, "fear": 0.24, "helpless": 0.36, "trapped": 0.34,
            "worthless": 0.40, "breakdown": 0.38, "hurting": 0.28, "numb": 0.27,
            "struggling": 0.25, "insomnia": 0.22, "miserable": 0.30, "terrified": 0.32
        }

        protective_lexicon = {
            "happy": -0.25, "grateful": -0.30, "peace": -0.28, "calm": -0.25,
            "better": -0.20, "healing": -0.28, "support": -0.22, "love": -0.18,
            "good": -0.15, "relieved": -0.25, "hope": -0.22, "improving": -0.26,
            "excited": -0.20, "enjoy": -0.18, "glad": -0.20, "progress": -0.24
        }

        pronoun_words = {"i", "me", "my", "myself", "mine"}
        negation_words = {"not", "no", "never", "cannot", "can't", "won't", "don't", "hardly"}

        pronoun_count = sum(1 for w in words if w in pronoun_words)
        negation_count = sum(1 for w in words if w in negation_words)
        pronoun_ratio = pronoun_count / total_words

        base_score = 0.35 + min(pronoun_ratio * 0.4, 0.20) + min((negation_count / total_words) * 0.3, 0.15)

        matched_explanations = []
        for word, weight in high_risk_lexicon.items():
            if word in words:
                count = words.count(word)
                contribution = min(weight * (1 + 0.2 * (count - 1)), 0.6)
                base_score += contribution
                matched_explanations.append({"word": word, "weight": round(contribution, 3), "category": "risk"})

        for word, weight in protective_lexicon.items():
            if word in words:
                count = words.count(word)
                reduction = max(weight * (1 + 0.1 * (count - 1)), -0.4)
                base_score += reduction
                matched_explanations.append({"word": word, "weight": round(reduction, 3), "category": "protective"})

        if model_type == "bert":
            base_score = base_score * 1.05
        elif model_type == "bilstm":
            base_score = base_score * 1.02

        confidence = max(0.02, min(0.98, base_score))
        risk_label = "stressed" if confidence >= threshold else "not stressed"

        matched_explanations = sorted(matched_explanations, key=lambda x: abs(x["weight"]), reverse=True)[:10]
        sentiment_score = max(-1.0, min(1.0, 1.0 - (confidence * 2.0)))

        return {
            "risk": risk_label,
            "confidence": round(confidence, 4),
            "threshold": threshold,
            "model_used": model_type or "baseline",
            "explanation": matched_explanations,
            "features": {
                "word_count": total_words,
                "first_person_ratio": round(pronoun_ratio, 4),
                "negation_count": negation_count,
                "sentiment_valence": round(sentiment_score, 3)
            },
            "disclaimer": "This is an AI conversational screening aid, not a diagnostic or clinical tool."
        }

    def _clean_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        text = html.unescape(text)
        text = re.sub(r"https?://\S+|www\.\S+|<url>", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\br/[A-Za-z0-9_]+", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\bu/[A-Za-z0-9_-]+", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\[deleted\]|\[removed\]", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"[^A-Za-z0-9\s\.\,\!\?\'\"]", " ", text)
        return re.sub(r"\s+", " ", text).strip()
