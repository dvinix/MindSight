import os
import re
import html
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib

app = FastAPI(
    title="MindSight Conversational Screening API",
    description="Assistive AI screening service for conversational mental health risk detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextPredictionRequest(BaseModel):
    text: str = Field(..., min_length=1)
    model_type: Optional[str] = Field(default="baseline")
    threshold: Optional[float] = Field(default=0.45, ge=0.0, le=1.0)

class ExplanationToken(BaseModel):
    word: str
    weight: float
    category: str

class ExtractedFeatures(BaseModel):
    word_count: int
    first_person_ratio: float
    negation_count: int
    sentiment_valence: float

class TextPredictionResponse(BaseModel):
    risk: str
    confidence: float
    threshold: float
    model_used: str
    explanation: List[ExplanationToken]
    features: ExtractedFeatures
    disclaimer: str

def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"https?://\S+|www\.\S+|<url>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\br/[A-Za-z0-9_]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bu/[A-Za-z0-9_-]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[deleted\]|\[removed\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Za-z0-9\s\.\,\!\?\'\"]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def compute_linguistic_signals(text: str) -> Dict[str, Any]:
    cleaned = clean_text(text)
    words = cleaned.lower().split()
    total_words = max(len(words), 1)

    pronoun_words = {"i", "me", "my", "myself", "mine"}
    negation_words = {"not", "no", "never", "cannot", "can't", "won't", "don't", "hardly"}

    pronoun_count = sum(1 for w in words if w in pronoun_words)
    negation_count = sum(1 for w in words if w in negation_words)
    pronoun_ratio = pronoun_count / total_words

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

    base_score = 0.35 + min(pronoun_ratio * 0.4, 0.20) + min((negation_count / total_words) * 0.3, 0.15)
    matched_explanations = []

    for word, weight in high_risk_lexicon.items():
        if word in words:
            count = words.count(word)
            contrib = min(weight * (1 + 0.2 * (count - 1)), 0.6)
            base_score += contrib
            matched_explanations.append({"word": word, "weight": round(contrib, 3), "category": "risk"})

    for word, weight in protective_lexicon.items():
        if word in words:
            count = words.count(word)
            reduction = max(weight * (1 + 0.1 * (count - 1)), -0.4)
            base_score += reduction
            matched_explanations.append({"word": word, "weight": round(reduction, 3), "category": "protective"})

    confidence = max(0.02, min(0.98, base_score))
    sentiment_score = max(-1.0, min(1.0, 1.0 - (confidence * 2.0)))

    return {
        "confidence": round(confidence, 4),
        "explanation": sorted(matched_explanations, key=lambda x: abs(x["weight"]), reverse=True)[:10],
        "features": {
            "word_count": total_words if words else 0,
            "first_person_ratio": round(pronoun_ratio, 4) if words else 0.0,
            "negation_count": negation_count,
            "sentiment_valence": round(sentiment_score, 3)
        }
    }

@app.get("/")
def root():
    return {
        "service": "MindSight Conversational Screening API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": ["/health", "/predict", "/explain"]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "mindsight-api",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=TextPredictionResponse)
def predict_distress(req: TextPredictionRequest):
    cleaned = clean_text(req.text)
    if not cleaned:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text payload contains no valid alphanumeric tokens"
        )

    model_dir = os.environ.get("MODEL_PATH", "models")
    lr_path = os.path.join(model_dir, "logistic_regression.pkl")
    vec_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")

    if os.path.exists(lr_path) and os.path.exists(vec_path):
        try:
            model = joblib.load(lr_path)
            vectorizer = joblib.load(vec_path)
            x_vec = vectorizer.transform([cleaned])
            prob = float(model.predict_proba(x_vec)[0, 1])
            signals = compute_linguistic_signals(req.text)
            signals["confidence"] = round(prob, 4)
        except Exception:
            signals = compute_linguistic_signals(req.text)
    else:
        signals = compute_linguistic_signals(req.text)

    threshold = req.threshold if req.threshold is not None else 0.45
    risk_label = "stressed" if signals["confidence"] >= threshold else "not stressed"

    return TextPredictionResponse(
        risk=risk_label,
        confidence=signals["confidence"],
        threshold=threshold,
        model_used=req.model_type or "baseline",
        explanation=[ExplanationToken(**item) for item in signals["explanation"]],
        features=ExtractedFeatures(**signals["features"]),
        disclaimer="This is an assistive conversational screening aid, not a diagnostic or clinical tool."
    )

@app.post("/explain")
def explain_text(req: TextPredictionRequest):
    cleaned = clean_text(req.text)
    if not cleaned:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text payload contains no valid alphanumeric tokens"
        )
    signals = compute_linguistic_signals(req.text)
    threshold = req.threshold if req.threshold is not None else 0.45
    return {
        "text": req.text,
        "clean_text": cleaned,
        "risk": "stressed" if signals["confidence"] >= threshold else "not stressed",
        "confidence": signals["confidence"],
        "tokens": signals["explanation"]
    }
