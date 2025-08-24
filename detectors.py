# detectors.py  — API-backed, no local model loads
import os
import requests
from typing import List, Dict, Any

HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # set in Streamlit secrets at deploy time
HF_API_URL = "https://api-inference.huggingface.co/models"

HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}

def _hf_post(model: str, payload: Dict[str, Any]) -> Any:
    url = f"{HF_API_URL}/{model}"
    r = requests.post(url, headers=HEADERS, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def classify_sentiment(text: str) -> Dict[str, Any]:
    """Binary sentiment, fast & small."""
    model = "distilbert-base-uncased-finetuned-sst-2-english"
    out = _hf_post(model, {"inputs": text})
    # API returns list of list of {label, score}
    pred = out[0][0] if isinstance(out, list) and out and isinstance(out[0], list) else {"label":"NEUTRAL","score":0.0}
    return {"model": model, "label": pred["label"], "score": float(pred["score"])}

def classify_toxicity(text: str) -> Dict[str, Any]:
    """Toxic content detection."""
    model = "unitary/toxic-bert"
    out = _hf_post(model, {"inputs": text})
    # returns list of list of {label, score}
    labels = {d["label"]: float(d["score"]) for d in out[0]}
    toxic_score = max(labels.values()) if labels else 0.0
    return {"model": model, "scores": labels, "toxic_score": toxic_score}

def classify_hate(text: str) -> Dict[str, Any]:
    """Hate/offensive detection."""
    model = "Hate-speech-CNERG/bert-base-uncased-hatexplain"
    out = _hf_post(model, {"inputs": text})
    # normalize response into a top label/score
    pred = out[0][0] if isinstance(out, list) and out and isinstance(out[0], list) else {"label":"SAFE","score":0.0}
    return {"model": model, "label": pred["label"], "score": float(pred["score"])}

def zero_shot_claim_check(text: str, labels: List[str]) -> Dict[str, Any]:
    """
    Zero-shot classification to approximate 'misinformation' signals by label confidence.
    Use a distilled MNLI model to keep latency/cost reasonable on HF.
    """
    model = "valhalla/distilbart-mnli-12-1"
    out = _hf_post(model, {"inputs": text, "parameters": {"candidate_labels": labels}})
    # structure: {labels: [...], scores: [...], sequence: "..."}
    return {"model": model, **out}
