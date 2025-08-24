# detectors.py — HF Inference API, robust token loading + friendly 401s
import os
import requests
from typing import List, Dict, Any

try:
    import streamlit as st  # available on Streamlit Cloud
except Exception:  # local or non-streamlit env
    st = None

def _get_hf_token() -> str | None:
    # Prefer env, then st.secrets if available
    token = os.getenv("HF_API_TOKEN")
    if not token and st is not None:
        token = st.secrets.get("HF_API_TOKEN", None)  # type: ignore[attr-defined]
    return token

def _headers() -> Dict[str, str]:
    token = _get_hf_token()
    return {"Authorization": f"Bearer {token}"} if token else {}

HF_API_URL = "https://api-inference.huggingface.co/models"

def _hf_post(model: str, payload: Dict[str, Any]) -> Any:
    url = f"{HF_API_URL}/{model}"
    headers = _headers()
    if not headers:
        raise RuntimeError(
            "HF_API_TOKEN is missing. Set it in Streamlit → App → Settings → Secrets."
        )
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    # Give clearer errors for common cases
    if r.status_code == 401:
        raise RuntimeError(
            "Unauthorized (401) from Hugging Face. Check HF_API_TOKEN value/expiry."
        )
    if r.status_code == 403:
        raise RuntimeError(
            "Forbidden (403) from Hugging Face. Your account/token may lack access."
        )
    r.raise_for_status()
    return r.json()

def classify_sentiment(text: str) -> Dict[str, Any]:
    model = "distilbert-base-uncased-finetuned-sst-2-english"
    out = _hf_post(model, {"inputs": text})
    pred = out[0][0] if isinstance(out, list) and out and isinstance(out[0], list) else {"label":"NEUTRAL","score":0.0}
    return {"model": model, "label": pred["label"], "score": float(pred["score"])}

def classify_toxicity(text: str) -> Dict[str, Any]:
    model = "unitary/toxic-bert"
    out = _hf_post(model, {"inputs": text})
    labels = {d["label"]: float(d["score"]) for d in out[0]}
    toxic_score = max(labels.values()) if labels else 0.0
    return {"model": model, "scores": labels, "toxic_score": toxic_score}

def classify_hate(text: str) -> Dict[str, Any]:
    model = "Hate-speech-CNERG/bert-base-uncased-hatexplain"
    out = _hf_post(model, {"inputs": text})
    pred = out[0][0] if isinstance(out, list) and out and isinstance(out[0], list) else {"label":"SAFE","score":0.0}
    return {"model": model, "label": pred["label"], "score": float(pred["score"])}

def zero_shot_claim_check(text: str, labels: List[str]) -> Dict[str, Any]:
    # Distilled MNLI for zero-shot classification
    model = "valhalla/distilbart-mnli-12-1"
    out = _hf_post(model, {"inputs": text, "parameters": {"candidate_labels": labels}})
    # expected shape: {"labels":[...], "scores":[...], "sequence":"..."}
    return {"model": model, **out}
