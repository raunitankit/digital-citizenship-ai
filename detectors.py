# detectors.py — HF Inference API with robust token loading + model fallbacks
import os
import requests
from typing import List, Dict, Any, Optional

try:
    import streamlit as st
except Exception:
    st = None  # local/non-Streamlit env

HF_API_URL = "https://api-inference.huggingface.co/models"

KEYS = ["HF_API_TOKEN", "HUGGINGFACE_API_TOKEN", "HUGGINGFACEHUB_API_TOKEN"]

def _normalize(tok: Optional[str]) -> Optional[str]:
    if not tok:
        return None
    return str(tok).strip().strip('"').strip("'") or None

def _get_token() -> Optional[str]:
    # env first
    for k in KEYS:
        v = _normalize(os.getenv(k))
        if v:
            return v
    # then Streamlit secrets
    if st is not None:
        try:
            for k in KEYS:
                v = _normalize(st.secrets.get(k))  # type: ignore[attr-defined]
                if v:
                    return v
        except Exception:
            pass
    return None

def _headers() -> Dict[str, str]:
    tok = _get_token()
    if not tok:
        raise RuntimeError(
            "No Hugging Face token found in env or Streamlit secrets. "
            "Add HF_API_TOKEN under Streamlit → Settings → Secrets."
        )
    return {
        "Authorization": f"Bearer {tok}",
        "Accept": "application/json",
        "Cache-Control": "no-cache",
        "User-Agent": "ThinkPythonAI-DigCit/1.1",
    }

def _hf_post(model: str, payload: Dict[str, Any]) -> Any:
    url = f"{HF_API_URL}/{model}"
    r = requests.post(url, headers=_headers(), json=payload, timeout=40)
    # Friendly messages for common errors
    if r.status_code == 401:
        raise RuntimeError("Unauthorized (401) from Hugging Face. Token missing/invalid/expired.")
    if r.status_code == 403:
        raise RuntimeError("Forbidden (403). Token lacks permission for this model.")
    if r.status_code == 404:
        raise FileNotFoundError(f"Model not found (404): {model}")
    if r.status_code == 503:
        # Warming up models sometimes return 503; surface it clearly
        raise RuntimeError(f"Model {model} is warming up or unavailable (503). Try again.")
    r.raise_for_status()
    return r.json()

# ---------------- Zero-shot classification ----------------
# Use a canonical, widely-available model first
ZSC_MODELS = [
    "facebook/bart-large-mnli",
    # Fallbacks (smaller or alt checkpoints if above is unavailable)
    "joeddav/bart-large-mnli-yahoo-answers",
    "typeform/distilbert-base-uncased-mnli",
]

def zero_shot_claim_check(text: str, labels: List[str]) -> Dict[str, Any]:
    last_err = None
    for m in ZSC_MODELS:
        try:
            out = _hf_post(
                m,
                {
                    "inputs": text,
                    "parameters": {"candidate_labels": labels, "multi_label": True},
                    "options": {"use_cache": True},
                },
            )
            # Expected shape: {"labels":[...], "scores":[...], "sequence":"..."}  OR HF pipeline list format
            if isinstance(out, dict) and "labels" in out and "scores" in out:
                return {"model": m, **out}
            # Some server variants return a list with one dict
            if isinstance(out, list) and out and isinstance(out[0], dict) and "labels" in out[0]:
                d = out[0]
                d["model"] = m
                return d
            # If we got here, shape was unexpected—try next model
            last_err = RuntimeError(f"Unexpected zero-shot output shape from {m}: {type(out)}")
        except (FileNotFoundError, RuntimeError) as e:
            last_err = e
            continue
    raise RuntimeError(f"Zero-shot classification failed across models: {last_err}")

# ---------------- Toxicity classification ----------------
TOX_MODELS = [
    "unitary/toxic-bert",                 # multi-label toxic categories
    "s-nlp/roberta_toxicity_classifier",  # good English toxicity classifier
]

def classify_toxicity(text: str) -> Dict[str, Any]:
    last_err = None
    for m in TOX_MODELS:
        try:
            out = _hf_post(m, {"inputs": text})
            # Two common shapes:
            # 1) sequence classification list-of-lists: [[{"label": "...", "score": ...}, ...]]
            # 2) direct list: [{"label": "...", "score": ...}, ...]
            labels: Dict[str, float] = {}
            if isinstance(out, list):
                # out could be list of lists or list of dicts
                if out and isinstance(out[0], list):
                    labels = {d["label"]: float(d["score"]) for d in out[0]}
                elif out and isinstance(out[0], dict):
                    labels = {d["label"]: float(d["score"]) for d in out}
            elif isinstance(out, dict) and "labels" in out and "scores" in out:
                labels = {lbl: float(scr) for lbl, scr in zip(out["labels"], out["scores"])}
            toxic_score = max(labels.values()) if labels else 0.0
            return {"model": m, "scores": labels, "toxic_score": toxic_score}
        except (FileNotFoundError, RuntimeError) as e:
            last_err = e
            continue
    raise RuntimeError(f"Toxicity classification failed across models: {last_err}")

# ---------------- Optional extras kept for completeness ----------------
def classify_sentiment(text: str) -> Dict[str, Any]:
    models = [
        "distilbert-base-uncased-finetuned-sst-2-english",
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
    ]
    last_err = None
    for m in models:
        try:
            out = _hf_post(m, {"inputs": text})
            if isinstance(out, list) and out and isinstance(out[0], list) and out[0]:
                pred = out[0][0]
            elif isinstance(out, list) and out and isinstance(out[0], dict):
                pred = out[0]
            else:
                pred = {"label": "NEUTRAL", "score": 0.0}
            return {"model": m, "label": pred.get("label", "NEUTRAL"), "score": float(pred.get("score", 0.0))}
        except (FileNotFoundError, RuntimeError) as e:
            last_err = e
            continue
    raise RuntimeError(f"Sentiment classification failed across models: {last_err}")

def classify_hate(text: str) -> Dict[str, Any]:
    models = [
        "Hate-speech-CNERG/bert-base-uncased-hatexplain",
        "unitary/unbiased-toxic-roberta",
    ]
    last_err = None
    for m in models:
        try:
            out = _hf_post(m, {"inputs": text})
            if isinstance(out, list) and out and isinstance(out[0], list) and out[0]:
                pred = out[0][0]
            elif isinstance(out, list) and out and isinstance(out[0], dict):
                pred = out[0]
            else:
                pred = {"label": "SAFE", "score": 0.0}
            return {"model": m, "label": pred.get("label", "SAFE"), "score": float(pred.get("score", 0.0))}
        except (FileNotFoundError, RuntimeError) as e:
            last_err = e
            continue
    raise RuntimeError(f"Hate classification failed across models: {last_err}")
