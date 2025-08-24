# detectors.py — robust HF Inference API (canonical models + fallbacks)
import os, requests
from typing import List, Dict, Any, Optional

try:
    import streamlit as st
except Exception:
    st = None

HF_API_URL = "https://api-inference.huggingface.co/models"
KEYS = ["HF_API_TOKEN", "HUGGINGFACE_API_TOKEN", "HUGGINGFACEHUB_API_TOKEN"]

def _norm(v: Optional[str]) -> Optional[str]:
    return (str(v).strip().strip('"').strip("'")) if v else None

def _get_token() -> Optional[str]:
    for k in KEYS:
        v = _norm(os.getenv(k))
        if v: return v
    if st is not None:
        try:
            for k in KEYS:
                v = _norm(st.secrets.get(k))  # type: ignore[attr-defined]
                if v: return v
        except Exception:
            pass
    return None

def _headers() -> Dict[str, str]:
    tok = _get_token()
    # allow unauthenticated use as a fallback (lower limits), but warn upstream
    h = {"Accept": "application/json", "Cache-Control": "no-cache",
         "User-Agent": "ThinkPythonAI-DigCit/1.2"}
    if tok: h["Authorization"] = f"Bearer {tok}"
    return h

def _hf_post(model: str, payload: Dict[str, Any]) -> Any:
    url = f"{HF_API_URL}/{model}"
    r = requests.post(url, headers=_headers(), json=payload, timeout=40)
    # Helpful messages:
    if r.status_code == 401:
        raise RuntimeError("Unauthorized (401). Check HF token (read scope) in Streamlit Secrets.")
    if r.status_code == 403:
        raise RuntimeError("Forbidden (403). Token lacks permission for this model.")
    if r.status_code == 404:
        # Surface server message for clarity
        raise FileNotFoundError(f"404 from HF for '{model}': {r.text[:200]}")
    if r.status_code == 503:
        raise RuntimeError(f"{model} warming up/unavailable (503). Try again.")
    r.raise_for_status()
    return r.json()

# -------- Zero-shot classification (canonical + fallbacks) --------
ZSC_MODELS = [
    "facebook/bart-large-mnli",                    # canonical
    "joeddav/bart-large-mnli-yahoo-answers",       # fallback
    "typeform/distilbert-base-uncased-mnli",       # fallback
]

def zero_shot_claim_check(text: str, labels: List[str]) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for m in ZSC_MODELS:
        try:
            out = _hf_post(m, {
                "inputs": text,
                "parameters": {"candidate_labels": labels, "multi_label": True},
                "options": {"use_cache": True},
            })
            # Parse common shapes
            if isinstance(out, dict) and "labels" in out and "scores" in out:
                return {"model": m, **out}
            if isinstance(out, list) and out and isinstance(out[0], dict) and "labels" in out[0]:
                d = out[0]; d["model"] = m; return d
            last_err = RuntimeError(f"Unexpected zero-shot output from {m}: {type(out)}")
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Zero-shot failed across models: {last_err}")

# -------- Toxicity (multi-label) --------
TOX_MODELS = [
    "unitary/toxic-bert",
    "s-nlp/roberta_toxicity_classifier",
]

def classify_toxicity(text: str) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for m in TOX_MODELS:
        try:
            out = _hf_post(m, {"inputs": text})
            labels: Dict[str, float] = {}
            if isinstance(out, list):
                if out and isinstance(out[0], list):
                    labels = {d["label"]: float(d["score"]) for d in out[0]}
                elif out and isinstance(out[0], dict):
                    labels = {d["label"]: float(d["score"]) for d in out}
            elif isinstance(out, dict) and "labels" in out and "scores" in out:
                labels = {lbl: float(scr) for lbl, scr in zip(out["labels"], out["scores"])}
            return {"model": m, "scores": labels, "toxic_score": max(labels.values()) if labels else 0.0}
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Toxicity failed across models: {last_err}")

# (Optional extras if you use them)
def classify_sentiment(text: str) -> Dict[str, Any]:
    for m in ["distilbert-base-uncased-finetuned-sst-2-english",
              "cardiffnlp/twitter-roberta-base-sentiment-latest"]:
        try:
            out = _hf_post(m, {"inputs": text})
            if isinstance(out, list) and out:
                pred = out[0][0] if isinstance(out[0], list) else out[0]
                return {"model": m, "label": pred.get("label","NEUTRAL"), "score": float(pred.get("score",0.0))}
        except Exception:
            continue
    return {"model": "n/a", "label": "NEUTRAL", "score": 0.0}

def classify_hate(text: str) -> Dict[str, Any]:
    for m in ["Hate-speech-CNERG/bert-base-uncased-hatexplain", "unitary/unbiased-toxic-roberta"]:
        try:
            out = _hf_post(m, {"inputs": text})
            if isinstance(out, list) and out:
                pred = out[0][0] if isinstance(out[0], list) else out[0]
                return {"model": m, "label": pred.get("label","SAFE"), "score": float(pred.get("score",0.0))}
        except Exception:
            continue
    return {"model": "n/a", "label": "SAFE", "score": 0.0}
