from feedback import feedback_for_labels
import os, requests, streamlit as st

def whoami():
    tok = (os.getenv("HF_API_TOKEN") or st.secrets.get("HF_API_TOKEN") or
           os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN"))
    if not tok: return "No token detected"
    tok = str(tok).strip().strip('"').strip("'")
    r = requests.get("https://huggingface.co/api/whoami-v2",
                     headers={"Authorization": f"Bearer {tok}"}, timeout=15)
    return f"whoami-v2: {r.status_code} {r.text[:120]}..."
with st.expander("Diagnostics (temporary)"):
    st.write(whoami())


def detect_token_source():
    keys = [
        ("env:HF_API_TOKEN", os.getenv("HF_API_TOKEN")),
        ("env:HUGGINGFACE_API_TOKEN", os.getenv("HUGGINGFACE_API_TOKEN")),
        ("env:HUGGINGFACEHUB_API_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN")),
    ]
    # also check st.secrets
    try:
        keys += [
            ("secrets:HF_API_TOKEN", st.secrets.get("HF_API_TOKEN")),
            ("secrets:HUGGINGFACE_API_TOKEN", st.secrets.get("HUGGINGFACE_API_TOKEN")),
            ("secrets:HUGGINGFACEHUB_API_TOKEN", st.secrets.get("HUGGINGFACEHUB_API_TOKEN")),
        ]
    except Exception:
        pass
    found = [(k, v) for k, v in keys if v and str(v).strip()]
    if not found:
        return "No token detected"
    k, v = found[0]
    v = str(v).strip()
    masked = (v[:4] + "..." + v[-4:]) if len(v) >= 12 else "****"
    return f"Detected {k} = {masked}"

def try_plain_request():
    # minimal GET ping (public) and POST (auth)
    token = (
        os.getenv("HF_API_TOKEN")
        or os.getenv("HUGGINGFACE_API_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or (st.secrets.get("HF_API_TOKEN") if "HF_API_TOKEN" in st.secrets else None)
        or (st.secrets.get("HUGGINGFACEHUB_API_TOKEN") if "HUGGINGFACEHUB_API_TOKEN" in st.secrets else None)
    )
    if not token or not str(token).strip():
        return "No token to test"
    token = str(token).strip()
    headers = {"Authorization": f"Bearer {token}"}
    # tiny auth-required call
    try:
        r = requests.post(
            "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english",
            headers=headers,
            json={"inputs": "hello"},
            timeout=15,
        )
        return f"HF POST status={r.status_code}"
    except Exception as e:
        return f"HF POST error: {e}"

with st.expander("Diagnostics (temporary)"):
    st.write(detect_token_source())
    st.write(try_plain_request())
# --- /TEMP DIAGNOSTICS ---

def _has_hf_token() -> bool:
    if os.getenv("HF_API_TOKEN"):
        return True
    try:
        return bool(st.secrets.get("HF_API_TOKEN"))
    except Exception:
        return False

if not _has_hf_token():
    st.warning(
        "HF_API_TOKEN is not set. In Streamlit Cloud, go to **App → Settings → Secrets** and add:\n\n"
        "HF_API_TOKEN = \"hf_...\"\n\n"
        "Then reboot the app.",
        icon="⚠️",
    )

# API-backed detectors (no local torch/transformers)
from detectors import (
    zero_shot_claim_check,
    classify_toxicity,
    classify_sentiment,  # optional (unused in UI but available)
    classify_hate,       # optional (unused in UI but available)
)

# ----------------------- Config & Presets -----------------------
PRESETS = {
    "Safe – Stranger request": "I only accept requests from people I actually know.",
    "Respectful – Don’t share": "Let’s not share that photo—it could hurt their feelings.",
    "Borderline – Just a joke": "Relax, it’s just a joke. Everyone shares stuff.",
    "Toxic": "You're such a loser.",
    "Scam": "Want to earn $2500 for 1 hour of work",
}

st.set_page_config(
    page_title="Digital Citizenship Detector (AI Edition)",
    page_icon="🛡️",
    layout="centered",
)

st.title("🛡️ Digital Citizenship Detector — AI Edition")
st.write(
    """
Paste a short answer or a chat/message. The AI will estimate:
- **Digital behavior labels** (Safe / Respectful / Risky / Disrespectful / Scam)
- **Toxicity score** (0→1)

Then you'll see a **friendly feedback** message.
"""
)

# Helpful tip about the token (won't stop the app if missing, just warn)
if not os.getenv("HF_API_TOKEN"):
    st.warning(
        "HF_API_TOKEN is not set. Add it in **Streamlit → App → Settings → Secrets**.\n\n"
        "Without a token, the Hugging Face Inference API may fail or be rate-limited.",
        icon="⚠️",
    )

# ----------------------- Input -----------------------
user_text = st.text_area(
    "Your text:",
    placeholder="Write how you would respond if a stranger sent you a friend request...",
    height=160,
)

with st.expander("Try a preset"):
    preset_key = st.selectbox("Pick an example", ["(none)"] + list(PRESETS.keys()))
    if preset_key and preset_key != "(none)":
        user_text = PRESETS[preset_key]

cols = st.columns(2)
with cols[0]:
    analyze_btn = st.button("Analyze", use_container_width=True)
with cols[1]:
    demo_btn = st.button("Use demo text", use_container_width=True)

if demo_btn:
    user_text = "I wouldn't accept the request. I only connect with people I know."
    st.session_state["demo_used"] = True
    st.experimental_rerun()

# ----------------------- Helpers -----------------------
DISPLAY_LABELS = ["Safe", "Respectful", "Risky", "Disrespectful", "Scam"]
CANDIDATE_LABELS = [l.lower() for l in DISPLAY_LABELS]  # what we send to the HF API

def format_scores(scores_dict):
    """
    Pretty-print a dict of label -> score (0..1) sorted by score desc.
    """
    if not scores_dict:
        return "No scores."
    items = sorted(scores_dict.items(), key=lambda kv: kv[1], reverse=True)
    md_lines = ["| Label | Score |", "|---|---:|"]
    for label, score in items:
        md_lines.append(f"| {label} | {score:.3f} |")
    return "\n".join(md_lines)

# ----------------------- Main Analyze -----------------------
if analyze_btn and user_text.strip():
    with st.spinner("Thinking..."):
        results = {}
        errors = []

        # 1) Zero-shot labels (safe/respectful/risky/disrespectful/scam)
        try:
            zs = zero_shot_claim_check(user_text, CANDIDATE_LABELS)
            # Expected: {"model": ..., "labels": [...], "scores": [...], "sequence": "..."}
            raw_labels = zs.get("labels", [])
            raw_scores = zs.get("scores", [])
            label_scores = {lbl: float(scr) for lbl, scr in zip(raw_labels, raw_scores)}
            # Remap back to Title Case for display
            display_scores = {lbl.title(): label_scores.get(lbl, 0.0) for lbl in CANDIDATE_LABELS}
            results["labels"] = display_scores
        except Exception as e:
            errors.append(f"Zero-shot classification failed: {e}")
            results["labels"] = {}

        # 2) Toxicity
        try:
            tox = classify_toxicity(user_text)  # {"model":..., "scores":{...}, "toxic_score": float}
            results["toxicity"] = float(tox.get("toxic_score", 0.0))
        except Exception as e:
            errors.append(f"Toxicity classification failed: {e}")
            results["toxicity"] = 0.0

    # ----------------------- Show Results -----------------------
    st.subheader("Results")

    # Digital Behavior Labels
    st.markdown("**Digital Behavior Labels**")
    st.markdown(format_scores(results.get("labels", {})))

    # Metrics row
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Toxicity", f"{results.get('toxicity', 0.0):.3f}")
    with c2:
        # Extract scam likelihood (title-case key)
        scam_score = results.get("labels", {}).get("Scam", 0.0)
        st.metric("Scam likelihood", f"{scam_score:.3f}")

    # Feedback
    st.subheader("Feedback")
    try:
        fb = feedback_for_labels(results.get("labels", {}))
        st.success(fb)
    except Exception as e:
        st.info(
            "Feedback module couldn't interpret the labels. "
            "Please check `feedback_for_labels` to accept a dict of label→score.\n\n"
            f"Error: {e}"
        )

    if errors:
        with st.expander("Diagnostics"):
            for er in errors:
                st.write("•", er)

st.caption("Note: These are estimates from hosted models; always apply human judgment.")
