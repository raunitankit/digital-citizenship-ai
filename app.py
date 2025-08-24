import os
import streamlit as st
from feedback import feedback_for_labels
from detectors import (
    zero_shot_claim_check,
    classify_toxicity,
    classify_sentiment,  # optional (unused in UI but available)
    classify_hate,       # optional (unused in UI but available)
)

# ----------------------- Page & Presets -----------------------
st.set_page_config(
    page_title="Digital Citizenship Detector (AI Edition)",
    page_icon="🛡️",
    layout="centered",
)

PRESETS = {
    "Safe – Stranger request": "I only accept requests from people I actually know.",
    "Respectful – Don’t share": "Let’s not share that photo—it could hurt their feelings.",
    "Borderline – Just a joke": "Relax, it’s just a joke. Everyone shares stuff.",
    "Toxic": "You're such a loser.",
    "Scam": "Want to earn $2500 for 1 hour of work",
}

# ----------------------- Session state -----------------------
if "user_text" not in st.session_state:
    st.session_state.user_text = ""
if "preset_sel" not in st.session_state:
    st.session_state.preset_sel = "(none)"

# ----------------------- Helper: token warning -----------------------
def _has_hf_token() -> bool:
    try:
        return bool(
            os.getenv("HF_API_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or (st.secrets.get("HF_API_TOKEN") if "HF_API_TOKEN" in st.secrets else None)
            or (st.secrets.get("HUGGINGFACEHUB_API_TOKEN") if "HUGGINGFACEHUB_API_TOKEN" in st.secrets else None)
        )
    except Exception:
        return False

if not _has_hf_token():
    st.warning(
        "HF_API_TOKEN is not set. Add it in **Streamlit → App → Settings → Secrets** to avoid rate limits:\n\n"
        "HF_API_TOKEN = \"hf_...\"\n"
        "HUGGINGFACEHUB_API_TOKEN = \"hf_...\"\n\n"
        "Then **Clear cache and reboot** the app.",
        icon="⚠️",
    )

# ----------------------- Callbacks -----------------------
def apply_demo_text():
    st.session_state.user_text = "I wouldn't accept the request. I only connect with people I know."
    # No st.rerun() needed; button triggers a rerun automatically.

def apply_preset():
    sel = st.session_state.preset_sel
    if sel and sel in PRESETS:
        st.session_state.user_text = PRESETS[sel]
    # selectbox on_change triggers rerun automatically.

# ----------------------- UI -----------------------
st.title("🛡️ Digital Citizenship Detector — AI Edition")
st.write(
    """
Paste a short answer or a chat/message. The AI will estimate:
- **Digital behavior labels** (Safe / Respectful / Risky / Disrespectful / Scam)
- **Toxicity score** (0→1)

Then you'll see a **friendly feedback** message.
"""
)

# Text input (bound to session state)
st.text_area(
    "Your text:",
    key="user_text",
    placeholder="Write how you would respond if a stranger sent you a friend request...",
    height=160,
)

with st.expander("Try a preset"):
    st.selectbox(
        "Pick an example",
        ["(none)"] + list(PRESETS.keys()),
        key="preset_sel",
        on_change=apply_preset,
    )

cols = st.columns(2)
with cols[0]:
    analyze_btn = st.button("Analyze", use_container_width=True)
with cols[1]:
    demo_btn = st.button("Use demo text", use_container_width=True, on_click=apply_demo_text)

# ----------------------- Helpers -----------------------
DISPLAY_LABELS = ["Safe", "Respectful", "Risky", "Disrespectful", "Scam"]
CANDIDATE_LABELS = [l.lower() for l in DISPLAY_LABELS]  # what we send to the HF API

def format_scores(scores_dict):
    if not scores_dict:
        return "No scores."
    items = sorted(scores_dict.items(), key=lambda kv: kv[1], reverse=True)
    md_lines = ["| Label | Score |", "|---|---:|"]
    for label, score in items:
        md_lines.append(f"| {label} | {score:.3f} |")
    return "\n".join(md_lines)

# ----------------------- Main Analyze -----------------------
if analyze_btn and st.session_state.user_text.strip():
    text = st.session_state.user_text
    with st.spinner("Thinking..."):
        results = {}
        errors = []

        # 1) Zero-shot (multi-label) → Safe/Respectful/Risky/Disrespectful/Scam
        try:
            zs = zero_shot_claim_check(text, CANDIDATE_LABELS)
            raw_labels = zs.get("labels", [])
            raw_scores = zs.get("scores", [])
            label_scores = {lbl: float(scr) for lbl, scr in zip(raw_labels, raw_scores)}
            display_scores = {lbl.title(): label_scores.get(lbl, 0.0) for lbl in CANDIDATE_LABELS}
            results["labels"] = display_scores
        except Exception as e:
            errors.append(f"Zero-shot classification failed: {e}")
            results["labels"] = {}

        # 2) Toxicity
        try:
            tox = classify_toxicity(text)  # {"model":..., "scores":{...}, "toxic_score": float}
            results["toxicity"] = float(tox.get("toxic_score", 0.0))
        except Exception as e:
            errors.append(f"Toxicity classification failed: {e}")
            results["toxicity"] = 0.0

    # ----------------------- Show Results -----------------------
    st.subheader("Results")

    st.markdown("**Digital Behavior Labels**")
    st.markdown(format_scores(results.get("labels", {})))

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Toxicity", f"{results.get('toxicity', 0.0):.3f}")
    with c2:
        scam_score = results.get("labels", {}).get("Scam", 0.0)
        st.metric("Scam likelihood", f"{scam_score:.3f}")

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
        with st.expander("Diagnostics (errors)"):
            for er in errors:
                st.write("•", er)

st.caption("Note: These are estimates from hosted models; always apply human judgment.")
