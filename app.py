import streamlit as st
from feedback import feedback_for_labels
from detectors import (
    zero_shot_claim_check,
    classify_toxicity,
)

# ----------------------- Config -----------------------
PRESETS = {
    "Safe – Stranger request": "I only accept requests from people I actually know.",
    "Respectful – Don’t share": "Let’s not share that photo—it could hurt their feelings.",
    "Borderline – Just a joke": "Relax, it’s just a joke. Everyone shares stuff.",
    "Toxic": "You're such a loser.",
    "Scam": "Want to earn $2500 for 1 hour of work",
}

DISPLAY_LABELS = ["Safe", "Respectful", "Risky", "Disrespectful", "Scam"]
CANDIDATE_LABELS = [l.lower() for l in DISPLAY_LABELS]

st.set_page_config(
    page_title="Digital Citizenship Detector (AI Edition)",
    page_icon="🛡️",
    layout="centered",
)

# ----------------------- Back button (top) -----------------------
st.markdown(
    """
    <div style="margin: 8px 0 14px; text-align: left;">
      <a href="https://thinkpythonai.com" target="_self"
         style="display:inline-block; background:#0f172a; color:white;
                padding:10px 16px; border-radius:8px; font-weight:600;
                text-decoration:none;">
         🔙 Back to ThinkPythonAI
      </a>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------- Title -----------------------
st.title("🛡️ Digital Citizenship Detector — AI Edition")
st.write(
    """
Paste a short answer or a chat/message. The AI will estimate:
- **Digital behavior labels** (Safe / Respectful / Risky / Disrespectful / Scam)
- **Toxicity score** (0→1)

Then you'll see a **friendly feedback** message.
"""
)

# ----------------------- Input -----------------------
if "user_text" not in st.session_state:
    st.session_state.user_text = ""

user_text = st.text_area(
    "Your text:",
    placeholder="Write how you would respond if a stranger sent you a friend request...",
    height=160,
    value=st.session_state.user_text,
)

with st.expander("Try a preset"):
    preset_key = st.selectbox("Pick an example", ["(none)"] + list(PRESETS.keys()))
    if preset_key and preset_key != "(none)":
        user_text = PRESETS[preset_key]
        st.session_state.user_text = user_text

cols = st.columns(2)
with cols[0]:
    analyze_btn = st.button("Analyze", use_container_width=True)
with cols[1]:
    demo_btn = st.button("Use demo text", use_container_width=True)

if demo_btn:
    user_text = "I wouldn't accept the request. I only connect with people I know."
    st.session_state.user_text = user_text

# ----------------------- Helpers -----------------------
def format_scores(scores_dict):
    """Pretty-print a dict of label -> score (0..1) sorted by score desc."""
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

        # 1) Zero-shot labels
        try:
            zs = zero_shot_claim_check(user_text, CANDIDATE_LABELS)
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
            tox = classify_toxicity(user_text)  # {"toxic_score": float}
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
            f"Error: {e}"
        )

    if errors:
        with st.expander("Diagnostics"):
            for er in errors:
                st.write("•", er)

# ----------------------- Back button (bottom) -----------------------
st.markdown(
    """
    <div style="margin: 20px 0 0; text-align: center;">
      <a href="https://thinkpythonai.com" target="_self"
         style="display:inline-block; background:#0f172a; color:white;
                padding:10px 16px; border-radius:8px; font-weight:600;
                text-decoration:none;">
         🔙 Back to ThinkPythonAI
      </a>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("Note: These are estimates from hosted models; always apply human judgment.")
