
import streamlit as st
from feedback import feedback_for_labels
from detectors import get_zero_shot_clf, analyze_text, format_scores

PRESETS = {
    "Safe – Stranger request": "I only accept requests from people I actually know.",
    "Respectful – Don’t share": "Let’s not share that photo—it could hurt their feelings.",
    "Borderline – Just a joke": "Relax, it’s just a joke. Everyone shares stuff.",
    "Toxic": "You're such a loser.",
    "Scam": "Want to earn $2500 for 1 hour of work",
}
st.set_page_config(page_title="Digital Citizenship Detector (AI Edition)", page_icon="🛡️", layout="centered")

st.title("🛡️ Digital Citizenship Detector — AI Edition")
st.write("""
Paste a short answer or a chat/message. The AI will estimate:
- **Digital behavior labels** (Safe / Risky / Respectful / Disrespectful)
- **Toxicity score** (0→1)
- **Scam likelihood** (0→1)
Then you'll see a **friendly feedback** message.
""")

user_text = st.text_area("Your text:", placeholder="Write how you would respond if a stranger sent you a friend request...")

 
if "clf" not in st.session_state:
    st.session_state.clf = get_zero_shot_clf()

cols = st.columns(2)
with cols[0]:
    analyze_btn = st.button("Analyze")
with cols[1]:
    demo_btn = st.button("Use demo text")

if demo_btn:
    user_text = "I wouldn't accept the request. I only connect with people I know."
    st.session_state["demo_used"] = True
    st.experimental_rerun()

if analyze_btn and user_text.strip():
    with st.spinner("Thinking..."):
        result = analyze_text(user_text, st.session_state.clf)
    st.subheader("Results")
    st.markdown("**Digital Behavior Labels**")
    st.write(format_scores(result['labels']))

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Toxicity", f"{result['toxicity']:.3f}")
    with col2:
        st.metric("Scam likelihood", f"{result['scam']:.3f}")

    st.subheader("Feedback")
    fb = feedback_for_labels(result['labels'])
    st.success(fb)

st.caption("Note: These are estimates from small language models; always apply human judgment.")
