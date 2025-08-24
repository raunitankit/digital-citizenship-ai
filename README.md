
# Digital Citizenship Detector — AI Edition (Middle School)

A classroom-friendly project where students build an AI-assisted "Digital Citizenship Detector" that:
- Classifies short free-text answers (Safe / Risky / Respectful / Disrespectful)
- Checks basic toxicity and scam "vibe" on messages
- Gives friendly, template-based feedback
- Runs locally with a simple Streamlit web app

## Quick Start

```bash
# 1) (Recommended) Create a fresh virtual environment
python3 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 2) Install deps (CPU-only)
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

> First run will download small open models. Internet is required once for model download; afterwards it works offline.

## Project Structure

```
digicit_ai_project/
├─ app.py                # Streamlit UI
├─ detectors.py          # AI helpers (zero-shot, toxicity, scam)
├─ feedback.py           # Feedback templates and helpers
├─ requirements.txt
├─ README.md
├─ docs/
│  ├─ teacher_guide.md   # Lesson plan, assessments, extensions
│  └─ student_handout.md # Step-by-step student tasks
└─ data/
   └─ sample_inputs.txt
```

## Teaching Flow (4 weeks)

- **Week 1:** Plain quiz (no AI), discuss labels & rubric.
- **Week 2:** Add zero-shot classification for free-text answers (Safe vs. Risky, etc.).
- **Week 3:** Add toxicity + scam checks; compare AI vs. student judgment.
- **Week 4:** Polish Streamlit UI; poster/demo day.
- **Extension:** Error analysis: "When did AI get it wrong and why?"

## Safety & Privacy

- Keep all examples anonymous and age-appropriate.
- The app runs locally. No student accounts are required.
- Teach that AI is *assistive*, not a final authority.
