
from transformers import pipeline

LABELS = ["Safe behavior", "Risky behavior", "Respectful", "Disrespectful"]

def get_zero_shot_clf():
    # zero-shot classifier (small, widely used)
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def get_toxicity_clf():
    return pipeline("text-classification", model="unitary/toxic-bert", return_all_scores=True)

def zero_shot_labels(text, clf):
    out = clf(text, candidate_labels=LABELS, multi_label=True)
    return dict(zip(out["labels"], out["scores"]))

def toxicity_score(text):
    tox = get_toxicity_clf()
    scores = tox(text)[0]  # list of dicts
    for s in scores:
        if s['label'].lower() == 'toxic':
            return float(s['score'])
    return max(s['score'] for s in scores)

def scam_score(text):
    scam_clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    labels = ["Likely scam", "Likely legitimate"]
    out = scam_clf(text, candidate_labels=labels)
    idx = out["labels"].index("Likely scam")
    return float(out["scores"][idx])

def analyze_text(text, clf):
    labels = zero_shot_labels(text, clf)
    tox = toxicity_score(text)
    scam = scam_score(text)
    return {"labels": labels, "toxicity": tox, "scam": scam}

def format_scores(d):
    items = sorted(((k, v) for k, v in d.items()), key=lambda kv: -kv[1])
    return {k: round(v, 3) for k, v in items}
