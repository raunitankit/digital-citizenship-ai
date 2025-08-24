from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class Rule:
    name: str
    keywords: List[str]
    weight: int
    explanation: str

# Simple category rules
RULES = [
    Rule("insult", ["idiot", "stupid", "dumb", "loser"], 3, "Potential insult"),
    Rule("profanity", ["hell", "crap", "wtf"], 2, "Contains profanity"),
    Rule("harassment", ["shut up", "nobody likes you", "go away"], 4, "Harassing phrase"),
    Rule("threat", ["hurt you", "beat you", "find you", "kill"], 6, "Possible threat"),
    Rule("exclusion", ["you can't sit", "not invited", "leave us"], 2, "Exclusionary language"),
]

SAFE_TONE = [
    "please", "thank you", "sorry", "let's work together", "can we", "may we"
]

def analyze_message(text: str) -> Dict:
    t = " " + text.lower().strip() + " "
    hits: List[Tuple[str, str]] = []
    score = 0

    for r in RULES:
        for kw in r.keywords:
            if f" {kw} " in t or kw in t.replace(",", " ").replace("!", " ").replace(".", " "):
                hits.append((r.name, kw))
                score += r.weight

    # Gentle reduction if polite markers present
    for s in SAFE_TONE:
        if s in t:
            score = max(0, score - 1)

    # Normalize to 0-10 scale
    max_score = sum(r.weight for r in RULES)
    risk = round(min(10, (score / max_score) * 10), 2)

    # Simple label
    if risk >= 7:
        label = "High Risk"
    elif risk >= 4:
        label = "Medium Risk"
    else:
        label = "Low Risk"

    reasons = []
    for rname, kw in hits:
        reason = next((r.explanation for r in RULES if r.name == rname), rname)
        reasons.append(f"{reason} (matched: “{kw}”)")

    return {
        "text": text,
        "label": label,
        "risk_score_0_10": risk,
        "matches": reasons or ["No risky patterns found"],
        "notes": "This is a simple rule-based demo. It can miss context or sarcasm."
    }

# --- Try it ---
samples = [
    "You're so dumb, nobody likes you!",
    "Hey, can we please keep the chat on-topic?",
    "I'm going to find you after school.",
    "That was a crap move, but let's work together on the project."
]

for s in samples:
    print(analyze_message(s))
