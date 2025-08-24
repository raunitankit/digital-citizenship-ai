
def feedback_for_labels(label_scores: dict) -> str:
    if not label_scores:
        return "Good effort—let's keep improving our digital choices."
    top = max(label_scores.items(), key=lambda kv: kv[1])[0]
    bank = {
        "Safe behavior": "Great choice—this protects your privacy and keeps you safe online.",
        "Risky behavior": "Think twice—this choice could expose personal info or lead to unsafe interactions.",
        "Respectful": "Nice! That supports a positive, kind online community.",
        "Disrespectful": "Consider how this might make others feel—let's try a more considerate approach."
    }
    return bank.get(top, "Good effort—let's keep improving!")
