from collections import Counter
import re
import unicodedata


def normalize_answer(text):
    text = unicodedata.normalize("NFC",text).lower()
    return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith("P"))


def answer_scores(prediction, answers):
    """Project-defined character F1, not an official KorQuAD scoring implementation."""
    pred = normalize_answer(prediction)
    best_em, best_f1 = 0.0, 0.0
    for answer in answers:
        gold = normalize_answer(answer)
        common = sum((Counter(pred)&Counter(gold)).values())
        f1 = (2*common/(len(pred)+len(gold))) if pred or gold else 1.0
        best_em = max(best_em,float(pred==gold)); best_f1 = max(best_f1,f1)
    return {"exact_match":best_em,"character_f1":best_f1}


def repetition(text, n=4):
    if n < 1: raise ValueError("n must be positive")
    pieces = [text[i:i+n] for i in range(max(0,len(text)-n+1))]
    return 0.0 if not pieces else 1-len(set(pieces))/len(pieces)
