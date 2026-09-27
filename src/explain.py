"""SHAP explanations turned into sentences a teacher can read.

SHAP tells us how much each feature pushed a child's score up or down. We show
the top few reasons as plain facts about the child ("Reads 12 words/min, grade
benchmark is 45"), marked as raising or lowering concern. We deliberately
don't claim these are causes; they are the signals the model leaned on.
"""
import numpy as np
import pandas as pd
import shap

from src.features import FEATURES, OBS_ITEMS

OBS_LABELS = {
    "guesses_words": "guesses words from the first letter",
    "letter_reversal": "reverses letters like b/d",
    "avoids_reading": "avoids reading aloud",
    "symbol_confusion": "confuses maths symbols (+, −, ×)",
    "finger_counting": "counts on fingers for simple sums",
    "attention_drift": "attention drifts during tasks",
}

FEATURE_LABELS = {
    "attendance_pct": "Attendance", "lang_marks_pct": "Language marks", "math_marks_pct": "Maths marks",
    "orf_ratio": "Reading fluency", "letter_sound_pct": "Letter sounds", "listening_comp": "Listening comprehension",
    "reading_comp": "Reading comprehension", "spelling_pct": "Spelling", "number_sense": "Number sense",
    "calc_ratio": "Calculation fluency", "math_reasoning_oral": "Maths reasoning (oral)",
    "listen_minus_read": "Listening vs reading gap", "reason_minus_calc": "Reasoning vs calculation gap",
    "reading_response": "Response to reading support", "maths_response": "Response to maths support",
    **{f"obs_{k}": f"Observed: {v}" for k, v in OBS_LABELS.items()},
}


def explainer(model):
    return shap.TreeExplainer(model.gb)


def shap_values(model, X: pd.DataFrame) -> np.ndarray:
    vals = explainer(model).shap_values(X[FEATURES])
    if isinstance(vals, list):          # older shap returns one array per class
        vals = vals[1]
    vals = np.asarray(vals)
    if vals.ndim == 3:                  # (n, features, classes)
        vals = vals[:, :, 1]
    return vals


def describe(feature: str, r: pd.Series) -> str:
    """A factual phrase about this child for one feature."""
    f = feature
    if f == "attendance_pct":
        return f"Attendance is {r.attendance_pct:.0f}%"
    if f == "lang_marks_pct":
        return f"Language term marks: {r.lang_marks_pct:.0f}%"
    if f == "math_marks_pct":
        return f"Maths term marks: {r.math_marks_pct:.0f}%"
    if f == "orf_ratio":
        return f"Reads {r.orf_wcpm:.0f} words/min (grade {r.grade} benchmark: {r.orf_benchmark})"
    if f == "letter_sound_pct":
        return f"Knows {r.letter_sound_pct:.0f}% of letter sounds"
    if f == "listening_comp":
        return f"Understands a story read aloud to them: {r.listening_comp:.1f}/10"
    if f == "reading_comp":
        return f"Understands a story they read themselves: {r.reading_comp:.1f}/10"
    if f == "spelling_pct":
        return f"Spelling accuracy {r.spelling_pct:.0f}%"
    if f == "number_sense":
        return f"Number sense score {r.number_sense:.1f}/10"
    if f == "calc_ratio":
        return f"Solves {r.calc_facts_per_min:.0f} maths facts/min (benchmark: {r.calc_benchmark})"
    if f == "math_reasoning_oral":
        return f"Solves word problems read aloud: {r.math_reasoning_oral:.1f}/10"
    if f == "listen_minus_read":
        if r.listen_minus_read >= 2:
            return (f"Understands stories {r.listen_minus_read:.1f} points better when they're read aloud "
                    f"than when reading alone (a typical dyslexia pattern)")
        return f"Listening and reading comprehension are at a similar level ({r.listen_minus_read:+.1f})"
    if f == "reason_minus_calc":
        if r.reason_minus_calc >= 0.3:
            return "Reasons well in maths, but calculation is much weaker"
        return "Maths reasoning and calculation are at a similar level"
    if f in ("reading_response", "maths_response"):
        area = f.split("_")[0]
        if not r[f"in_{area}_support"]:
            return f"Hasn't needed a {area} support group so far"
        weeks = int(r[f"{area}_weeks"])
        return (f"Grew {r[f] * 100:.1f}% per week in {weeks} weeks of {area} support "
                f"(children who are just behind typically grow ~8%)")
    if f.startswith("obs_"):
        item = f[4:]
        n = round(r[f] * r.obs_weeks)
        return f"Teacher noticed '{OBS_LABELS[item]}' in {n} of {int(r.obs_weeks)} weeks"
    return f"{FEATURE_LABELS.get(f, f)}: {r[f]}"


def reasons(row: pd.Series, shap_row: np.ndarray, top_k: int = 4) -> list[dict]:
    order = np.argsort(-np.abs(shap_row))
    out = []
    for i in order:
        if len(out) == top_k or abs(shap_row[i]) < 0.05:
            break
        f = FEATURES[i]
        # "rarely seen" raising concern (or "often seen" lowering it) is true to the model but
        # confusing for a teacher, so we leave those out of the plain-language list
        if f.startswith("obs_") and ((shap_row[i] > 0 and row[f] < 0.25) or (shap_row[i] < 0 and row[f] >= 0.5)):
            continue
        text = describe(f, row)
        if f == "attendance_pct" and shap_row[i] > 0 and row.attendance_pct >= 85:
            # struggling despite regular attendance = "unexpected" difficulty, a classic SLD marker
            text = f"Attends regularly ({row.attendance_pct:.0f}%), so missed lessons don't explain the difficulty"
        out.append(dict(feature=f, label=FEATURE_LABELS.get(f, f), impact=float(shap_row[i]),
                        raises=bool(shap_row[i] > 0), text=text))
    return out
