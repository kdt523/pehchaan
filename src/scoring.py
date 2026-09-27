"""Scores a cohort with both models and attaches recommendations."""
import pandas as pd

from src import features, model
from src.explain import shap_values
from src.recommend import maths_level, reading_level, recommend


def score(year: str, bundle=None) -> dict:
    bundle = bundle or model.load()
    df = features.build(year)
    out = {}
    for key in ("gap", "sld"):
        m = bundle["models"][key]
        df[f"{key}_p"] = m.predict_proba(df)
        out[f"{key}_shap"] = pd.DataFrame(shap_values(m, df), index=df.index, columns=features.FEATURES)

    recs = {sid: recommend(r, r.gap_p, r.sld_p) for sid, r in df.iterrows()}
    df["tier"] = [recs[s]["tier"] for s in df.index]
    df["status"] = [recs[s]["status"] for s in df.index]
    df["reading_level"] = df.apply(reading_level, axis=1)
    df["maths_level"] = df.apply(maths_level, axis=1)
    out.update(df=df, recs=recs)
    return out
