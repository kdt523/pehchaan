"""Trains the two screening models on past cohorts.

  gap  - "Learning-gap risk": will this child miss the end-of-year foundational
         benchmark if nothing changes?
  sld  - "SLD-pattern signal": does this child's profile look like the children
         who later had a Specific Learning Disability confirmed?

For each we compare an interpretable baseline (logistic regression) with
gradient boosting, calibrate the boosted model on a held-out split so that
"0.7" means roughly 70%, and run a fairness audit by gender and medium.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import GAP_THRESHOLD, HISTORICAL_YEARS, MODEL_DIR, SLD_THRESHOLD
from src.features import FEATURES, training_frame

TARGETS = {
    "gap": dict(target="gap_target", threshold=GAP_THRESHOLD, title="Learning-gap risk"),
    "sld": dict(target="sld_target", threshold=SLD_THRESHOLD, title="SLD-pattern signal"),
}
BUNDLE_PATH = MODEL_DIR / "models.joblib"


class CalibratedModel:
    """Gradient boosting + isotonic calibration. SHAP explains `.gb`; the app shows calibrated probabilities."""

    def __init__(self, gb, iso):
        self.gb, self.iso = gb, iso

    def predict_proba(self, X):
        raw = self.gb.predict_proba(X[FEATURES])[:, 1]
        # a screening tool should never claim certainty either way
        return np.clip(self.iso.predict(raw), 0.02, 0.98)


def _metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(auc=roc_auc_score(y, p), recall=recall_score(y, pred, zero_division=0),
                precision=precision_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0),
                brier=brier_score_loss(y, p), flag_rate=pred.mean(), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def _fairness(test: pd.DataFrame, y, p, threshold):
    rows = []
    pred = (p >= threshold).astype(int)
    for attr, label in [("gender", "Gender"), ("medium", "Medium of instruction")]:
        for g in sorted(test[attr].unique()):
            m = (test[attr] == g).values
            yt, yp = y[m], pred[m]
            pos, neg = yt.sum(), (1 - yt).sum()
            rows.append(dict(attribute=label, group=g, n=int(m.sum()), base_rate=float(yt.mean()),
                             flag_rate=float(yp.mean()),
                             recall=float((yp[yt == 1]).mean()) if pos else np.nan,
                             false_positive_rate=float((yp[yt == 0]).mean()) if neg else np.nan))
    return pd.DataFrame(rows)


def train(seed: int = 7):
    df = training_frame(HISTORICAL_YEARS)
    bundle = dict(features=FEATURES, trained_on=", ".join(HISTORICAL_YEARS), n_students=len(df), models={}, report={})

    for key, spec in TARGETS.items():
        y = df[spec["target"]].values
        X = df[FEATURES]
        idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=0.25, stratify=y, random_state=seed)
        idx_fit, idx_cal = train_test_split(idx_train, test_size=0.25, stratify=y[idx_train], random_state=seed)

        baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        baseline.fit(X.iloc[idx_train], y[idx_train])

        gb = GradientBoostingClassifier(n_estimators=250, max_depth=3, learning_rate=0.05,
                                        subsample=0.8, random_state=seed)
        gb.fit(X.iloc[idx_fit], y[idx_fit])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(gb.predict_proba(X.iloc[idx_cal])[:, 1], y[idx_cal])
        model = CalibratedModel(gb, iso)

        test = df.iloc[idx_test]
        p_gb = model.predict_proba(test)
        p_lr = baseline.predict_proba(test[FEATURES])[:, 1]
        thr = spec["threshold"]

        # Calibration curve on the test split
        bins = pd.cut(p_gb, bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0], include_lowest=True)
        calib = (pd.DataFrame(dict(bin=bins, p=p_gb, y=y[idx_test]))
                 .groupby("bin", observed=True).agg(predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
                 .reset_index(drop=True))

        bundle["models"][key] = model
        bundle["report"][key] = dict(
            title=spec["title"], threshold=thr, prevalence=float(y.mean()),
            n_train=len(idx_fit), n_calibration=len(idx_cal), n_test=len(idx_test),
            gradient_boosting=_metrics(y[idx_test], p_gb, thr),
            logistic_regression=_metrics(y[idx_test], p_lr, thr if key == "sld" else 0.5),
            fairness=_fairness(test, y[idx_test], p_gb, thr),
            calibration=calib,
        )

    joblib.dump(bundle, BUNDLE_PATH)
    return bundle


def load():
    return joblib.load(BUNDLE_PATH)


if __name__ == "__main__":
    b = train()
    for k, r in b["report"].items():
        g, l = r["gradient_boosting"], r["logistic_regression"]
        print(f"{r['title']}: prevalence {r['prevalence']:.1%} | GB AUC {g['auc']:.3f} recall {g['recall']:.2f} "
              f"precision {g['precision']:.2f} | LR AUC {l['auc']:.3f}")
