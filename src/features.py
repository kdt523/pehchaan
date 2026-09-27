"""Turns raw DB tables into one feature row per child.

Three kinds of features, following the research notes:
  * level     - how the child is doing (marks, probes, attendance)
  * contrast  - uneven profiles (understands a story read aloud but not when reading
                alone; reasons well in maths but can't calculate). Typical of SLD.
  * response  - how fast the child improved during a support cycle. Children who
                are simply behind usually respond quickly; SLD children often don't.
Gender, medium and home language are kept as metadata for the fairness audit
and are never used as model inputs.
"""
import numpy as np
import pandas as pd

from config import CALC_BENCHMARK, ORF_BENCHMARK
from src import db

OBS_ITEMS = ["guesses_words", "letter_reversal", "avoids_reading",
             "symbol_confusion", "finger_counting", "attention_drift"]

LEVEL = ["attendance_pct", "lang_marks_pct", "math_marks_pct", "orf_ratio", "letter_sound_pct",
         "listening_comp", "reading_comp", "spelling_pct", "number_sense", "calc_ratio", "math_reasoning_oral"]
CONTRAST = ["listen_minus_read", "reason_minus_calc"]
OBSERVATION = [f"obs_{i}" for i in OBS_ITEMS]
RESPONSE = ["reading_response", "maths_response"]
FEATURES = LEVEL + CONTRAST + OBSERVATION + RESPONSE

EXPECTED_GROWTH = 0.08   # value used for children who never needed a support cycle
MIN_WEEKS_FOR_SLOPE = 3


def response_slopes(year: str) -> pd.DataFrame:
    """Weekly growth rate (fraction of baseline per week) for each child's latest cycle per area."""
    iv = db.read("""SELECT i.intervention_id, i.student_id, i.area, i.code, i.baseline_score, i.goal_score,
                           i.weeks_planned, i.status, i.start_date
                    FROM interventions i JOIN students s ON s.student_id = i.student_id
                    WHERE s.academic_year = :y""", y=year)
    if iv.empty:
        return pd.DataFrame(columns=["student_id"])
    pp = db.read("""SELECT p.intervention_id, p.week, p.score FROM progress_probes p
                    JOIN interventions i ON i.intervention_id = p.intervention_id
                    JOIN students s ON s.student_id = i.student_id WHERE s.academic_year = :y""", y=year)
    rows = []
    for _, r in iv.iterrows():
        pts = pp[pp.intervention_id == r.intervention_id].sort_values("week")
        slope, weeks = np.nan, len(pts)
        if weeks >= MIN_WEEKS_FOR_SLOPE and r.baseline_score > 0:
            w = np.r_[0, pts.week.values]
            s = np.r_[1.0, pts.score.values / r.baseline_score]
            slope = float(np.polyfit(w, s, 1)[0])
        rows.append(dict(student_id=r.student_id, area=r.area, intervention_id=r.intervention_id,
                         slope=slope, weeks_done=weeks, start_date=r.start_date))
    df = pd.DataFrame(rows).sort_values(["student_id", "area", "start_date", "intervention_id"])
    latest = df.groupby(["student_id", "area"]).tail(1)
    wide = latest.pivot(index="student_id", columns="area", values=["slope", "weeks_done"])
    wide.columns = [f"{a}_{m}" for m, a in wide.columns]
    wide = wide.rename(columns={"reading_slope": "reading_response", "maths_slope": "maths_response",
                                "reading_weeks_done": "reading_weeks", "maths_weeks_done": "maths_weeks"})
    return wide.reset_index()


def build(year: str) -> pd.DataFrame:
    meta = db.read("""SELECT s.student_id, s.name, s.grade, s.section, s.gender, s.home_lang_matches,
                             s.academic_year, sc.school_id, sc.name AS school, sc.medium, sc.area AS school_area
                      FROM students s JOIN schools sc ON sc.school_id = s.school_id
                      WHERE s.academic_year = :y""", y=year)
    probes = db.read("""SELECT p.* FROM skill_probes p JOIN students s ON s.student_id = p.student_id
                        WHERE s.academic_year = :y AND p.term = 'T1'""", y=year).drop(columns=["term"])
    obs = db.read("""SELECT o.student_id, o.item, AVG(CASE WHEN o.observed THEN 1.0 ELSE 0.0 END) AS rate,
                            COUNT(*) AS weeks
                     FROM observations o JOIN students s ON s.student_id = o.student_id
                     WHERE s.academic_year = :y GROUP BY o.student_id, o.item""", y=year)
    obs_wide = obs.pivot(index="student_id", columns="item", values="rate").add_prefix("obs_").reset_index()
    obs_weeks = obs.groupby("student_id")["weeks"].max().rename("obs_weeks").reset_index()

    df = (meta.merge(probes, on="student_id", how="left")
              .merge(obs_wide, on="student_id", how="left")
              .merge(obs_weeks, on="student_id", how="left")
              .merge(response_slopes(year), on="student_id", how="left"))

    df["orf_benchmark"] = df.grade.map(ORF_BENCHMARK)
    df["calc_benchmark"] = df.grade.map(CALC_BENCHMARK)
    df["orf_ratio"] = df.orf_wcpm / df.orf_benchmark
    df["calc_ratio"] = df.calc_facts_per_min / df.calc_benchmark
    df["listen_minus_read"] = df.listening_comp - df.reading_comp
    df["reason_minus_calc"] = df.math_reasoning_oral / 10 - df.calc_ratio.clip(upper=1.5) / 1.5

    for col in ["reading_response", "maths_response", "reading_weeks", "maths_weeks"]:
        if col not in df:
            df[col] = np.nan
    df["in_reading_support"] = df.reading_response.notna()
    df["in_maths_support"] = df.maths_response.notna()
    df[RESPONSE] = df[RESPONSE].fillna(EXPECTED_GROWTH)
    df[OBSERVATION] = df[OBSERVATION].fillna(0.0)
    return df.set_index("student_id")


def training_frame(years: list[str]) -> pd.DataFrame:
    df = pd.concat([build(y) for y in years])
    out = db.read("SELECT * FROM outcomes").set_index("student_id")
    df = df.join(out, how="inner")
    df["gap_target"] = (~df.benchmark_met_end_year.astype(bool)).astype(int)
    df["sld_target"] = df.sld_confirmed.astype(bool).astype(int)
    return df
