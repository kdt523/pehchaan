"""Parent summary and referral pack text."""
from datetime import date

import pandas as pd

from src.explain import OBS_LABELS


def _strengths(r) -> list[str]:
    s = []
    if r.listening_comp >= 6.5:
        s.append("understands stories well when they are read aloud")
    if r.math_reasoning_oral >= 6.5:
        s.append("thinks through maths problems well when they are explained out loud")
    if r.orf_ratio >= 0.9:
        s.append("reads fluently for their class")
    if r.number_sense >= 7:
        s.append("has a good feel for numbers")
    if r.attendance_pct >= 90:
        s.append("comes to school regularly")
    return s or ["is trying hard and coming to class"]


HOME_ACTIVITIES = {
    "reading": ["Read a short story aloud together for 10 minutes and talk about it.",
                "Play 'I spy' with sounds: 'I spy something that starts with /m/'.",
                "Point out words on signboards and packets and sound them out together."],
    "maths": ["Count real things together: rotis, spoons, steps.",
              "While shopping, ask 'if we buy 3 of these, how many is that?'",
              "Play simple dice or card games that need adding."],
    "attendance": ["Help your child come to school every day; even a few missed days make catching up harder."],
}


def parent_summary(r, rec) -> str:
    first = r["name"].split()[0]
    areas = {c["area"] for c in rec["cards"]}
    lines = [f"**Dear parent of {first},**", "",
             f"{first} {', and '.join(_strengths(r))}. We are happy with this.", ""]
    working = []
    if "reading" in areas:
        working.append("reading (matching letters to sounds)")
    if "maths" in areas:
        working.append("maths (numbers and calculation)")
    if "language" in areas:
        working.append("building vocabulary")
    if working:
        lines += [f"At school, {first} is getting extra small-group practice in {', '.join(working)}. "
                  "Many children need this at some stage, and it helps most when it starts early.", ""]
    else:
        lines += [f"{first} is doing well. Keep encouraging them at home.", ""]
    acts = [a for k in ("reading", "maths", "attendance") if k in areas for a in HOME_ACTIVITIES[k][:2]]
    if acts:
        lines += ["**Things you can do at home (10 minutes a day):**"] + [f"- {a}" for a in acts] + [""]
    lines += ["Please meet the class teacher any time if you have questions.", "", "*Class teacher*"]
    return "\n".join(lines)


def referral_pack(r, rec, reasons: list, interventions: pd.DataFrame, progress: pd.DataFrame) -> str:
    L = [f"# Referral pack: {r['name']}",
         f"*Generated {date.today():%d %b %Y} by Pehchaan (screening support tool, not a diagnosis)*", "",
         "## Student",
         f"- School: {r.school} ({r.medium} medium)",
         f"- Class: {r.grade}{r.section}, academic year {r.academic_year}",
         f"- Attendance so far: {r.attendance_pct:.0f}%", "",
         "## Why the school is referring",
         f"- Screening status: **{rec['status']}**",
         f"- SLD-pattern signal: {r.sld_p:.0%} · Learning-gap risk: {r.gap_p:.0%}"]
    L += [f"- {x['text']}" for x in reasons if x["raises"]]
    L += ["", "## Skill probes (term 1)",
          "| Skill | Score |", "|---|---|",
          f"| Oral reading fluency | {r.orf_wcpm:.0f} wcpm (benchmark {r.orf_benchmark}) |",
          f"| Letter sounds known | {r.letter_sound_pct:.0f}% |",
          f"| Listening comprehension | {r.listening_comp:.1f}/10 |",
          f"| Reading comprehension | {r.reading_comp:.1f}/10 |",
          f"| Spelling | {r.spelling_pct:.0f}% |",
          f"| Number sense | {r.number_sense:.1f}/10 |",
          f"| Maths facts per minute | {r.calc_facts_per_min:.0f} (benchmark {r.calc_benchmark}) |",
          f"| Maths reasoning (oral) | {r.math_reasoning_oral:.1f}/10 |",
          "", f"## Teacher observations ({int(r.obs_weeks)} weekly checks)"]
    for k, label in OBS_LABELS.items():
        n = round(r[f"obs_{k}"] * r.obs_weeks)
        if n:
            L.append(f"- {label}: {n} of {int(r.obs_weeks)} weeks")
    L += ["", "## Support already given and response"]
    if interventions.empty:
        L.append("- None recorded yet.")
    for _, iv in interventions.iterrows():
        pts = progress[progress.intervention_id == iv.intervention_id].sort_values("week")
        scores = ", ".join(f"wk{int(w)}: {s:g}" for w, s in zip(pts.week, pts.score))
        L.append(f"- **{iv.title}** ({iv.area}), started {iv.start_date}, status: {iv.status}. "
                 f"Baseline {iv.baseline_score:g}, goal {iv.goal_score:g}. Weekly scores: {scores or 'none yet'}")
    L += ["", "## Accommodations already in place"]
    L += [f"- {c['title']}" for c in rec["cards"] if c["tier"] == 1] or ["- None"]
    L += ["", "---", "*This pack summarises school observations to support a professional assessment. "
          "It does not diagnose a learning disability.*"]
    return "\n".join(L)
