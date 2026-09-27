"""Synthetic data generator.

There is no public Indian dataset that links school records to confirmed
learning-disability outcomes, so the prototype runs on simulated schools.
The simulation is not random noise: each child gets a hidden "profile" and
every measurement is generated from that profile, using numbers from the
research:

  * ~8% of children have a Specific Learning Disability (SLD) (Scaria et al. 2023
    meta-analysis), split here into ~5% reading (dyslexia) and ~3% maths
    (dyscalculia), with a higher rate in boys (dyslexia meta-analysis: 14.5% vs 6%).
  * A large share of children are simply behind grade level (ASER 2024: only
    23.4% of Std III children in government schools read Std II text), more so
    when the home language differs from the medium of instruction.
  * Children who are behind respond well to targeted remedial teaching (TaRL
    evidence); children with SLD respond much more slowly. That difference in
    "response to intervention" is what lets the model tell the two apart.
  * Children with dyslexia typically understand a story READ TO them far better
    than one they read themselves (the listening vs reading contrast).

The hidden profile is never shown to the models. For past cohorts we save
two outcomes the models learn from: whether the child met the end-of-year
benchmark, and whether an SLD was later confirmed (with realistic misses).
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from config import CALC_BENCHMARK, CURRENT_YEAR, HISTORICAL_YEARS, INTERVENTION_WEEKS, ORF_BENCHMARK

SCHOOLS = [
    # school_id, name, district, medium, area, base share of children behind, share with home-language mismatch
    (1, "ZP Primary School, Wadgaon", "Pune", "Marathi", "rural", 0.42, 0.20),
    (2, "Govt. Primary School, Rampur", "Sitapur", "Hindi", "rural", 0.50, 0.25),
    (3, "Municipal School No. 12", "Nagpur", "Marathi", "urban", 0.33, 0.35),
    (4, "Sunrise English Medium School", "Nashik", "English", "semi-urban", 0.30, 0.70),
]

BOY_NAMES = ["Aarav", "Vivaan", "Aditya", "Arjun", "Sai", "Reyansh", "Krishna", "Ishaan", "Rohan", "Kabir",
             "Om", "Yash", "Aryan", "Pranav", "Harsh", "Siddharth", "Tanmay", "Rudra", "Ayaan", "Dev",
             "Nikhil", "Soham", "Atharva", "Karan", "Manav", "Parth", "Vedant", "Aniket", "Raju", "Shivam"]
GIRL_NAMES = ["Ananya", "Diya", "Aadhya", "Saanvi", "Pari", "Riya", "Isha", "Kavya", "Myra", "Sneha",
              "Anika", "Pooja", "Gauri", "Tanvi", "Neha", "Shruti", "Aarohi", "Meera", "Kiara", "Lakshmi",
              "Sakshi", "Radha", "Nandini", "Priya", "Janhavi", "Vaishnavi", "Khushi", "Mahi", "Siya", "Tara"]
SURNAME_INITIALS = list("ABCDGJKMNPRSTVWY")

OBS_ITEMS = {
    # item: probability observed in a given week, by hidden profile
    "guesses_words":     {"on_track": .08, "gap": .30, "sld_reading": .48, "sld_maths": .12},
    "letter_reversal":   {"on_track": .06, "gap": .18, "sld_reading": .38, "sld_maths": .10},
    "avoids_reading":    {"on_track": .08, "gap": .32, "sld_reading": .45, "sld_maths": .14},
    "symbol_confusion":  {"on_track": .06, "gap": .22, "sld_reading": .10, "sld_maths": .40},
    "finger_counting":   {"on_track": .12, "gap": .38, "sld_reading": .15, "sld_maths": .50},
    "attention_drift":   {"on_track": .12, "gap": .28, "sld_reading": .30, "sld_maths": .30},
}
OBS_WEEKS = 8

# Skill parameters by hidden profile: (mean, sd)
SKILLS = {
    #               decoding(x benchmark) oral lang /10  number sense /10  calc (x benchmark)  maths reasoning oral /10
    "on_track":    dict(dec=(1.00, .25), oral=(7.2, 1.5), num=(7.2, 1.6), calc=(0.95, .25), reas=(6.9, 1.6)),
    "gap":         dict(dec=(0.58, .22), oral=(5.3, 1.7), num=(4.9, 1.7), calc=(0.58, .24), reas=(5.0, 1.7)),
    "sld_reading": dict(dec=(0.38, .17), oral=(6.6, 1.7), num=(6.2, 1.7), calc=(0.80, .25), reas=(6.2, 1.7)),
    "sld_maths":   dict(dec=(0.85, .26), oral=(6.5, 1.6), num=(3.8, 1.5), calc=(0.40, .17), reas=(5.6, 1.7)),
}
# Weekly growth during a 6-week intervention, as a fraction of the starting score
RESPONSE = {"on_track": (.075, .04), "gap": (.07, .045), "sld_reading": (.028, .03), "sld_maths": (.025, .03)}


def _clip(x, lo, hi):
    return float(np.clip(x, lo, hi))


def _profile(rng, gender, behind_share, lang_mismatch):
    sld_r = 0.065 if gender == "M" else 0.035
    sld_m = 0.033 if gender == "M" else 0.027
    r = rng.random()
    if r < sld_r:
        return "sld_reading"
    if r < sld_r + sld_m:
        return "sld_maths"
    p_gap = behind_share + (0.15 if lang_mismatch else 0.0)
    return "gap" if rng.random() < p_gap else "on_track"


def generate(seed: int = 42, per_section: int = 36):
    rng = np.random.default_rng(seed)
    schools = pd.DataFrame(
        [s[:5] for s in SCHOOLS], columns=["school_id", "name", "district", "medium", "area"])

    students, probes, obs, outcomes, interventions, progress = [], [], [], [], [], []
    sid, iid = 1, 1

    for year in [*HISTORICAL_YEARS, CURRENT_YEAR]:
        is_current = year == CURRENT_YEAR
        start_year = int(year[:4])
        intervention_start = date(start_year, 8, 17) if is_current else date(start_year, 7, 21)
        # The current cohort is mid-cycle: 5 of 6 weekly probes entered so far
        weeks_recorded = 5 if is_current else INTERVENTION_WEEKS

        for school_id, _, _, _, _, behind_share, mismatch_share in SCHOOLS:
            for grade in (3, 4, 5):
                for section in ("A", "B"):
                    for _ in range(per_section):
                        gender = "M" if rng.random() < 0.5 else "F"
                        name = (rng.choice(BOY_NAMES if gender == "M" else GIRL_NAMES)
                                + " " + rng.choice(SURNAME_INITIALS) + ".")
                        mismatch = bool(rng.random() < mismatch_share)
                        prof = _profile(rng, gender, behind_share, mismatch)
                        sk = {k: rng.normal(*v) for k, v in SKILLS[prof].items()}
                        # A few SLD children also have weak oral language, so the contrast isn't a giveaway
                        if prof.startswith("sld") and rng.random() < 0.15:
                            sk["oral"] -= 2.0
                        sk["dec"] = max(sk["dec"], 0.05)
                        sk["calc"] = max(sk["calc"], 0.05)

                        if prof == "gap":
                            attendance = _clip(rng.normal(78, 10), 35, 100)
                        else:
                            attendance = _clip(rng.normal(89, 6), 55, 100)

                        orf_b, calc_b = ORF_BENCHMARK[grade], CALC_BENCHMARK[grade]
                        dec_c = min(sk["dec"], 1.3) / 1.3
                        orf = _clip(sk["dec"] * orf_b * rng.normal(1, .15), 0, 160)
                        letter_sound = _clip(35 + 62 * dec_c - (10 if prof == "sld_reading" else 0)
                                             + rng.normal(0, 6), 0, 100)
                        oral = _clip(sk["oral"], 0, 10)
                        reading_comp = _clip(oral * min(1.0, sk["dec"] / 0.9) ** 0.8 + rng.normal(0, 1.1), 0, 10)
                        spelling = _clip(20 + 70 * dec_c - (12 if prof == "sld_reading" else 0)
                                         + rng.normal(0, 7), 0, 100)
                        num = _clip(sk["num"] + rng.normal(0, .6), 0, 10)
                        calc = _clip(sk["calc"] * calc_b * rng.normal(1, .15), 0, 60)
                        reas = _clip(sk["reas"], 0, 10)
                        lang_marks = _clip(18 + 40 * dec_c + 3.2 * oral + rng.normal(0, 6), 0, 100)
                        math_marks = _clip(15 + 4 * num + 22 * min(sk["calc"], 1.3) / 1.3 + 1.6 * reas
                                           + rng.normal(0, 6), 0, 100)

                        students.append(dict(student_id=sid, school_id=school_id, academic_year=year,
                                             grade=grade, section=section, name=name, gender=gender,
                                             home_lang_matches=not mismatch))
                        probes.append(dict(student_id=sid, term="T1", attendance_pct=round(attendance, 1),
                                           lang_marks_pct=round(lang_marks, 1), math_marks_pct=round(math_marks, 1),
                                           orf_wcpm=round(orf), letter_sound_pct=round(letter_sound),
                                           listening_comp=round(oral, 1), reading_comp=round(reading_comp, 1),
                                           spelling_pct=round(spelling), number_sense=round(num, 1),
                                           calc_facts_per_min=round(calc), math_reasoning_oral=round(reas, 1)))

                        for week in range(1, OBS_WEEKS + 1):
                            for item, p in OBS_ITEMS.items():
                                # absent weeks: teacher can't observe; noisy teacher judgement
                                obs.append(dict(student_id=sid, week=week, item=item,
                                                observed=bool(rng.random() < p[prof])))

                        # Teacher's simple T1 flag starts a Tier 2 cycle (what schools do today with remedial classes)
                        growth = {}
                        for area, flagged, base, code in (
                            ("reading", orf < 0.6 * orf_b, orf, "R2-PHONICS" if letter_sound < 60 else "R2-TARL"),
                            ("maths", num < 5.0, num * 2, "M2-TARL"),  # maths probe: items correct out of 20
                        ):
                            if not flagged:
                                continue
                            slope = max(rng.normal(*RESPONSE[prof]), -0.01) * (attendance / 90) ** 2
                            growth[area] = slope
                            base = max(base, 2.0)
                            goal = base * (1 + 0.08 * INTERVENTION_WEEKS)  # expect ~8% growth a week
                            if is_current:
                                status = "active"
                            else:
                                status = "stepped_down" if slope >= 0.05 else "escalated"
                            interventions.append(dict(intervention_id=iid, student_id=sid, code=code, area=area,
                                                      start_date=intervention_start, weeks_planned=INTERVENTION_WEEKS,
                                                      baseline_score=round(base, 1), goal_score=round(goal, 1),
                                                      status=status))
                            for w in range(1, weeks_recorded + 1):
                                score = base * (1 + slope * w) * rng.normal(1, .09)
                                if area == "maths":
                                    score = min(score, 20)
                                progress.append(dict(intervention_id=iid, week=w, score=round(max(score, 0), 1)))
                            iid += 1

                        if not is_current:
                            # End-of-year: growth over the year + any gain from the intervention
                            base_growth = {"on_track": 1.12, "gap": 1.05, "sld_reading": 1.03, "sld_maths": 1.05}[prof]
                            end_dec = sk["dec"] * base_growth * (1 + 0.6 * INTERVENTION_WEEKS * growth.get("reading", 0))
                            end_num = num * base_growth * (1 + 0.6 * INTERVENTION_WEEKS * growth.get("maths", 0))
                            end_dec *= (attendance / 88) ** 0.5 * rng.lognormal(0, .22)   # home support, health, luck
                            end_num = end_num * rng.lognormal(0, .18)
                            met = bool(end_dec >= 0.75 and end_num >= 5.5)
                            if prof.startswith("sld"):
                                confirmed = bool(rng.random() < 0.85)   # some SLD children are never assessed
                            else:
                                confirmed = bool(rng.random() < 0.01)   # rare misdiagnosis
                            outcomes.append(dict(student_id=sid, benchmark_met_end_year=met, sld_confirmed=confirmed))
                        sid += 1

    return dict(
        schools=schools,
        students=pd.DataFrame(students),
        skill_probes=pd.DataFrame(probes),
        observations=pd.DataFrame(obs),
        outcomes=pd.DataFrame(outcomes),
        interventions=pd.DataFrame(interventions),
        progress_probes=pd.DataFrame(progress),
    )


if __name__ == "__main__":
    tables = generate()
    for k, v in tables.items():
        print(f"{k:18s} {len(v):>7,d} rows")
