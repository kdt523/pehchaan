# Pehchaan: early identification of learning difficulties (prototype)

*Pehchaan (पहचान) means "recognition".* Mid-semester prototype for the Fujitsu problem statement:
**identify learning difficulties early and recommend suitable academic interventions.**

Stack: Python · scikit-learn · SHAP · PostgreSQL (SQLite fallback) · Streamlit · Plotly

---

## The idea in one paragraph

Most children who struggle in primary school are simply **behind** (missed lessons, language mismatch,
weak foundations) and catch up quickly with good remedial teaching. A smaller group (about 8%) has a
**Specific Learning Disability** (dyslexia, dyscalculia). On a marksheet both groups look the same, so
SLD children spend years in the wrong kind of help. Pehchaan gives teachers **two separate signals**:
learning-gap risk and SLD-pattern signal. It explains both in plain language with SHAP, recommends tiered
interventions, and then tracks **whether the child responds**. A child who doesn't improve after a good
6-week support cycle *and* shows the SLD pattern is flagged for referral, with a referral pack ready
for the psychologist.

## What data are we using?

**Synthetic data.** No public Indian dataset links school records (marks, attendance, reading probes)
to confirmed learning-disability outcomes. So `src/generate_data.py` simulates 4 schools × classes
3–5 × 2 sections: 3 past cohorts (2,592 students, with known outcomes) for training, and the current
cohort (864 students) that the app scores.

It's not random noise. Every number in the simulation is anchored to published research:

| Simulation assumption | Source |
|---|---|
| ~8% SLD prevalence, split ~5% reading / ~3% maths, more common in boys | Scaria et al. 2023; IOPN 2022 meta-analyses |
| Large share of children behind grade level, higher with home-language mismatch | ASER 2024 |
| Children who are behind grow fast with level-based teaching; SLD children grow slowly | J-PAL/Pratham TaRL RCTs; RTI/MTSS literature |
| Dyslexic children understand stories read aloud far better than ones they read | Definition of dyslexia (decoding vs comprehension) |
| Teacher observation items | NCERT PRASHAST SLD checklist (reworded) |
| Label noise: ~15% of SLD children never get confirmed | Under-identification in India |

**Real datasets for the next milestone:** UCI *Student Performance* (grades + absences) to validate the
learning-gap model; the *Dytective* dyslexia dataset from Rello et al. 2020 (PLOS ONE) for the SLD-signal
side; and ideally a partner school's anonymised records + PRASHAST results.

We say this openly in the demo: the **pipeline** is what we're showing, not real-world accuracy.

## Run it

```bash
# 1. Python 3.11 or 3.12 recommended (shap wheels)
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Mac/Linux)
pip install -r requirements.txt

# 2. (Optional) use PostgreSQL instead of the SQLite fallback
createdb -U postgres pehchaan
set DATABASE_URL=postgresql+psycopg2://postgres:<password>@localhost:5432/pehchaan

# 3. Generate data, load DB, train models (~1 min)
python setup_prototype.py

# 4. Launch
streamlit run app.py
```

## How it works

```
 skill probes, marks,          weekly 2-min teacher         6-week support cycles
 attendance (T1)               observation checklist        + weekly progress probes
        \                             |                            /
         └──────────────►  PostgreSQL (one timeline per child)  ◄──┘
                                      │
                         features.py: level + contrast + response features
                                      │
              ┌───────────────────────┴───────────────────────┐
     Learning-gap risk model                          SLD-pattern signal model
     (GradientBoosting + isotonic calibration, compared with logistic regression)
              └───────────────────────┬───────────────────────┘
                          SHAP → plain-language reasons
                                      │
                recommend.py: rule-based tiered interventions (RTI/MTSS)
                                      │
      Streamlit: class view · student profile · support tracker · block dashboard · model & fairness
```

**Feature groups** (`src/features.py`)
- *Level*: attendance, marks, reading fluency vs benchmark, letter sounds, comprehension, number sense…
- *Contrast*: listening minus reading comprehension; maths reasoning minus calculation (uneven profiles)
- *Response*: weekly growth during a support cycle (the Response to Intervention signal)
- *Observation*: how often the teacher ticked each checklist item

**Not model inputs:** gender, medium, home language, caste, religion, income. Gender and medium are used only
in the fairness audit.

## Demo script for the review (5 minutes)

1. **Class overview → "The two signals" tab.** Show the scatter: most flagged kids are bottom-right
   (behind, will catch up); a few are top-left/top-right (SLD pattern). *This is the core idea.*
2. **Students tab → click a red "Refer (Tier 3)" child.** Walk through the "Why" cards: e.g. "understands
   stories 4 points better when read aloud", "grew 1.5% per week in 5 weeks of reading support".
3. Scroll to **Recommended support** (tiered cards with evidence) and the **progress chart** vs goal line.
   Enter a weekly probe score live to show it saves to the database and rescoring happens.
4. **Share → Referral pack**: the document parents currently can't get from anyone.
5. **Support tracker**: non-responders with SLD pattern separate clearly from responders.
6. **Model & fairness**: AUC vs logistic baseline, calibration, SHAP importance, recall/FPR by gender and medium.
7. **Block dashboard**: where the one special educator should go first.

## Project structure

```
pehchaan/
├── app.py                 Streamlit app (6 views)
├── setup_prototype.py     schema → synthetic data → DB → train
├── config.py              DB URL, benchmarks, thresholds
├── db/schema.sql          PostgreSQL/SQLite schema
└── src/
    ├── generate_data.py   research-grounded synthetic data
    ├── db.py              SQLAlchemy access + writes from the app
    ├── features.py        level / contrast / response / observation features
    ├── model.py           two models, calibration, metrics, fairness audit
    ├── explain.py         SHAP → teacher-readable sentences
    ├── recommend.py       tiered intervention rules + TaRL level grouping
    ├── library.py         intervention cards (editable by a counsellor)
    ├── scoring.py         scores a cohort end to end
    └── reports.py         parent summary + referral pack
```

## Known limitations / next steps

- Synthetic data; validate on UCI + Dytective, then a real school pilot.
- Reading benchmarks (45/60/75 wcpm for classes 3/4/5) are placeholders; need per-language norms.
- Only term-1 probes are used as features; add term-over-term trends.
- No login/roles yet (teacher vs counsellor vs parent views); no consent log yet (DPDP Rules 2025).
- Parent summary is English only; add Hindi/Marathi templates.
- Referral pack is Markdown; export to PDF.
