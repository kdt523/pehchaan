"""Project-wide settings for the Pehchaan prototype."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Set DATABASE_URL to use PostgreSQL, e.g.
#   postgresql+psycopg2://postgres:<password>@localhost:5432/pehchaan
# Without it the prototype falls back to a local SQLite file so it runs anywhere.
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{(DATA_DIR / 'pehchaan.db').as_posix()}")

CURRENT_YEAR = "2026-27"     # cohort we score in the app
HISTORICAL_YEARS = ["2023-24", "2024-25", "2025-26"]  # past cohorts with known outcomes, used for training

# Oral reading fluency benchmarks (words correct per minute) and maths fact
# fluency benchmarks (facts per minute) by grade. These are prototype
# assumptions, loosely based on Indian-language ORF studies; a real
# deployment would use state/NIPUN benchmarks for each language.
ORF_BENCHMARK = {3: 45, 4: 60, 5: 75}
CALC_BENCHMARK = {3: 12, 4: 16, 5: 20}

# Decision thresholds (on calibrated probabilities)
GAP_THRESHOLD = 0.50
SLD_THRESHOLD = 0.30   # lower on purpose: missing a child costs more than an extra check

INTERVENTION_WEEKS = 6
