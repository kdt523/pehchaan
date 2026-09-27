"""Database access. Works with PostgreSQL (DATABASE_URL) or the SQLite fallback."""
from functools import lru_cache

import pandas as pd
from sqlalchemy import create_engine, text

from config import DATABASE_URL, ROOT
from src.library import LIBRARY

LOAD_ORDER = ["schools", "students", "skill_probes", "observations", "outcomes",
              "intervention_library", "interventions", "progress_probes"]


@lru_cache(maxsize=1)
def engine():
    return create_engine(DATABASE_URL, future=True)


def backend_name() -> str:
    return "PostgreSQL" if engine().dialect.name == "postgresql" else "SQLite (fallback)"


def create_schema():
    ddl = (ROOT / "db" / "schema.sql").read_text()
    with engine().begin() as conn:
        for stmt in ddl.split(";"):
            # strip comment lines inside a statement
            clean = "\n".join(l for l in stmt.splitlines() if not l.strip().startswith("--"))
            if clean.strip():
                conn.execute(text(clean))


def load_tables(tables: dict):
    tables = dict(tables)
    tables["intervention_library"] = pd.DataFrame(LIBRARY)
    with engine().begin() as conn:
        for name in LOAD_ORDER:
            df = tables[name]
            if len(df):
                df.to_sql(name, conn, if_exists="append", index=False, method="multi", chunksize=2000)


def read(sql: str, **params) -> pd.DataFrame:
    with engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def execute(sql: str, **params):
    with engine().begin() as conn:
        conn.execute(text(sql), params)


# ---------- writes used by the app ----------

def add_progress_probe(intervention_id: int, week: int, score: float):
    execute("DELETE FROM progress_probes WHERE intervention_id = :i AND week = :w", i=intervention_id, w=week)
    execute("INSERT INTO progress_probes (intervention_id, week, score) VALUES (:i, :w, :s)",
            i=intervention_id, w=week, s=score)


def start_intervention(student_id: int, code: str, area: str, start_date, baseline: float, goal: float, weeks: int):
    next_id = int(read("SELECT COALESCE(MAX(intervention_id), 0) + 1 AS n FROM interventions")["n"][0])
    execute("""INSERT INTO interventions
               (intervention_id, student_id, code, area, start_date, weeks_planned, baseline_score, goal_score, status)
               VALUES (:id, :sid, :code, :area, :d, :w, :b, :g, 'active')""",
            id=next_id, sid=student_id, code=code, area=area, d=start_date, w=weeks, b=baseline, g=goal)
    return next_id


def set_intervention_status(intervention_id: int, status: str):
    execute("UPDATE interventions SET status = :s WHERE intervention_id = :i", s=status, i=intervention_id)
