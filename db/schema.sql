-- Pehchaan prototype schema. Written to run on both PostgreSQL and SQLite.
-- One longitudinal record per child: probes, observations, interventions and
-- their weekly progress all hang off students.student_id.

DROP TABLE IF EXISTS progress_probes;
DROP TABLE IF EXISTS interventions;
DROP TABLE IF EXISTS observations;
DROP TABLE IF EXISTS outcomes;
DROP TABLE IF EXISTS skill_probes;
DROP TABLE IF EXISTS students;
DROP TABLE IF EXISTS schools;
DROP TABLE IF EXISTS intervention_library;

CREATE TABLE schools (
    school_id   INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    district    TEXT,
    medium      TEXT NOT NULL,          -- medium of instruction
    area        TEXT                    -- rural / urban / semi-urban
);

CREATE TABLE students (
    student_id          INTEGER PRIMARY KEY,
    school_id           INTEGER NOT NULL REFERENCES schools(school_id),
    academic_year       TEXT NOT NULL,
    grade               INTEGER NOT NULL,
    section             TEXT NOT NULL,
    name                TEXT NOT NULL,
    gender              TEXT,           -- used ONLY for fairness audit, never as a model input
    home_lang_matches   BOOLEAN         -- used ONLY for context/audit, never as a model input
);

-- Term-level record: attendance, term-test marks and the 2-minute skill probes
CREATE TABLE skill_probes (
    student_id              INTEGER NOT NULL REFERENCES students(student_id),
    term                    TEXT NOT NULL,      -- 'T1' baseline (June-July)
    attendance_pct          REAL,
    lang_marks_pct          REAL,
    math_marks_pct          REAL,
    orf_wcpm                REAL,               -- oral reading fluency, words correct per minute
    letter_sound_pct        REAL,
    listening_comp          REAL,               -- 0-10, story read aloud to the child
    reading_comp            REAL,               -- 0-10, child reads the story
    spelling_pct            REAL,
    number_sense            REAL,               -- 0-10
    calc_facts_per_min      REAL,
    math_reasoning_oral     REAL,               -- 0-10, word problems read aloud
    PRIMARY KEY (student_id, term)
);

-- Weekly 2-minute teacher observation check (PRASHAST-inspired items)
CREATE TABLE observations (
    student_id  INTEGER NOT NULL REFERENCES students(student_id),
    week        INTEGER NOT NULL,
    item        TEXT NOT NULL,
    observed    BOOLEAN NOT NULL,
    PRIMARY KEY (student_id, week, item)
);

-- Only known for past cohorts. This is what the models learn from
CREATE TABLE outcomes (
    student_id              INTEGER PRIMARY KEY REFERENCES students(student_id),
    benchmark_met_end_year  BOOLEAN NOT NULL,
    sld_confirmed           BOOLEAN NOT NULL    -- confirmed later by a qualified professional
);

CREATE TABLE intervention_library (
    code            TEXT PRIMARY KEY,
    area            TEXT NOT NULL,      -- reading / maths / attendance / attention / language
    tier            INTEGER NOT NULL,   -- 1 = whole class, 2 = small group, 3 = specialist
    title           TEXT NOT NULL,
    what_to_do      TEXT NOT NULL,
    duration        TEXT,
    materials       TEXT,
    evidence        TEXT
);

CREATE TABLE interventions (
    intervention_id INTEGER PRIMARY KEY,
    student_id      INTEGER NOT NULL REFERENCES students(student_id),
    code            TEXT NOT NULL REFERENCES intervention_library(code),
    area            TEXT NOT NULL,      -- reading / maths (what the weekly probe measures)
    start_date      DATE NOT NULL,
    weeks_planned   INTEGER NOT NULL,
    baseline_score  REAL NOT NULL,
    goal_score      REAL NOT NULL,
    status          TEXT NOT NULL       -- active / completed / escalated / stepped_down
);

CREATE TABLE progress_probes (
    intervention_id INTEGER NOT NULL REFERENCES interventions(intervention_id),
    week            INTEGER NOT NULL,
    score           REAL NOT NULL,
    PRIMARY KEY (intervention_id, week)
);
