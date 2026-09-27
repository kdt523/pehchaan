"""Pehchaan: early identification of learning difficulties (prototype).

    streamlit run app.py
"""
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import CURRENT_YEAR, GAP_THRESHOLD, INTERVENTION_WEEKS, MODEL_DIR, SLD_THRESHOLD
from src import db, model, scoring
from src.explain import FEATURE_LABELS, OBS_LABELS, reasons
from src.features import FEATURES, OBS_ITEMS
from src.library import LIBRARY_BY_CODE
from src.recommend import NON_RESPONSE
from src.reports import parent_summary, referral_pack

st.set_page_config(page_title="Pehchaan · Learning difficulty screening", page_icon="🌱", layout="wide")

STATUS_COLORS = {"On track": "#2e7d32", "Watch (Tier 1)": "#f9a825",
                 "Needs support (Tier 2)": "#ef6c00", "Refer (Tier 3)": "#c62828"}
TIER_BADGE = {1: "🟡 Tier 1 · whole class", 2: "🟠 Tier 2 · small group", 3: "🔴 Tier 3 · specialist"}
PAGES = ["🏫 Class overview", "👤 Student profile", "📈 Support tracker",
         "🏛️ Block dashboard", "🧠 Model & fairness", "🗄️ Data"]

st.markdown("""
<style>
.badge {display:inline-block;padding:2px 10px;border-radius:12px;color:white;font-size:0.85rem;font-weight:600}
.reason-up {color:#c62828;font-weight:600}
.reason-down {color:#2e7d32;font-weight:600}
.small {font-size:0.85rem;color:#666}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- setup & data
def run_setup():
    from setup_prototype import main
    with st.spinner("Generating synthetic data, loading the database and training models (about a minute)…"):
        main()
    st.cache_data.clear()
    st.cache_resource.clear()


if not (MODEL_DIR / "models.joblib").exists():
    st.title("🌱 Pehchaan")
    st.info("First run: no data or models yet.")
    if st.button("Set up prototype (synthetic data + train models)", type="primary"):
        run_setup()
        st.rerun()
    st.stop()

st.session_state.setdefault("version", 0)


@st.cache_resource
def get_bundle():
    return model.load()


@st.cache_data(show_spinner="Scoring students…")
def get_scores(year: str, version: int):
    return scoring.score(year, get_bundle())


@st.cache_data
def get_interventions(version: int):
    iv = db.read("""SELECT i.*, s.academic_year FROM interventions i
                    JOIN students s ON s.student_id = i.student_id WHERE s.academic_year = :y""", y=CURRENT_YEAR)
    iv["title"] = iv.code.map(lambda c: LIBRARY_BY_CODE[c]["title"])
    pp = db.read("""SELECT p.* FROM progress_probes p JOIN interventions i ON i.intervention_id = p.intervention_id
                    JOIN students s ON s.student_id = i.student_id WHERE s.academic_year = :y""", y=CURRENT_YEAR)
    return iv, pp


def bump():
    st.session_state.version += 1


S = get_scores(CURRENT_YEAR, st.session_state.version)
DF, RECS = S["df"], S["recs"]
IV, PP = get_interventions(st.session_state.version)

# ---------------------------------------------------------------- sidebar
if "goto" in st.session_state:
    st.session_state.nav = st.session_state.pop("goto")
# Deep links for sharing a view, e.g. ?page=1&student=1234 (first load only)
if "nav" not in st.session_state and "page" in st.query_params:
    st.session_state.nav = PAGES[int(st.query_params["page"])]
    if "student" in st.query_params:
        st.session_state.student = int(st.query_params["student"])

with st.sidebar:
    st.markdown("## 🌱 Pehchaan")
    st.caption("Early identification of learning difficulties · prototype")
    page = st.radio("View", PAGES, key="nav", label_visibility="collapsed")
    st.divider()
    schools = DF[["school_id", "school"]].drop_duplicates().sort_values("school_id")
    school = st.selectbox("School", schools.school, key="school")
    c1, c2 = st.columns(2)
    grade = c1.selectbox("Class", [3, 4, 5], key="grade")
    section = c2.selectbox("Section", ["A", "B"], key="section")
    st.divider()
    st.caption(f"Academic year **{CURRENT_YEAR}** · Database: **{db.backend_name()}**")
    st.warning("Demo data is **synthetic**. No real student records.", icon="🧪")

CLASS = DF[(DF.school == school) & (DF.grade == grade) & (DF.section == section)].copy()
CLASS = CLASS.sort_values(["tier", "sld_p", "gap_p"], ascending=False)


def status_badge(status: str) -> str:
    return f'<span class="badge" style="background:{STATUS_COLORS[status]}">{status}</span>'


def top_reason(sid: int) -> str:
    r = DF.loc[sid]
    key = "sld" if r.sld_p >= SLD_THRESHOLD else "gap"
    rs = reasons(r, S[f"{key}_shap"].loc[sid].values, top_k=1)
    return rs[0]["text"] if rs and rs[0]["raises"] else "—"


# ================================================================ Class overview
def page_class():
    st.title(f"Class {grade}{section}")
    st.caption(f"{school} · {len(CLASS)} students")
    counts = CLASS.status.value_counts()
    cols = st.columns(5)
    cols[0].metric("Students", len(CLASS))
    for col, s in zip(cols[1:], STATUS_COLORS):
        col.metric(s, int(counts.get(s, 0)))

    t1, t2, t3 = st.tabs(["📋 Students", "🎯 The two signals", "👥 Level groups (TaRL)"])
    with t1:
        st.markdown("Sorted by who needs attention first. **Click a row** to open the student profile.")
        view = CLASS.assign(reason=[top_reason(s) for s in CLASS.index], gap_p=CLASS.gap_p * 100,
                            sld_p=CLASS.sld_p * 100)
        table = view[["name", "status", "gap_p", "sld_p", "reading_level", "maths_level", "attendance_pct", "reason"]]
        event = st.dataframe(
            table, hide_index=True, width="stretch", height=560,
            on_select="rerun", selection_mode="single-row",
            column_config={
                "name": "Student", "status": "Status",
                "gap_p": st.column_config.ProgressColumn("Learning-gap risk", format="%.0f%%", min_value=0, max_value=100),
                "sld_p": st.column_config.ProgressColumn("SLD-pattern signal", format="%.0f%%", min_value=0, max_value=100),
                "reading_level": "Reading level", "maths_level": "Maths level",
                "attendance_pct": st.column_config.NumberColumn("Attendance", format="%.0f%%"),
                "reason": st.column_config.TextColumn("Main reason", width="large"),
            })
        if event.selection.rows:
            st.session_state.student = int(table.index[event.selection.rows[0]])
            st.session_state.goto = PAGES[1]
            st.rerun()
    with t2:
        st.markdown(
            "Most struggling children are **behind** (high learning-gap risk) and catch up with good remedial "
            "teaching. A few show an **uneven profile and slow response to support**: the SLD pattern. "
            "Looking at the two signals separately is the core idea of Pehchaan.")
        whole = st.toggle("Show the whole school", value=True)
        pts = DF[DF.school == school] if whole else CLASS
        fig = px.scatter(pts.assign(size=8), x="gap_p", y="sld_p", color="status", hover_name="name",
                         size="size", size_max=10, color_discrete_map=STATUS_COLORS,
                         hover_data={"gap_p": ":.0%", "sld_p": ":.0%", "size": False, "status": False,
                                     "grade": True, "section": True}, opacity=0.75,
                         labels={"gap_p": "Learning-gap risk", "sld_p": "SLD-pattern signal"})
        fig.add_hline(y=SLD_THRESHOLD, line_dash="dot", line_color="grey")
        fig.add_vline(x=GAP_THRESHOLD, line_dash="dot", line_color="grey")
        fig.add_annotation(x=0.98, y=0.14, text="Behind: likely to catch up with Tier 2", showarrow=False,
                           xanchor="right", font=dict(color="grey"))
        fig.add_annotation(x=0.98, y=0.98, text="SLD pattern: watch response, consider referral",
                           showarrow=False, xanchor="right", font=dict(color="grey"))
        fig.update_layout(height=520, xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                          xaxis_range=[-0.03, 1.03], yaxis_range=[-0.03, 1.03])
        st.plotly_chart(fig, width="stretch")
    with t3:
        st.markdown("Teaching at the Right Level: group children by **level**, not by class, for the daily "
                    "support period. Re-group after every probe.")
        for subject, col in (("Reading", "reading_level"), ("Maths", "maths_level")):
            st.subheader(subject)
            levels = sorted(CLASS[col].unique())
            cs = st.columns(len(levels))
            for c, lvl in zip(cs, levels):
                names = CLASS[CLASS[col] == lvl].name.tolist()
                with c.container(border=True):
                    st.markdown(f"**{lvl}** · {len(names)}")
                    st.markdown("\n".join(f"- {n}" for n in names))


# ================================================================ Student profile
def reason_list(items):
    if not items:
        st.markdown('<span class="small">No strong signals either way.</span>', unsafe_allow_html=True)
    for x in items:
        cls, arrow = ("reason-up", "▲ raises") if x["raises"] else ("reason-down", "▼ lowers")
        st.markdown(f'<span class="{cls}">{arrow}</span> {x["text"]}', unsafe_allow_html=True)


def shap_bar(sid, key):
    vals = S[f"{key}_shap"].loc[sid]
    top = vals.reindex(vals.abs().sort_values(ascending=False).index[:8])[::-1]
    fig = go.Figure(go.Bar(x=top.values, y=[FEATURE_LABELS[f] for f in top.index], orientation="h",
                           marker_color=["#c62828" if v > 0 else "#2e7d32" for v in top.values]))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="SHAP value (log-odds push)")
    return fig


def page_student():
    options = CLASS.index.tolist()
    default = st.session_state.get("student")
    idx = options.index(default) if default in options else 0
    sid = st.selectbox("Student", options, index=idx, format_func=lambda s: f"{DF.loc[s, 'name']} · {DF.loc[s, 'status']}")
    st.session_state.student = sid
    r, rec = DF.loc[sid], RECS[sid]

    st.markdown(f"# {r['name']} &nbsp; {status_badge(r.status)}", unsafe_allow_html=True)
    st.caption(f"Class {r.grade}{r.section} · {r.school} · {r.medium} medium")
    c = st.columns(4)
    c[0].metric("Learning-gap risk", f"{r.gap_p:.0%}", help="Chance of missing the end-of-year foundational benchmark if nothing changes")
    c[1].metric("SLD-pattern signal", f"{r.sld_p:.0%}", help="How closely the profile matches children who later had an SLD confirmed. A screening signal, not a diagnosis.")
    c[2].metric("Attendance", f"{r.attendance_pct:.0f}%")
    c[3].metric("Support level", f"Tier {rec['tier']}" if rec["tier"] else "None needed")

    st.subheader("Why is this child flagged?")
    g_reasons = reasons(r, S["gap_shap"].loc[sid].values)
    s_reasons = reasons(r, S["sld_shap"].loc[sid].values)
    a, b = st.columns(2)
    with a.container(border=True):
        st.markdown(f"**Learning-gap risk · {r.gap_p:.0%}**")
        reason_list(g_reasons)
        with st.expander("SHAP detail"):
            st.plotly_chart(shap_bar(sid, "gap"), width="stretch")
    with b.container(border=True):
        st.markdown(f"**SLD-pattern signal · {r.sld_p:.0%}**")
        reason_list(s_reasons)
        with st.expander("SHAP detail"):
            st.plotly_chart(shap_bar(sid, "sld"), width="stretch")
    st.caption("Not used by the models: gender, caste, religion, income, medium or home language.")

    st.subheader("Skill profile")
    skills = pd.DataFrame({
        "skill": ["Reading fluency", "Letter sounds", "Spelling", "Reading comprehension", "Listening comprehension",
                  "Number sense", "Calculation fluency", "Maths reasoning (oral)"],
        "value": [min(r.orf_ratio, 1.3), r.letter_sound_pct / 100, r.spelling_pct / 100,
                  r.reading_comp / 10, r.listening_comp / 10, r.number_sense / 10, min(r.calc_ratio, 1.3),
                  r.math_reasoning_oral / 10],
        "area": ["Reading"] * 5 + ["Maths"] * 3})
    skills["value"] = skills.value.clip(0, 1.3)
    fig = px.bar(skills, x="value", y="skill", color="area", orientation="h",
                 color_discrete_map={"Reading": "#1565c0", "Maths": "#6a1b9a"},
                 labels={"value": "Share of grade expectation", "skill": ""})
    fig.add_vline(x=1.0, line_dash="dot", line_color="grey", annotation_text="Grade level")
    fig.update_layout(height=320, xaxis_tickformat=".0%", margin=dict(t=20, b=10), legend_title=None)
    fig.update_yaxes(categoryorder="array", categoryarray=skills.skill[::-1])
    st.plotly_chart(fig, width="stretch")

    st.subheader("Recommended support")
    for n in rec["notes"]:
        st.info(n)
    if not rec["cards"]:
        st.success("No extra support needed right now.")
    my_iv = IV[IV.student_id == sid]
    active_areas = set(my_iv[my_iv.status == "active"].area)
    for card in rec["cards"]:
        with st.container(border=True):
            h1, h2 = st.columns([4, 1.4])
            h1.markdown(f"**{card['title']}**")
            h2.markdown(TIER_BADGE[card["tier"]])
            st.markdown(f"*Why:* {card['why']}")
            st.markdown(card["what_to_do"])
            st.caption(f"⏱ {card['duration']} · 🧰 {card['materials']} · 📚 {card['evidence']}")
            if card["tier"] == 2 and card["area"] in ("reading", "maths") and card["area"] not in active_areas:
                if st.button("Start 6-week cycle", key=f"start_{card['code']}"):
                    base = float(r.orf_wcpm if card["area"] == "reading" else r.number_sense * 2)
                    db.start_intervention(sid, card["code"], card["area"], date.today(), base,
                                          round(base * (1 + 0.08 * INTERVENTION_WEEKS), 1), INTERVENTION_WEEKS)
                    bump()
                    st.rerun()

    st.subheader("Support cycles & progress")
    if my_iv.empty:
        st.caption("No support cycle started yet.")
    for _, iv in my_iv.iterrows():
        pts = PP[PP.intervention_id == iv.intervention_id].sort_values("week")
        with st.container(border=True):
            unit = "words correct / min" if iv.area == "reading" else "items correct / 20"
            st.markdown(f"**{iv.title}** · started {iv.start_date} · status: `{iv.status}`")
            weeks = np.arange(0, iv.weeks_planned + 1)
            goal = iv.baseline_score + (iv.goal_score - iv.baseline_score) * weeks / iv.weeks_planned
            fig = go.Figure()
            fig.add_scatter(x=weeks, y=goal, name="Goal line", line=dict(dash="dot", color="grey"))
            fig.add_scatter(x=np.r_[0, pts.week], y=np.r_[iv.baseline_score, pts.score], name="Weekly probe",
                            mode="lines+markers", line=dict(color="#1565c0", width=3))
            fig.update_layout(height=260, margin=dict(t=10, b=10), xaxis_title="Week", yaxis_title=unit,
                              legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, width="stretch")
            if len(pts) >= 3:
                slope = float(np.polyfit(np.r_[0, pts.week], np.r_[1, pts.score / iv.baseline_score], 1)[0])
                msg = f"Growth so far: **{slope:.1%} per week** (goal ≈ 8%)."
                (st.success if slope >= 0.05 else st.warning if slope >= NON_RESPONSE else st.error)(msg)
            if iv.status == "active":
                with st.form(f"probe_{iv.intervention_id}"):
                    fc = st.columns([1, 1, 1])
                    next_week = int(pts.week.max() + 1) if len(pts) else 1
                    wk = fc[0].number_input("Week", 1, int(iv.weeks_planned), min(next_week, int(iv.weeks_planned)))
                    sc = fc[1].number_input(f"Score ({unit})", 0.0, 200.0, float(pts.score.iloc[-1]) if len(pts) else float(iv.baseline_score))
                    if fc[2].form_submit_button("Save weekly probe"):
                        db.add_progress_probe(int(iv.intervention_id), int(wk), float(sc))
                        bump()
                        st.rerun()
                b1, b2, _ = st.columns([1, 1, 3])
                if b1.button("✅ Responded, step down", key=f"down_{iv.intervention_id}"):
                    db.set_intervention_status(int(iv.intervention_id), "stepped_down"); bump(); st.rerun()
                if b2.button("⬆️ Escalate", key=f"up_{iv.intervention_id}"):
                    db.set_intervention_status(int(iv.intervention_id), "escalated"); bump(); st.rerun()

    with st.expander(f"Teacher observation log ({int(r.obs_weeks)} weekly checks)"):
        obs = db.read("SELECT week, item, observed FROM observations WHERE student_id = :s", s=int(sid))
        grid = obs.pivot(index="item", columns="week", values="observed").reindex(OBS_ITEMS)
        grid.index = [OBS_LABELS[i] for i in grid.index]
        st.dataframe(grid.map(lambda v: "✓" if v else ""), width="stretch")

    st.subheader("Share")
    t1, t2 = st.tabs(["👪 Parent summary", "📄 Referral pack"])
    with t1:
        text = parent_summary(r, rec)
        st.markdown(text)
        st.download_button("Download", text, file_name=f"parent_summary_{sid}.md")
    with t2:
        pack = referral_pack(r, rec, s_reasons + g_reasons, my_iv, PP)
        st.markdown(pack)
        st.download_button("Download referral pack", pack, file_name=f"referral_pack_{sid}.md")


# ================================================================ Support tracker
def cycle_table(iv, pp):
    rows = []
    for _, x in iv.iterrows():
        pts = pp[pp.intervention_id == x.intervention_id].sort_values("week")
        slope = (float(np.polyfit(np.r_[0, pts.week], np.r_[1, pts.score / x.baseline_score], 1)[0])
                 if len(pts) >= 3 else np.nan)
        s = DF.loc[x.student_id]
        if np.isnan(slope):
            action = "Keep going; too early to judge"
        elif slope >= 0.05:
            action = "Responding: continue, step down after cycle"
        elif s.sld_p >= SLD_THRESHOLD:
            action = "Not responding + SLD pattern: prepare referral"
        else:
            action = "Slow: check attendance, adjust group"
        rows.append(dict(student=s["name"], cls=f"{s.grade}{s.section}", support=x.title, area=x.area,
                         weeks=len(pts), growth=slope, sld=s.sld_p, status=x.status, action=action,
                         student_id=x.student_id))
    return pd.DataFrame(rows)


def page_tracker():
    st.title("Support tracker")
    scope = st.radio("Scope", [f"Class {grade}{section}", f"Whole school ({school})"], horizontal=True)
    ids = CLASS.index if scope.startswith("Class") else DF[DF.school == school].index
    iv = IV[IV.student_id.isin(ids) & (IV.status == "active")]
    if iv.empty:
        st.info("No active support cycles."); return
    t = cycle_table(iv, PP)
    c = st.columns(4)
    c[0].metric("Active cycles", len(t))
    c[1].metric("Responding (≥5%/wk)", int((t.growth >= 0.05).sum()))
    c[2].metric("Slow", int(((t.growth < 0.05) & (t.growth >= NON_RESPONSE)).sum()))
    c[3].metric("Not responding (<4%/wk)", int((t.growth < NON_RESPONSE).sum()))

    fig = px.strip(t.assign(sld_flag=np.where(t.sld >= SLD_THRESHOLD, "SLD pattern", "No SLD pattern")),
                   x="growth", y="area", color="sld_flag", hover_name="student",
                   color_discrete_map={"SLD pattern": "#c62828", "No SLD pattern": "#1565c0"},
                   labels={"growth": "Weekly growth during support", "area": ""})
    fig.add_vline(x=NON_RESPONSE, line_dash="dot", annotation_text="non-response line")
    fig.update_layout(height=280, xaxis_tickformat=".0%", legend_title=None, margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")
    st.caption("Children who are simply behind usually grow fast once support starts. Flat growth plus an "
               "SLD pattern is the strongest reason to refer (Response to Intervention).")
    shown = t.drop(columns=["student_id"]).sort_values("growth").assign(growth=t.growth * 100, sld=t.sld * 100)
    st.dataframe(shown, hide_index=True, width="stretch",
                 column_config={"growth": st.column_config.NumberColumn("Growth / week", format="%.1f%%"),
                                "sld": st.column_config.ProgressColumn("SLD signal", format="%.0f%%", min_value=0, max_value=100),
                                "cls": "Class", "action": st.column_config.TextColumn("Suggested action", width="large")})


# ================================================================ Block dashboard
def page_block():
    st.title("Block / cluster dashboard")
    st.caption("Aggregated view for a cluster resource person or block education officer. No names, only counts.")
    d = DF.copy()
    d["needs_support"] = d.tier >= 2
    d["refer"] = d.tier == 3
    d["sld_flag"] = d.sld_p >= SLD_THRESHOLD
    summary = d.groupby("school").agg(students=("name", "size"), on_track=("tier", lambda t: (t == 0).mean()),
                                      needs_support=("needs_support", "mean"), sld_flags=("sld_flag", "sum"),
                                      referrals=("refer", "sum"), attendance=("attendance_pct", "mean")).reset_index()
    t = cycle_table(IV[IV.status == "active"], PP)
    t["school"] = t.student_id.map(DF.school)
    resp = t.groupby("school").growth.apply(lambda g: (g >= 0.05).mean()).rename("responding").reset_index()
    summary = summary.merge(resp, on="school", how="left").sort_values("sld_flags", ascending=False)

    c = st.columns(4)
    c[0].metric("Students screened", len(d))
    c[1].metric("Need Tier 2+ support", f"{d.needs_support.mean():.0%}")
    c[2].metric("SLD-pattern flags", int(d.sld_flag.sum()))
    c[3].metric("Ready for referral", int(d.refer.sum()))

    a, b = st.columns([1.2, 1])
    with a:
        heat = d.pivot_table(index="school", columns="grade", values="needs_support", aggfunc="mean")
        fig = px.imshow(heat, text_auto=".0%", color_continuous_scale="Oranges", aspect="auto",
                        labels=dict(x="Class", y="", color="Need support"))
        fig.update_layout(height=320, margin=dict(t=30), title="Share of children needing Tier 2+ support")
        st.plotly_chart(fig, width="stretch")
    with b:
        fig = px.bar(summary.sort_values("sld_flags"), x="sld_flags", y="school", orientation="h",
                     labels={"sld_flags": "Children with SLD-pattern signal", "school": ""},
                     title="Where to send the special educator first", color_discrete_sequence=["#c62828"])
        fig.update_layout(height=320, margin=dict(t=40))
        st.plotly_chart(fig, width="stretch")
    for col in ("on_track", "needs_support", "responding"):
        summary[col] = summary[col] * 100
    st.dataframe(summary, hide_index=True, width="stretch", column_config={
        "on_track": st.column_config.NumberColumn("On track", format="%.0f%%"),
        "needs_support": st.column_config.NumberColumn("Need Tier 2+", format="%.0f%%"),
        "responding": st.column_config.NumberColumn("Support cycles responding", format="%.0f%%"),
        "attendance": st.column_config.NumberColumn("Avg attendance", format="%.0f%%"),
        "sld_flags": "SLD-pattern flags", "referrals": "Ready for referral"})


# ================================================================ Model & fairness
def page_model():
    b = get_bundle()
    st.title("Model & fairness")
    st.info(f"Trained on **{b['n_students']:,} simulated students** from past cohorts ({b['trained_on']}), "
            "evaluated on a held-out 25%. Numbers show the pipeline works; they are not real-world accuracy.",
            icon="🧪")
    for key in ("gap", "sld"):
        rep = b["report"][key]
        st.header(rep["title"])
        st.caption(f"Prevalence in training data: {rep['prevalence']:.1%} · decision threshold {rep['threshold']:.0%} · "
                   f"train {rep['n_train']} / calibration {rep['n_calibration']} / test {rep['n_test']}")
        g, l = rep["gradient_boosting"], rep["logistic_regression"]
        c = st.columns(5)
        c[0].metric("ROC-AUC", f"{g['auc']:.3f}", f"{g['auc'] - l['auc']:+.3f} vs logistic reg.")
        c[1].metric("Recall", f"{g['recall']:.0%}", help="Of children who truly had the outcome, how many we flagged")
        c[2].metric("Precision", f"{g['precision']:.0%}", help="Of children we flagged, how many truly had the outcome")
        c[3].metric("Flag rate", f"{g['flag_rate']:.0%}")
        c[4].metric("Brier score", f"{g['brier']:.3f}", help="Calibration + accuracy, lower is better")

        a, bcol, ccol = st.columns(3)
        with a:
            cal = rep["calibration"]
            fig = go.Figure()
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="grey"), name="Perfect")
            fig.add_scatter(x=cal.predicted, y=cal.actual, mode="lines+markers", name="Model",
                            text=cal.n, hovertemplate="predicted %{x:.0%}<br>actual %{y:.0%}<br>n=%{text}")
            fig.update_layout(title="Calibration", height=300, xaxis_title="Predicted", yaxis_title="Actual",
                              xaxis_tickformat=".0%", yaxis_tickformat=".0%", margin=dict(t=40), showlegend=False)
            st.plotly_chart(fig, width="stretch")
        with bcol:
            z = [[g["tn"], g["fp"]], [g["fn"], g["tp"]]]
            fig = px.imshow(z, text_auto=True, x=["Not flagged", "Flagged"], y=["No", "Yes"],
                            color_continuous_scale="Blues", labels=dict(x="Model", y="Actual"))
            fig.update_layout(title="Confusion matrix (test)", height=300, margin=dict(t=40), coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")
        with ccol:
            imp = S[f"{key}_shap"].abs().mean().sort_values(ascending=False).head(8)[::-1]
            fig = px.bar(x=imp.values, y=[FEATURE_LABELS[f] for f in imp.index], orientation="h",
                         labels={"x": "mean |SHAP|", "y": ""})
            fig.update_layout(title="What the model relies on", height=300, margin=dict(t=40))
            st.plotly_chart(fig, width="stretch")

        st.markdown("**Fairness audit** (gender and medium are *not* model inputs; used only here)")
        st.dataframe(rep["fairness"], hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="%.2f") for c in ("base_rate", "flag_rate", "recall", "false_positive_rate")})
        st.divider()


# ================================================================ Data
def page_data():
    st.title("Data")
    st.subheader("Where does the data come from?")
    st.markdown("""
There is **no public Indian dataset** that links school records to confirmed learning-disability outcomes,
so this prototype runs on a **synthetic dataset** built from published statistics:

| Assumption in the simulation | Source |
|---|---|
| ~8% of children have a Specific Learning Disability; dyslexia more common in boys | Scaria et al. 2023 meta-analysis; IOPN 2022 meta-analysis |
| A large share of children are behind grade level, more with home-language mismatch | ASER 2024 (23.4% of Std III govt school children read Std II text) |
| Children who are behind respond quickly to level-based remedial teaching | J-PAL / Pratham TaRL evaluations |
| Children with dyslexia understand stories read aloud much better than when reading alone | Dyslexia definition (decoding vs comprehension) |
| SLD children respond slowly to Tier 2 support | Response to Intervention (RTI/MTSS) literature |
| Teacher observation items | NCERT PRASHAST SLD checklist items (reworded) |

**Real datasets we can validate parts of the pipeline on** (next milestone): UCI *Student Performance*
(grades + absences, Portugal), OULAD (Open University), and the *Dytective* dyslexia dataset released
with Rello et al. 2020 (PLOS ONE, 3,600+ participants, Spanish). **For a real pilot**: a partner school's
own marks, attendance and PRASHAST results, with consent, labelled over time by actual certifications.
""")
    counts = {t: int(db.read(f"SELECT COUNT(*) AS n FROM {t}")["n"][0]) for t in db.LOAD_ORDER}
    st.subheader(f"Tables in {db.backend_name()}")
    st.dataframe(pd.DataFrame(counts.items(), columns=["table", "rows"]), hide_index=True)

    st.subheader("Import term probes (CSV)")
    st.markdown("Teachers can upload term probe scores from a spreadsheet. Download the template for this class, "
                "fill it, and upload it. Rows are matched on `student_id`.")
    cols = ["student_id", "attendance_pct", "lang_marks_pct", "math_marks_pct", "orf_wcpm", "letter_sound_pct",
            "listening_comp", "reading_comp", "spelling_pct", "number_sense", "calc_facts_per_min", "math_reasoning_oral"]
    template = CLASS.reset_index()[["student_id", "name"] + cols[1:]]
    st.download_button("Download template for this class", template.to_csv(index=False),
                       file_name=f"probes_{grade}{section}.csv")
    up = st.file_uploader("Upload filled CSV", type="csv")
    if up is not None:
        new = pd.read_csv(up)
        missing = [c for c in cols if c not in new.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            bounds = {"attendance_pct": 100, "lang_marks_pct": 100, "math_marks_pct": 100, "letter_sound_pct": 100,
                      "spelling_pct": 100, "listening_comp": 10, "reading_comp": 10, "number_sense": 10,
                      "math_reasoning_oral": 10}
            bad = [c for c, hi in bounds.items() if ((new[c] < 0) | (new[c] > hi)).any()]
            unknown = set(new.student_id) - set(DF.index)
            if bad or unknown:
                st.error(f"Out-of-range columns: {bad}. Unknown student ids: {sorted(unknown)[:10]}")
            else:
                st.dataframe(new.head(), hide_index=True)
                if st.button("Save to database", type="primary"):
                    for _, row in new.iterrows():
                        sets = ", ".join(f"{c} = :{c}" for c in cols[1:])
                        db.execute(f"UPDATE skill_probes SET {sets} WHERE student_id = :student_id AND term = 'T1'",
                                   **{c: float(row[c]) if c != "student_id" else int(row[c]) for c in cols})
                    bump()
                    st.success(f"Updated {len(new)} students. Scores recalculated.")

    st.subheader("Reset")
    if st.button("Regenerate synthetic data and retrain models"):
        run_setup()
        bump()
        st.rerun()


{PAGES[0]: page_class, PAGES[1]: page_student, PAGES[2]: page_tracker,
 PAGES[3]: page_block, PAGES[4]: page_model, PAGES[5]: page_data}[page]()
