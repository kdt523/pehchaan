"""Rule-based intervention recommender (transparent and editable on purpose).

Follows the RTI / MTSS idea: nobody is referred for assessment straight off a
model score. A child with a high SLD-pattern signal first gets a good Tier 2
cycle; only if they don't respond do we recommend Tier 3 referral.
"""
from config import GAP_THRESHOLD, SLD_THRESHOLD
from src.library import LIBRARY_BY_CODE

NON_RESPONSE = 0.04     # growth per week below this after 4+ weeks = not responding
MIN_WEEKS = 4

STATUS = {0: "On track", 1: "Watch (Tier 1)", 2: "Needs support (Tier 2)", 3: "Refer (Tier 3)"}


def _non_responder(r, area):
    return bool(r[f"in_{area}_support"] and r[f"{area}_weeks"] >= MIN_WEEKS and r[f"{area}_response"] < NON_RESPONSE)


def recommend(r, gap_p: float, sld_p: float) -> dict:
    cards, notes = [], []
    sld_flag = sld_p >= SLD_THRESHOLD
    gap_flag = gap_p >= GAP_THRESHOLD

    def add(code, why):
        if code not in [c["code"] for c in cards]:
            cards.append(dict(**LIBRARY_BY_CODE[code], why=why))

    reading_concern = r.orf_ratio < 0.7 or r.letter_sound_pct < 60
    maths_concern = r.number_sense < 5.5 or r.calc_ratio < 0.6

    if reading_concern:
        add("R1-READALOUD", "Reading fluency is below grade level.")
        if sld_flag and _non_responder(r, "reading"):
            add("R3-REFER", f"Hasn't responded to {int(r.reading_weeks)} weeks of reading support and "
                            f"the profile matches the SLD pattern.")
            add("R2-PHONICS", "Keep the phonics group going while the assessment is pending.")
        elif r.listen_minus_read >= 2 or r.letter_sound_pct < 60:
            add("R2-PHONICS", "Decoding (letter sounds) is the weak link; understanding is fine when read aloud.")
        elif gap_flag or r.orf_ratio < 0.6:
            add("R2-TARL", "Reading is behind across the board; a level-based group should close the gap.")
        if r.listening_comp < 5:
            add("L1-ORAL", "Understanding is low even when the story is read aloud, so build oral language too.")

    if maths_concern:
        add("M1-SUPPORTS", "Maths skills are below grade level.")
        if sld_flag and _non_responder(r, "maths"):
            add("M3-REFER", f"Hasn't responded to {int(r.maths_weeks)} weeks of maths support and "
                            f"the profile matches the SLD pattern.")
            add("M2-TARL", "Keep the maths group going while the assessment is pending.")
        elif r.number_sense < 5:
            add("M2-TARL", "Number sense (place value, counting) needs rebuilding with concrete materials.")
        elif r.reason_minus_calc >= 0.3 or r.calc_ratio < 0.6:
            add("M2-FACTS", "Reasoning is fine but calculation is slow.")

    if r.attendance_pct < 80:
        add("A1-ATTEND", f"Attendance is {r.attendance_pct:.0f}%; missed lessons explain part of the gap.")
    if r.obs_attention_drift >= 0.5:
        add("F1-ATTENTION", "Teacher often notices attention drifting.")
        if sld_flag and r.obs_attention_drift >= 0.6:
            add("F3-COUNSEL", "Attention concerns alongside a learning-difficulty pattern.")

    if sld_flag and not any(c["tier"] == 3 for c in cards):
        in_support = r.in_reading_support or r.in_maths_support
        notes.append("SLD-pattern signal is high. "
                     + ("Watch the current support cycle closely; if growth stays flat, refer."
                        if in_support else "Start a Tier 2 group now and re-check after 6 weeks before any referral."))

    tier = max([c["tier"] for c in cards], default=0)
    if tier == 0 and (gap_flag or sld_flag):
        tier = 1
        notes.append("Scores are borderline. No action needed beyond re-checking at the next probe.")
    if tier == 1 and sld_flag:
        tier = 2
    return dict(tier=tier, status=STATUS[tier], cards=sorted(cards, key=lambda c: -c["tier"]), notes=notes)


def reading_level(r) -> str:
    """TaRL-style reading level for level grouping."""
    if r.letter_sound_pct < 50 or r.orf_ratio < 0.2:
        return "1 · Letter"
    if r.orf_ratio < 0.5:
        return "2 · Word"
    if r.orf_ratio < 0.8:
        return "3 · Paragraph"
    return "4 · Story"


def maths_level(r) -> str:
    if r.number_sense < 3:
        return "1 · Numbers 1-9"
    if r.number_sense < 5:
        return "2 · Numbers 10-99"
    if r.number_sense < 7.5 or r.calc_ratio < 0.8:
        return "3 · Operations"
    return "4 · Word problems"
