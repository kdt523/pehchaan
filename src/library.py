"""Curated intervention library.

Kept as plain data so a counsellor or special educator can edit it without
touching model code. The ML models decide WHO needs support; these cards and
the rules in recommend.py decide WHAT support.
"""

LIBRARY = [
    # ---- Reading ----
    dict(code="R1-READALOUD", area="reading", tier=1,
         title="Read instructions aloud + reading buddy",
         what_to_do="Read all written instructions aloud, including in tests. Pair the child with a "
                    "patient reading buddy for 10 minutes a day. Don't ask them to read aloud in front "
                    "of the class unless they volunteer.",
         duration="Ongoing", materials="None",
         evidence="Standard classroom accommodation for reading difficulty (RPwD Act 2016, CBSE/State Board SLD concessions)."),
    dict(code="R2-PHONICS", area="reading", tier=2,
         title="Structured phonics group (multisensory)",
         what_to_do="Small group of 4-6 children, 20 minutes a day. Letter-sound cards, tracing letters "
                    "in sand or on the desk while saying the sound, then blending 2-3 sounds into words. "
                    "Same routine every day, move on only when 80% accurate.",
         duration="6 weeks, then review", materials="Letter-sound cards, slate, sand tray (optional)",
         evidence="Systematic, explicit phonics (structured literacy) is the best-supported approach for decoding difficulty."),
    dict(code="R2-TARL", area="reading", tier=2,
         title="Reading level group (Teaching at the Right Level)",
         what_to_do="Regroup children by reading level (letter / word / paragraph / story) instead of grade "
                    "for 45 minutes a day. Use level-appropriate activities, re-test every 10 days and move "
                    "children up as they progress.",
         duration="30-50 days", materials="Pratham TaRL reading charts, story cards",
         evidence="Randomised evaluations in India (J-PAL/Pratham): large gains, e.g. 0.70 SD in UP learning camps."),
    dict(code="R3-REFER", area="reading", tier=3,
         title="Refer for reading (dyslexia) assessment",
         what_to_do="Share the referral pack with the special educator / counsellor and the family. Continue "
                    "the phonics group and Tier 1 accommodations while the assessment is pending. Start early "
                    "so certification is ready well before board exams.",
         duration="Until assessed", materials="Referral pack (download from student page)",
         evidence="RTI/MTSS: persistent non-response to good Tier 2 support is a key indicator for SLD evaluation."),
    # ---- Language ----
    dict(code="L1-ORAL", area="language", tier=1,
         title="Oral language & vocabulary building",
         what_to_do="Daily 10-minute read-aloud with discussion. Introduce 3 new words a day with pictures. "
                    "Allow the child to answer in their home language first, then bridge to the school language.",
         duration="Ongoing", materials="Picture story books",
         evidence="Oral language underpins reading comprehension; home-language bridging helps children taught in a second language."),
    # ---- Maths ----
    dict(code="M1-SUPPORTS", area="maths", tier=1,
         title="Number line, table card and read-aloud word problems",
         what_to_do="Allow a number line and multiplication table card during practice. Read word problems "
                    "aloud so reading isn't the barrier to maths.",
         duration="Ongoing", materials="Printed number line, table card",
         evidence="Common accommodation for maths difficulty; separates reasoning from calculation."),
    dict(code="M2-TARL", area="maths", tier=2,
         title="Number sense group with bundles and sticks (TaRL maths)",
         what_to_do="Small group, 30 minutes a day. Build place value with sticks and bundles, count on a "
                    "number line, then move to written numbers. Group by level, not grade.",
         duration="6 weeks, then review", materials="Ice-cream sticks, rubber bands, number line",
         evidence="TaRL maths activities evaluated alongside reading in J-PAL/Pratham RCTs."),
    dict(code="M2-FACTS", area="maths", tier=2,
         title="Maths fact fluency games",
         what_to_do="10 minutes a day of addition/subtraction fact games (dice, cards) in pairs. Focus on "
                    "strategies (doubles, making 10) rather than rote memory.",
         duration="6 weeks, then review", materials="Dice, playing cards",
         evidence="Strategy-based fact practice improves fluency for children with calculation difficulty."),
    dict(code="M3-REFER", area="maths", tier=3,
         title="Refer for maths (dyscalculia) assessment",
         what_to_do="Share the referral pack with the special educator / counsellor and the family. Keep "
                    "Tier 1 supports and the maths group going while the assessment is pending.",
         duration="Until assessed", materials="Referral pack (download from student page)",
         evidence="RTI/MTSS: persistent non-response to good Tier 2 support is a key indicator for SLD evaluation."),
    # ---- Attendance / attention ----
    dict(code="A1-ATTEND", area="attendance", tier=1,
         title="Attendance follow-up with family",
         what_to_do="Call or visit the family this week. Find the reason for absence (health, work, "
                    "migration, fear of class). Give a short catch-up sheet for missed work.",
         duration="2 weeks, then review", materials="Catch-up sheet",
         evidence="Chronic absence (missing 10%+ of days) is one of the strongest early warning indicators (Balfanz ABC)."),
    dict(code="F1-ATTENTION", area="attention", tier=1,
         title="Seating, chunked tasks and movement breaks",
         what_to_do="Seat the child near the teacher, away from the door. Break tasks into 5-minute chunks "
                    "with a tick after each. Give a small job (distributing books) as a movement break.",
         duration="Ongoing", materials="None",
         evidence="Standard classroom strategies for attention difficulty."),
    dict(code="F3-COUNSEL", area="attention", tier=3,
         title="Counsellor / special educator consult",
         what_to_do="If attention concerns continue alongside learning difficulty, ask the counsellor or "
                    "special educator to observe the child in class.",
         duration="1 observation, then plan", materials="Observation log (from student page)",
         evidence="Attention difficulties often co-occur with SLD and need separate assessment."),
]

LIBRARY_BY_CODE = {c["code"]: c for c in LIBRARY}
