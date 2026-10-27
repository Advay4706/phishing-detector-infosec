"""Pick 3 demo examples from REAL saved results (no defence), by fixed rules:

  1. clean          the email with no payload: explainer agreed (phishing) in every seed
  2. attack works   the same email with DO-1 at body_end: explainer said legitimate in every seed
  3. attack fails   the same email with FV-1 at body_end: explainer said phishing in every seed

The first email (by id) meeting all three rules is used, so the demo shows one email three ways.
Each example stores what the results file recorded, so the UI can show it next to a live run.

    python -m src.pick_demo_examples results/logs/pilot_3seeds.csv
"""
import json
import sys
from pathlib import Path

from src import config
from src.metrics import load_runs

RULES = [
    ("Clean email (no attack)", None, None, "clean"),
    ("Attack that works: DO-1 at body end", "DO-1", "body_end", "success"),
    ("Attack that fails: FV-1 at body end", "FV-1", "body_end", "resisted"),
]


def main(log):
    df = load_runs(log)
    df = df[(df.defence == "none") & (df.email_kind == "phishing")]
    seeds = sorted(df.seed.unique())

    def ok(g, payload, loc, want):
        if payload is None:
            g = g[g.payload_id == "none"]
            return len(g) == len(seeds) and (g.verdict == "phishing").all()
        g = g[(g.payload_id == payload) & (g.location == loc)]
        return len(g) == len(seeds) and (g.outcome == want).all()

    chosen = None
    for email_id, g in sorted(df.groupby("email_id"), key=lambda x: x[0]):
        if all(ok(g, p, loc, want) for _, p, loc, want in RULES):
            chosen = (email_id, g)
            break
    if chosen is None:
        sys.exit("no email meets all three rules")
    email_id, g = chosen
    examples = []
    for title, p, loc, want in RULES:
        rows = g[g.payload_id == (p or "none")]
        if p:
            rows = rows[rows.location == loc]
        verdicts = ", ".join(f"seed {r.seed}: {r.verdict}" for r in rows.itertuples())
        examples.append({
            "title": title, "email_id": email_id, "payload_id": p, "location": loc,
            "defence": "none", "expected_outcome": want,
            "recorded": f"classifier P(malicious) {rows.clf_prob_injected.iloc[0]:.2f}; "
                        f"explainer verdict {verdicts}",
        })
    # 4. Defended: from the Step 7 log, an attack that worked with no defence and was blocked by
    #    the combined defence (same email, payload, location, seed). Prefer the demo email.
    def_log = config.LOGS_DIR / "defences_15.csv"
    if def_log.exists():
        d = load_runs(def_log)
        att = d[d.attacked]
        key = ["email_id", "payload_id", "location", "seed"]
        both = att[att.defence == "none"].merge(att[att.defence == "combined"], on=key,
                                               suffixes=("_none", "_comb"))
        hits = both[(both.outcome_none == "success") & (both.outcome_comb == "resisted")]
        hits = hits.assign(pref=(hits.email_id != email_id), loc_pref=(hits.location != "body_end"),
                           p_pref=(hits.payload_id != "DO-1")).sort_values(
            ["pref", "p_pref", "loc_pref", "email_id"])
        if len(hits):
            h = hits.iloc[0]
            examples.append({
                "title": f"Same kind of attack with all defences on: {h.payload_id} at "
                         f"{h.location.replace('_', ' ')}",
                "email_id": h.email_id, "payload_id": h.payload_id, "location": h.location,
                "defence": "combined", "expected_outcome": "resisted",
                "recorded": f"seed {int(h.seed)}: no defence -> {h.verdict_none}; combined -> "
                            f"{h.verdict_comb}"
                            + (" (validator override)" if h.validator_intervened_comb else "")
                            + f"; classifier P(malicious) {h.clf_prob_injected_none:.2f}",
                "source_file": def_log.relative_to(config.ROOT).as_posix(),
            })
    out = {"source_file": Path(log).as_posix(), "seeds": [int(s) for s in seeds],
           "selection_rules": [r[0] for r in RULES] + ["Defended attack (from defences_15.csv)"],
           "examples": examples}
    path = config.RESULTS_DIR / "demo_examples.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/logs/pilot_3seeds.csv")
