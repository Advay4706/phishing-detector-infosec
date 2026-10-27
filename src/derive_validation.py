"""Output validation does not change the prompt: the model is asked exactly what the baseline
asks (same prompt, temperature 0, same seed), and only its ANSWER is checked afterwards.
So instead of calling the model again, this applies the same validator (src.defences.
validate_output) to the saved baseline replies and writes them as defence="validation" rows.

Each derived row says so in validator_note and prompt_version. Verdicts were 100% repeatable
for the same prompt and seed (Steps 4 and 6), which is what makes this valid.

    python -m src.derive_validation results/logs/defences_15.csv
"""
import sys

import pandas as pd

from src.defences import validate_output
from src.harness import LOG_FIELDS


def derive(path):
    log = pd.read_csv(path)
    base = log[log.defence == "none"].drop_duplicates("run_key", keep="last")
    have = set(log.run_key[log.defence == "validation"])
    rows = []
    for r in base.itertuples(index=False):
        key = r.run_key.rsplit("|", 1)[0] + "|validation"
        if key in have:
            continue
        rec = {"status": r.status, "verdict": None if pd.isna(r.verdict) else r.verdict,
               "reason": None if pd.isna(r.reason) else r.reason}
        out = validate_output(rec, int(r.clf_label_injected), float(r.clf_prob_injected))
        row = r._asdict()
        row.update(run_key=key, defence="validation",
                   prompt_version=f"{r.prompt_version.split('+')[0]}+validation(derived)",
                   verdict=out["verdict"], reason=out["reason"],
                   llm_verdict_raw=row["verdict"],
                   validator_intervened=out["validator_intervened"],
                   validator_note=("derived from the saved baseline reply; "
                                   + (out["validator_note"] or "no change")))
        rows.append(row)
    if rows:
        pd.DataFrame(rows)[LOG_FIELDS].to_csv(path, mode="a", header=False, index=False)
    n_int = sum(r["validator_intervened"] for r in rows)
    print(f"{path}: added {len(rows)} validation rows from baseline replies "
          f"({n_int} overridden by the validator)")


if __name__ == "__main__":
    derive(sys.argv[1] if len(sys.argv) > 1 else "results/logs/defences_15.csv")
