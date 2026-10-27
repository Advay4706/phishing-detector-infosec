"""One-off: bring logs written before Step 7 to the current format.

- removes the user_prompt column (it contained the full email body; logs must not)
- adds email_kind / llm_verdict_raw / validator_intervened / validator_note with their
  no-defence values, so old and new logs read the same way

    python -m src.migrate_logs results/logs/step5_pilot.csv results/logs/pilot_3seeds.csv
"""
import sys

import pandas as pd

from src.harness import LOG_FIELDS


def migrate(path):
    df = pd.read_csv(path)
    before = list(df.columns)
    df = df.drop(columns=[c for c in ("user_prompt",) if c in df])
    df["email_kind"] = df.get("email_kind", "phishing")
    if "llm_verdict_raw" not in df:
        df["llm_verdict_raw"] = df["verdict"]
    if "validator_intervened" not in df:
        df["validator_intervened"] = False
    if "validator_note" not in df:
        df["validator_note"] = ""
    df = df[LOG_FIELDS]
    df.to_csv(path, index=False)
    print(f"{path}: {len(df)} rows; removed {sorted(set(before) - set(df.columns))}; "
          f"added {sorted(set(df.columns) - set(before))}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        migrate(p)
