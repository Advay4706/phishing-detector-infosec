"""Quick look at any harness log. Uses the one metric definition in src/metrics.py.

    python -m src.summarise_run results/logs/pilot_3seeds.csv
"""
import sys

from src.metrics import load_runs, pct, table

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/logs/pilot_3seeds.csv"
    df = load_runs(path)
    print(f"{path}: {len(df)} runs, status {df.status.value_counts().to_dict()}")
    for d, r in table(df, "defence").iterrows():
        print(f"  {d:11s} ASR {pct(r.ASR)} [{pct(r.ASR_low)}, {pct(r.ASR_high)}]  "
              f"INV {pct(r.INV)}  FLIP {pct(r.FLIP)}  LAR {pct(r.LAR)}")
    print("\nASR % by payload x location:")
    t = table(df, ["defence", "payload_id", "location"]).ASR.unstack("location") * 100
    print(t.round(1).to_string())
