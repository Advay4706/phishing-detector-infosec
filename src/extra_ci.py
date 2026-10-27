"""95% Wilson intervals for rates that earlier steps reported without one, computed from the
exact counts in the saved files. Writes results/metrics/extra_ci.csv.

    python -m src.extra_ci
"""
import json

import pandas as pd

from src import config
from src.metrics import METRICS_DIR, wilson

rows = []


def add(name, k, n, source):
    lo, hi = wilson(k, n)
    rows.append({"metric": name, "k": int(k), "n": int(n), "rate": k / n, "low": lo, "high": hi,
                 "source": source})


m = json.loads((config.RESULTS_DIR / "stage1_metrics.json").read_text(encoding="utf-8"))
src = "results/stage1_metrics.json"
for s in m["test_per_set"]:
    n = s["n"]
    k = s["flagged_malicious"] if s["true_kind"] != "ham" else n - s["flagged_malicious"]
    add(f"stage1 correct on {s['eval_set']} ({s['source']} {s['true_kind']})", k, n, src)
(tn, fp), (fn, tp) = m["test_main_confusion_matrix"]
add("stage1 recall (Nazario phishing vs CEAS ham)", tp, tp + fn, src)
add("stage1 precision (Nazario phishing vs CEAS ham)", tp, tp + fp, src)
add("stage1 false-alarm rate on CEAS ham", fp, fp + tn, src)

s4 = pd.read_csv(config.LOGS_DIR / "step4_clean_test.csv")
r1 = s4[s4.run == 1]
src4 = "results/logs/step4_clean_test.csv"
add("step4 valid JSON (run 1)", (r1.status == "valid").sum(), len(r1), src4)
agree = (r1.verdict == r1.clf_label.map({1: "phishing", 0: "legitimate"})).sum()
add("step4 explainer agrees with classifier (run 1)", agree, len(r1), src4)
add("step4 same verdict run 1 vs run 2", (s4[s4.run == 1].verdict.values ==
                                         s4[s4.run == 2].verdict.values).sum(), len(r1), src4)

fl = pd.read_csv(config.LOGS_DIR / "step4_flipped_label_check.csv")
srcf = "results/logs/step4_flipped_label_check.csv"
for kind in ("phishing", "legitimate"):
    g = fl[fl["true"] == kind]
    add(f"step4 flipped label: kept true verdict on {kind}", (g["llm"] == g["true"]).sum(), len(g),
        srcf)

out = pd.DataFrame(rows)
out.to_csv(METRICS_DIR / "extra_ci.csv", index=False)
print(out.assign(rate=out.rate.round(4), low=out.low.round(4), high=out.high.round(4))
      .drop(columns="source").to_string(index=False))
