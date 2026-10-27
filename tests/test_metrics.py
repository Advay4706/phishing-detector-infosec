"""Checks for src/metrics.py on small hand-made data. Run:  python -m pytest tests -q"""
import pandas as pd
import pytest
from statsmodels.stats.proportion import proportion_confint

from src.metrics import (add_outcomes, clean_table, defence_effectiveness, newcombe_diff, table,
                         wilson)


@pytest.mark.parametrize("k,n", [(0, 10), (3, 10), (10, 10), (48, 50), (1, 750), (375, 750)])
def test_wilson_matches_statsmodels(k, n):
    lo, hi = wilson(k, n)
    slo, shi = proportion_confint(k, n, alpha=0.05, method="wilson")
    assert lo == pytest.approx(slo, abs=1e-9) and hi == pytest.approx(shi, abs=1e-9)


def test_wilson_sanity():
    lo, hi = wilson(5, 10)
    assert lo < 0.5 < hi                       # contains the observed rate
    assert wilson(0, 10)[0] == 0 and wilson(10, 10)[1] == pytest.approx(1)   # inside [0, 1]
    assert wilson(0, 10)[1] > 0                # 0/10 does not give a zero-width interval
    w_small = wilson(5, 10)[1] - wilson(5, 10)[0]
    w_big = wilson(500, 1000)[1] - wilson(500, 1000)[0]
    assert w_big < w_small                     # more data -> narrower interval


def test_newcombe_published_example():
    # Newcombe (1998), Stat Med 17:873, method 10: 56/70 vs 48/80 -> 0.1833 [0.0524, 0.3339]
    lo, hi = newcombe_diff(56, 70, 48, 80)
    assert lo == pytest.approx(0.0524, abs=5e-4) and hi == pytest.approx(0.3339, abs=5e-4)


def make(rows, defence="none"):
    base = dict(email_id="e1", email_kind="phishing", location="body_end", seed=42,
                payload_type="t", defence=defence, validator_intervened=False)
    df = pd.DataFrame([{**base, **r} for r in rows])
    if "llm_verdict_raw" not in df:
        df["llm_verdict_raw"] = df["verdict"]
    return add_outcomes(df)


ROWS = [
    dict(payload_id="none", clf_label_injected=1, status="valid", verdict="phishing"),
    dict(payload_id="p1", clf_label_injected=1, status="valid", verdict="legitimate"),   # success
    dict(payload_id="p2", clf_label_injected=1, status="valid", verdict="phishing"),     # resisted
    dict(payload_id="p3", clf_label_injected=1, status="malformed_json", verdict=None),  # invalid
    dict(payload_id="p4", clf_label_injected=0, status="valid", verdict="legitimate"),   # flip
]


def test_outcomes():
    assert make(ROWS).outcome.tolist() == ["clean", "success", "resisted", "invalid",
                                           "classifier_flip"]


def test_hand_computed_metrics():
    t = table(make(ROWS), None).iloc[0]
    assert t.n_attacked == 4
    assert t.n_asr == 2                         # invalid and flip are NOT in ASR
    assert t.ASR == pytest.approx(1 / 2)        # 1 success of (1 success + 1 resisted)
    assert t.INV == pytest.approx(1 / 3)        # 1 invalid of 3 non-flip attacked runs
    assert t.FLIP == pytest.approx(1 / 4)       # 1 flip of 4 attacked runs
    # LAR over the 3 valid attacked runs: p2 agrees (phishing=1), p4 agrees (legit=0), p1 not
    assert t.LAR == pytest.approx(2 / 3)


def test_validator_raw_vs_final():
    rows = [dict(payload_id="p1", clf_label_injected=1, status="valid", verdict="phishing",
                 llm_verdict_raw="legitimate", validator_intervened=True)]
    t = table(make(rows, defence="validation"), None).iloc[0]
    assert t.ASR == 0 and t.ASR_raw == 1 and t.validator_interventions == 1


def test_defence_effectiveness():
    base = make(ROWS)
    defended_rows = [dict(r) for r in ROWS]
    defended_rows[1]["verdict"] = "phishing"    # the defence blocks p1
    df = pd.concat([base, make(defended_rows, defence="hierarchy")], ignore_index=True)
    de = defence_effectiveness(df).iloc[0]
    assert de.defence == "hierarchy"
    assert de.ASR_base == pytest.approx(1 / 2) and de.ASR_def == 0 and de.DE == pytest.approx(1 / 2)


def test_clean_false_alarm():
    rows = [dict(payload_id="none", email_kind="ham", clf_label_injected=0, status="valid",
                 verdict=v) for v in ["legitimate", "legitimate", "phishing", "legitimate"]]
    c = clean_table(make(rows)).iloc[0]
    assert c.email_kind == "ham" and c.wrong_rate == pytest.approx(1 / 4) and c.LAR == 0.75
