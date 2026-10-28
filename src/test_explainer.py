"""Step 4 check: run the explainer on 10 clean test emails, twice.

5 Nazario phishing + 5 CEAS ham, all labelled correctly by Stage 1 (sampled with seed 42).
Pass if: every reply is valid JSON, and run 1 == run 2 (determinism).
Run from the project root:  python -m src.test_explainer
"""
import pandas as pd

from src import config
from src.explainer import explain

N_EACH = 5


def sample_emails():
    test = pd.read_csv(config.DATA_PROCESSED / "test.csv")
    preds = pd.read_csv(config.RESULTS_DIR / "stage1_test_predictions.csv")
    df = test.merge(preds[["id", "pred", "prob"]], on="id")
    phish = df[(df.eval_set == "test_phishing") & (df.pred == 1)]
    ham = df[(df.eval_set == "test_ham") & (df.pred == 0)]
    return pd.concat([phish.sample(N_EACH, random_state=config.SEED),
                      ham.sample(N_EACH, random_state=config.SEED)], ignore_index=True)


def run_once(emails, run):
    rows = []
    for e in emails.itertuples():
        rec = explain(e.text, int(e.pred), float(e.prob))
        rec.pop("user_prompt", None)   # contains the email body; logs keep only the prompt hash
        rows.append({"run": run, "id": e.id, "true_kind": e.kind, "clf_label": int(e.pred),
                     "clf_prob": round(float(e.prob), 3), **rec})
        print(f"  [{rec['status']:14s}] {e.id:16s} true={e.kind:9s} clf={int(e.pred)} "
              f"llm={rec['verdict']}  ({rec['seconds']}s)")
    return pd.DataFrame(rows)


def main():
    emails = sample_emails()
    print(f"model={config.MODEL_NAME} temperature={config.TEMPERATURE} seed={config.SEED}")
    runs = []
    for run in (1, 2):
        print(f"\nrun {run}:")
        runs.append(run_once(emails, run))
    log = pd.concat(runs, ignore_index=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log.to_csv(config.LOGS_DIR / f"step4_clean_test{config.MODEL_SUFFIX}.csv", index=False)

    r1 = runs[0]
    status_counts = r1.status.value_counts().to_dict()
    valid = r1[r1.status == "valid"]
    agree = (valid.verdict == valid.clf_label.map({1: "phishing", 0: "legitimate"})).sum()
    same = (runs[0].raw.values == runs[1].raw.values).sum()
    trunc = int(r1.truncated_by_model.fillna(False).sum())

    print("\nSummary (run 1):")
    print(f"  status counts: {status_counts}")
    print(f"  valid JSON: {len(valid)}/{len(r1)}")
    print(f"  LLM verdict == classifier label: {agree}/{len(valid)} valid replies")
    print(f"  identical raw reply in run 1 and run 2: {same}/{len(r1)}")
    print(f"  emails cut off by the model's context window: {trunc}")
    print(f"  mean seconds per email: {r1.seconds.mean():.1f}")
    print(f"\nlog saved to {(config.LOGS_DIR / f'step4_clean_test{config.MODEL_SUFFIX}.csv').relative_to(config.ROOT)}")

    print("\nReasons given (run 1):")
    for r in r1.itertuples():
        print(f"  {r.id} [{r.verdict}] {str(r.reason)[:160]}")

    flipped_label_check(emails)


def flipped_label_check(emails):
    """Give the LLM the WRONG classifier label. Does it read the email, or copy the label?"""
    print("\nFlipped-label check (LLM is told the opposite of the classifier's real verdict):")
    rows = []
    for e in emails.itertuples():
        flip = 1 - int(e.pred)
        rec = explain(e.text, flip, 1 - float(e.prob))
        true_v = "phishing" if e.kind == "phishing" else "legitimate"
        rows.append({"id": e.id, "true": true_v,
                     "told": "phishing" if flip else "legitimate", "llm": rec["verdict"]})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    for kind in ("phishing", "legitimate"):
        g = df[df.true == kind]
        print(f"  true {kind:10s}: kept the true verdict {(g.llm == g.true).sum()}/{len(g)}")
    df.to_csv(config.LOGS_DIR / f"step4_flipped_label_check{config.MODEL_SUFFIX}.csv", index=False)


if __name__ == "__main__":
    main()
