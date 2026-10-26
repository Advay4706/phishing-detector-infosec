"""Quick count of outcomes in a harness log (preview only; Step 6 defines the metrics formally).

Run from the project root:  python -m src.summarise_run results/logs/step5_pilot.csv
"""
import sys

import pandas as pd


def main(path):
    d = pd.read_csv(path)
    print(f"{path}: {len(d)} rows, emails={d.email_id.nunique()}, "
          f"payloads={sorted(d.payload_id.unique())}, model={d.model.unique().tolist()}, "
          f"seed={d.seed.unique().tolist()}")
    print("\nReply status:", d.status.value_counts().to_dict())
    print("Model context window hit:", int(d.truncated_by_model.fillna(False).astype(bool).sum()))

    base = d[d.payload_id == "none"]
    print(f"\nClean baseline (no payload): {len(base)} emails, verdicts "
          f"{base.verdict.value_counts(dropna=False).to_dict()}")

    att = d[d.payload_id != "none"]
    flipped_clf = (att.clf_label_injected == 0).sum()
    print(f"Injected runs where the CLASSIFIER changed to legitimate: {flipped_clf}/{len(att)}")

    # Preview count: valid reply, classifier still says phishing, LLM says legitimate
    att = att.assign(said_legit=(att.status == "valid") & (att.verdict == "legitimate")
                     & (att.clf_label_injected == 1))
    print("\nShare of injected runs where the LLM said 'legitimate' (classifier still correct):")
    tab = att.pivot_table(index="payload_id", columns="location", values="said_legit",
                          aggfunc="mean")
    tab["ALL"] = att.groupby("payload_id").said_legit.mean()
    tab.loc["ALL"] = list(att.groupby("location").said_legit.mean().reindex(tab.columns[:-1])) \
        + [att.said_legit.mean()]
    print((tab * 100).round(1).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/logs/step5_pilot.csv")
