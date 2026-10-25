"""Step 3: choose, train and evaluate the Stage 1 classifier.

1. Model selection with leave-one-source-out on the TRAIN sources only
   (train on 2 of Enron/SpamAssassin/Ling, score on the 3rd). Test data is not touched.
2. Refit the winner on all training data, evaluate ONCE on the held-out test sets.
3. Save the model and the metrics.

Run from the project root:  python -m src.train_classifier
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from src import config
from src.classifier import MODEL_PATH, build_pipeline

MODELS = ["logreg", "random_forest"]
TRAIN_SOURCES = ["Enron", "SpamAssassin", "Ling"]


def scores(y_true, y_pred):
    return {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def leave_one_source_out(train):
    rows = []
    for name in MODELS:
        for held in TRAIN_SOURCES:
            fit_df, val_df = train[train.source != held], train[train.source == held]
            t = time.time()
            pipe = build_pipeline(name).fit(fit_df.text, fit_df.label)
            pred = pipe.predict(val_df.text)
            s = scores(val_df.label, pred)
            rows.append({"model": name, "held_out_source": held, **s,
                         "seconds": round(time.time() - t, 1)})
            print(f"  {name:13s} held out {held:12s}  F1={s['f1']:.3f}  acc={s['accuracy']:.3f}"
                  f"  ({rows[-1]['seconds']}s)")
    return pd.DataFrame(rows)


def evaluate_test(pipe, test):
    pred = pipe.predict(test.text)
    prob = pipe.predict_proba(test.text)[:, 1]
    test = test.assign(pred=pred, prob=prob)

    # Per eval set: each set has one true class, so accuracy = share labelled correctly
    per_set = []
    for es, g in test.groupby("eval_set"):
        per_set.append({"eval_set": es, "source": g.source.iloc[0], "true_kind": g.kind.iloc[0],
                        "n": len(g), "accuracy": accuracy_score(g.label, g.pred),
                        "flagged_malicious": int(g.pred.sum())})
    per_set = pd.DataFrame(per_set)

    # Main phishing task: Nazario phishing vs CEAS ham
    main = test[test.eval_set.isin(["test_phishing", "test_ham"])]
    allt = test
    cm = confusion_matrix(main.label, main.pred, labels=[0, 1])
    return test, per_set, scores(main.label, main.pred), scores(allt.label, allt.pred), cm


def main():
    train = pd.read_csv(config.DATA_PROCESSED / "train.csv")
    test = pd.read_csv(config.DATA_PROCESSED / "test.csv")
    print(f"train {len(train):,} rows, test {len(test):,} rows")

    print("\n1) Leave-one-source-out model selection (train sources only):")
    loso = leave_one_source_out(train)
    summary = loso.groupby("model")[["f1", "accuracy", "precision", "recall"]].mean()
    print("\nmean over the 3 held-out sources:")
    print(summary.round(3).to_string())
    best = summary["f1"].idxmax()
    print(f"\nbest by mean F1: {best}")

    print(f"\n2) Refit {best} on all training data and evaluate on held-out test sets")
    pipe = build_pipeline(best).fit(train.text, train.label)
    test_pred, per_set, main_s, all_s, cm = evaluate_test(pipe, test)

    print("\nPer test set (accuracy = share given the correct label):")
    print(per_set.round(3).to_string(index=False))
    print("\nMain task, Nazario phishing vs CEAS ham:")
    print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in main_s.items()))
    print(f"  confusion matrix [rows=true 0/1, cols=pred 0/1]: {cm.tolist()}")
    print("All test sets combined:")
    print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in all_s.items()))

    # Preview for Step 5: Nazario emails the classifier flags correctly when clean
    pool = test_pred[(test_pred.eval_set == "test_phishing") & (test_pred.pred == 1)]
    print(f"\nNazario phishing emails flagged correctly (future attack pool): "
          f"{len(pool):,} of {(test_pred.eval_set == 'test_phishing').sum():,}")

    # 3) Save
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipe, "model_name": best, "seed": config.SEED,
                 "trained_on": TRAIN_SOURCES}, MODEL_PATH)
    test_pred[["id", "source", "eval_set", "label", "pred", "prob"]].to_csv(
        config.RESULTS_DIR / "stage1_test_predictions.csv", index=False)
    metrics = {
        "best_model": best,
        "seed": config.SEED,
        "selection_metric": "mean F1 over leave-one-source-out folds",
        "loso_folds": loso.to_dict(orient="records"),
        "loso_mean": summary.reset_index().to_dict(orient="records"),
        "test_per_set": per_set.to_dict(orient="records"),
        "test_main_phishing_vs_ham": main_s,
        "test_main_confusion_matrix": cm.tolist(),
        "test_all": all_s,
        "attack_pool_size": int(len(pool)),
    }
    (config.RESULTS_DIR / "stage1_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    print(f"\nsaved {MODEL_PATH.relative_to(config.ROOT)}, results/stage1_metrics.json, "
          f"results/stage1_test_predictions.csv")


if __name__ == "__main__":
    np.random.seed(config.SEED)
    main()
