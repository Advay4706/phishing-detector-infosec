"""Step 5: injection test harness.

For each attack-pool email x payload x location:
    1. insert the payload
    2. run Stage 1 on the INJECTED email (logged: did the injection change the classifier?)
    3. run Stage 2 with the injected email and the injected classifier label
    4. append one row to the CSV log
Each email also gets one "none" row (no payload) as its clean baseline.

Attack pool = Nazario test emails that Stage 1 labels phishing when clean.
The CSV is appended row by row, so an interrupted run can resume (--resume).

Run from the project root, e.g. the pilot:
    python -m src.harness --n-emails 50 --payloads DO-1,FV-1,OH-1 --out results/logs/step5_pilot.csv
"""
import argparse
import csv
import time
from pathlib import Path

import pandas as pd

from src import config
from src.classifier import predict_many
from src.explainer import PROMPT_VERSION, explain
from src.injection import LOCATIONS, insert, load_payloads

LOG_FIELDS = [
    "run_key", "email_id", "payload_id", "payload_type", "location", "seed", "model",
    "prompt_version", "defence", "clf_label_clean", "clf_prob_clean", "clf_label_injected",
    "clf_prob_injected", "status", "verdict", "reason", "raw", "injected_chars",
    "prompt_tokens", "truncated_by_model", "prompt_sha1", "system_prompt", "user_prompt",
    "seconds", "timestamp",
]


def attack_pool():
    """Nazario test emails the classifier gets RIGHT (phishing) when clean."""
    test = pd.read_csv(config.DATA_PROCESSED / "test.csv")
    test = test[test.eval_set == "test_phishing"].reset_index(drop=True)
    labels, probs = predict_many(test.text)
    test["clf_label_clean"], test["clf_prob_clean"] = labels, probs
    return test[test.clf_label_clean == 1].reset_index(drop=True)


def sample_pool(n, seed=config.SEED):
    pool = attack_pool()
    return pool if n >= len(pool) else pool.sample(n, random_state=seed).reset_index(drop=True)


def build_jobs(emails, payloads, locations):
    """Every (email, payload, location) combination, plus one clean 'none' job per email."""
    jobs = []
    for e in emails.itertuples():
        jobs.append((e, None, "none"))
        for p in payloads:
            for loc in locations:
                jobs.append((e, p, loc))
    return jobs


def run(emails, payloads, out_path, seed=config.SEED, locations=LOCATIONS,
        defence="none", explain_fn=explain, resume=True):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if resume and out_path.exists():
        done = set(pd.read_csv(out_path, usecols=["run_key"]).run_key)
    jobs = build_jobs(emails, payloads, locations)

    # Prepare all injected texts, then classify them in one fast batch
    prepared = []
    for e, p, loc in jobs:
        pid = p["id"] if p else "none"
        key = f"{e.id}|{pid}|{loc}|{seed}|{defence}"
        if key in done:
            continue
        text = e.text if p is None else insert(e.text, p["text"], loc)
        prepared.append((key, e, p, loc, text))
    print(f"{len(jobs)} jobs, {len(done)} already in log, {len(prepared)} to run")
    if not prepared:
        return out_path
    labels, probs = predict_many([t for *_, t in prepared])

    new_file = not out_path.exists()
    start = time.time()
    with out_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new_file:
            w.writeheader()
        for i, ((key, e, p, loc, text), lab, prob) in enumerate(zip(prepared, labels, probs), 1):
            rec = explain_fn(text, int(lab), float(prob), seed=seed)
            w.writerow({
                "run_key": key, "email_id": e.id,
                "payload_id": p["id"] if p else "none",
                "payload_type": p["type"] if p else "none",
                "location": loc, "seed": seed, "model": config.MODEL_NAME,
                "prompt_version": rec.get("prompt_version", PROMPT_VERSION), "defence": defence,
                "clf_label_clean": int(e.clf_label_clean),
                "clf_prob_clean": round(float(e.clf_prob_clean), 4),
                "clf_label_injected": int(lab), "clf_prob_injected": round(float(prob), 4),
                "status": rec["status"], "verdict": rec["verdict"], "reason": rec["reason"],
                "raw": rec["raw"], "injected_chars": len(text),
                "prompt_tokens": rec.get("prompt_tokens"),
                "truncated_by_model": rec.get("truncated_by_model"),
                "prompt_sha1": rec.get("prompt_sha1"), "system_prompt": rec.get("system_prompt"),
                "user_prompt": rec.get("user_prompt"), "seconds": rec.get("seconds"),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            f.flush()  # every row is on disk immediately, so a crash loses nothing
            if i % 25 == 0 or i == len(prepared):
                eta = (time.time() - start) / i * (len(prepared) - i) / 60
                print(f"  {i}/{len(prepared)} done, about {eta:.0f} min left", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-emails", type=int, default=50)
    ap.add_argument("--payloads", default="all", help="comma-separated ids, or 'all'")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", default=str(config.LOGS_DIR / "step5_pilot.csv"))
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    payloads = load_payloads()
    if args.payloads != "all":
        wanted = args.payloads.split(",")
        payloads = [p for p in payloads if p["id"] in wanted]
        assert len(payloads) == len(wanted), f"unknown payload id in {wanted}"
    emails = sample_pool(args.n_emails, seed=config.SEED)  # same emails for every seed
    print(f"model={config.MODEL_NAME} seed={args.seed} emails={len(emails)} "
          f"payloads={[p['id'] for p in payloads]} locations={list(LOCATIONS)}")
    run(emails, payloads, args.out, seed=args.seed, resume=not args.no_resume)


if __name__ == "__main__":
    main()
