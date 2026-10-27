"""Injection test harness (Steps 5-8).

For each attack-pool email x payload x location (and one clean run per email):
    1. insert the payload
    2. run Stage 1 on the INJECTED email (logged: did the injection flip the classifier?)
    3. run Stage 2 with the chosen defence, given the injected classifier label
    4. append one row to the CSV log immediately
Optionally, clean legitimate (ham) emails are run too, to measure false alarms per defence.

Attack pool = Nazario test emails that Stage 1 labels phishing when clean.
Ham pool    = CEAS_08 ham test emails that Stage 1 labels legitimate when clean.
Logs never contain full email bodies: only ids, the payload id, a prompt hash and the reply.
Rows are appended one by one, so an interrupted run resumes where it stopped.

Examples (run from the project root):
    python -m src.harness --n-emails 50 --payloads DO-1,FV-1,OH-1 --out results/logs/pilot.csv
    python -m src.harness --defence hierarchy --seed 43 --out results/logs/pilot_hierarchy.csv
"""
import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import pandas as pd

from src import config
from src.classifier import predict_many
from src.defences import explain_with_defence
from src.injection import LOCATIONS, insert, load_payloads

LOG_FIELDS = [
    "run_key", "email_id", "email_kind", "payload_id", "payload_type", "location", "seed",
    "model", "prompt_version", "defence", "clf_label_clean", "clf_prob_clean",
    "clf_label_injected", "clf_prob_injected", "status", "verdict", "llm_verdict_raw",
    "validator_intervened", "validator_note", "reason", "raw", "injected_chars",
    "prompt_tokens", "truncated_by_model", "prompt_sha1", "system_prompt", "seconds",
    "timestamp",
]


def _pool(eval_set, want_label, n, seed):
    test = pd.read_csv(config.DATA_PROCESSED / "test.csv")
    test = test[test.eval_set == eval_set].reset_index(drop=True)
    labels, probs = predict_many(test.text)
    test["clf_label_clean"], test["clf_prob_clean"] = labels, probs
    pool = test[test.clf_label_clean == want_label].reset_index(drop=True)
    if n is not None and n < len(pool):
        pool = pool.sample(n, random_state=seed).reset_index(drop=True)
    return pool


def attack_pool():
    """All Nazario test emails the classifier gets RIGHT (phishing) when clean."""
    return _pool("test_phishing", 1, None, config.SEED)


def sample_pool(n, seed=config.SEED):
    """n attack-pool emails. Always sampled with config.SEED, so every seed and every defence
    uses the SAME emails."""
    return _pool("test_phishing", 1, n, seed)


def sample_ham(n, seed=config.SEED):
    """n CEAS ham emails the classifier labels legitimate when clean."""
    return _pool("test_ham", 0, n, seed)


def build_jobs(emails, payloads, locations, ham=None):
    jobs = []
    for e in emails.itertuples():
        jobs.append((e, "phishing", None, "none"))
        for p in payloads:
            for loc in locations:
                jobs.append((e, "phishing", p, loc))
    if ham is not None:
        for e in ham.itertuples():
            jobs.append((e, "ham", None, "none"))
    return jobs


def run(emails, payloads, out_path, seed=config.SEED, locations=LOCATIONS, defence="none",
        ham=None, resume=True, workers=config.WORKERS, explain_fn=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if explain_fn is None:
        explain_fn = partial(explain_with_defence, defence=defence)
    done = set()
    if resume and out_path.exists():
        # Rows with status "error" (timeout / server down) are retried; every other row is kept.
        # Metrics keep the LAST row per run_key, so a retried row replaces its error row.
        prev = pd.read_csv(out_path, usecols=["run_key", "status"])
        done = set(prev.run_key[prev.status != "error"])

    prepared = []
    for e, kind, p, loc in build_jobs(emails, payloads, locations, ham):
        pid = p["id"] if p else "none"
        key = f"{e.id}|{pid}|{loc}|{seed}|{defence}"
        if key in done:
            continue
        text = e.text if p is None else insert(e.text, p["text"], loc)
        prepared.append((key, e, kind, p, loc, text))
    total = len(build_jobs(emails, payloads, locations, ham))
    print(f"defence={defence} seed={seed}: {total} jobs, {total - len(prepared)} already in log, "
          f"{len(prepared)} to run", flush=True)
    if not prepared:
        return out_path
    labels, probs = predict_many([item[-1] for item in prepared])
    items = list(zip(prepared, labels, probs))

    def call(item):
        (key, e, kind, p, loc, text), lab, prob = item
        # nonce = run key, so the delimiting boundary token is random but reproducible
        return item, explain_fn(text, int(lab), float(prob), seed=seed, nonce=key)

    new_file = not out_path.exists()
    start = time.time()
    # Several requests go to Ollama at once (workers), but only this thread writes the CSV.
    with out_path.open("a", newline="", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new_file:
            w.writeheader()
        futures = [pool.submit(call, it) for it in items]
        for i, fut in enumerate(as_completed(futures), 1):
            ((key, e, kind, p, loc, text), lab, prob), rec = fut.result()
            w.writerow({
                "run_key": key, "email_id": e.id, "email_kind": kind,
                "payload_id": p["id"] if p else "none",
                "payload_type": p["type"] if p else "none",
                "location": loc, "seed": seed, "model": config.MODEL_NAME,
                "prompt_version": rec.get("prompt_version"), "defence": defence,
                "clf_label_clean": int(e.clf_label_clean),
                "clf_prob_clean": round(float(e.clf_prob_clean), 4),
                "clf_label_injected": int(lab), "clf_prob_injected": round(float(prob), 4),
                "status": rec["status"], "verdict": rec["verdict"],
                "llm_verdict_raw": rec.get("llm_verdict_raw", rec["verdict"]),
                "validator_intervened": rec.get("validator_intervened", False),
                "validator_note": rec.get("validator_note", ""),
                "reason": rec["reason"], "raw": rec["raw"], "injected_chars": len(text),
                "prompt_tokens": rec.get("prompt_tokens"),
                "truncated_by_model": rec.get("truncated_by_model"),
                "prompt_sha1": rec.get("prompt_sha1"), "system_prompt": rec.get("system_prompt"),
                "seconds": rec.get("seconds"), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            f.flush()  # every row is on disk immediately, so a crash loses nothing
            if i % 50 == 0 or i == len(prepared):
                eta = (time.time() - start) / i * (len(prepared) - i) / 60
                print(f"  [{defence} seed {seed}] {i}/{len(prepared)} done, "
                      f"about {eta:.0f} min left", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-emails", type=int, default=config.PILOT_N_EMAILS)
    ap.add_argument("--payloads", default=",".join(config.PILOT_PAYLOADS),
                    help="comma-separated ids, or 'all'")
    ap.add_argument("--n-ham", type=int, default=0, help="clean legitimate emails to add")
    ap.add_argument("--first", type=int, default=None,
                    help="use only the first N of the sampled emails and ham (a subset of the "
                         "same sample, so earlier runs on those emails are reused)")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--seeds", default=None, help="comma-separated, e.g. 42,43,44 (overrides --seed)")
    ap.add_argument("--defence", default=config.DEFENCE, choices=config.DEFENCES)
    ap.add_argument("--defences", default=None, help="comma-separated list, overrides --defence")
    ap.add_argument("--out", default=str(config.LOGS_DIR / "pilot.csv"))
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--workers", type=int, default=config.WORKERS,
                    help="parallel requests to Ollama (4 measured ~3x faster, same verdicts)")
    args = ap.parse_args()

    payloads = load_payloads()
    if args.payloads != "all":
        wanted = args.payloads.split(",")
        payloads = [p for p in payloads if p["id"] in wanted]
        assert len(payloads) == len(wanted), f"unknown payload id in {wanted}"
    emails = sample_pool(args.n_emails)
    ham = sample_ham(args.n_ham) if args.n_ham else None
    if args.first:
        emails = emails.head(args.first)
        ham = ham.head(args.first) if ham is not None else None
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    defences = args.defences.split(",") if args.defences else [args.defence]
    assert all(d in config.DEFENCES for d in defences), f"unknown defence in {defences}"
    print(f"model={config.MODEL_NAME} emails={len(emails)} ham={0 if ham is None else len(ham)} "
          f"payloads={len(payloads)} locations={len(LOCATIONS)} seeds={seeds} defences={defences}",
          flush=True)
    # Seed outer, defence inner: if a long run is cut short, every defence has the same seeds.
    for s in seeds:
        for d in defences:
            run(emails, payloads, args.out, seed=s, defence=d, ham=ham,
                resume=not args.no_resume, workers=args.workers)


if __name__ == "__main__":
    main()
