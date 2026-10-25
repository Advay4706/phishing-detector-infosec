"""Step 2: build one clean, source-split dataset from the Kaggle CSVs.

Output:
  data/processed/train.csv   Enron + SpamAssassin + Ling
  data/processed/test.csv    Nazario, Nigerian_Fraud, CEAS_08 (each tagged by eval_set)
  data/DATA.md               written from the real counts below

Run from the project root:  python -m src.prepare_data
"""
import csv
import re
import warnings

import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

from src import config

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
csv.field_size_limit(2**31 - 1)  # some email bodies are longer than csv's default limit

KAGGLE_DIR = config.DATA_RAW / "kaggle_phishing"

# source name -> (file, split, kind for label 0, kind for label 1)
# Label meanings were confirmed by reading samples (python -m src.inspect_raw).
SOURCES = {
    "Enron":          ("Enron.csv",          "train", "ham", "spam"),
    "SpamAssassin":   ("SpamAssasin.csv",    "train", "ham", "spam"),
    "Ling":           ("Ling.csv",           "train", "ham", "spam"),
    "Nazario":        ("Nazario.csv",        "test",  None,  "phishing"),
    "Nigerian_Fraud": ("Nigerian_Fraud.csv", "test",  None,  "phishing"),
    "CEAS_08":        ("CEAS_08.csv",        "test",  "ham", "spam"),
}
# Which copy survives when the same email is in two sources (train wins, so
# the test sets stay clean of anything the classifier was trained on).
PRIORITY = ["Enron", "SpamAssassin", "Ling", "Nazario", "Nigerian_Fraud", "CEAS_08"]

# Mailbox-software placeholder messages, not real emails (seen in Nazario.csv).
JUNK_PATTERNS = re.compile(r"FOLDER INTERNAL DATA|internal format of your mail folder", re.I)
HTML_TAG = re.compile(r"<\s*/?\s*[a-zA-Z!][^>]*>")


def strip_html(text):
    """Remove HTML tags but keep the visible words. Only parse if a tag is present."""
    if HTML_TAG.search(text):
        text = BeautifulSoup(text, "html.parser").get_text(" ")
    return text


def clean_text(subject, body):
    subject = "" if pd.isna(subject) else str(subject)
    body = "" if pd.isna(body) else str(body)
    text = strip_html(subject) + "\n" + strip_html(body)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)        # collapse spaces
    text = re.sub(r"\n\s*\n+", "\n", text).strip()    # collapse blank lines
    return text[: config.MAX_EMAIL_CHARS]             # length cap (truncation pitfall)


def dedup_key(text):
    """Lowercase letters+digits only, so '$ 39 . 85' and '$39.85' count as the same."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def load_source(name):
    fname, split, kind0, kind1 = SOURCES[name]
    bad = []
    df = pd.read_csv(KAGGLE_DIR / fname, engine="python", encoding_errors="replace",
                     on_bad_lines=lambda line: bad.append(line))
    rows_read = len(df)
    unknown = set(df["label"].dropna().unique()) - {0, 1}
    assert not unknown, f"{name}: unexpected labels {unknown}"
    if kind0 is None:
        assert (df["label"] == 1).all(), f"{name}: expected only label 1"

    out = pd.DataFrame({
        "text": [clean_text(s, b) for s, b in zip(df["subject"], df["body"])],
        "kind": df["label"].map({0: kind0, 1: kind1}),
        "source": name,
        "split": split,
    })
    out["label"] = (out["kind"] != "ham").astype(int)  # 0 = legitimate, 1 = malicious
    out["id"] = [f"{name}_{i}" for i in range(len(out))]
    return out, rows_read, len(bad)


def main():
    log = []  # (stage, source, removed, reason) for the report
    frames, raw_counts = [], {}
    for name in SOURCES:
        df, n, n_bad = load_source(name)
        raw_counts[name] = n
        print(f"loaded {name:15s} {n:6,} rows  (unparseable lines skipped: {n_bad})")
        if n_bad:
            log.append(("unparseable CSV line", name, n_bad))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["key"] = df["text"].map(dedup_key)

    # 1. Empty text, or placeholder junk
    empty = df["key"].str.len() < 20                         # under 20 letters/digits = no content
    junk = df["text"].str.contains(JUNK_PATTERNS)
    for name, n in df[empty].groupby("source").size().items():
        log.append(("empty / near-empty text", name, n))
    for name, n in df[junk & ~empty].groupby("source").size().items():
        log.append(("mailbox placeholder, not an email", name, n))
    df = df[~(empty | junk)]

    # 2. Same email, different labels in different places -> unreliable, drop all copies
    conflict_keys = df.groupby("key")["label"].nunique()
    conflict_keys = set(conflict_keys[conflict_keys > 1].index)
    conflict = df["key"].isin(conflict_keys)
    for name, n in df[conflict].groupby("source").size().items():
        log.append(("same text with conflicting labels", name, n))
    df = df[~conflict]

    # 3. Duplicates inside one source (keep the first)
    within = df.duplicated(subset=["source", "key"], keep="first")
    for name, n in df[within].groupby("source").size().items():
        log.append(("duplicate within same source", name, n))
    df = df[~within]

    # 4. Duplicates across sources (keep the highest-priority copy: train first)
    df = df.assign(_p=df["source"].map(PRIORITY.index)).sort_values(["_p", "id"])
    across = df.duplicated(subset=["key"], keep="first")
    for name, n in df[across].groupby("source").size().items():
        log.append(("duplicate of an email in another source", name, n))
    df = df[~across].drop(columns="_p")

    # Evaluation-set tag for the test file
    def eval_set(r):
        if r.split == "train":
            return "train"
        if r.source == "Nazario":
            return "test_phishing"
        if r.source == "Nigerian_Fraud":
            return "test_phishing_nigerian"
        return "test_ham" if r.kind == "ham" else "test_spam_transfer"
    df["eval_set"] = [eval_set(r) for r in df.itertuples()]

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    # Leakage check: no email text (normalised) may be in both train and test
    overlap = set(train["key"]) & set(test["key"])
    assert not overlap, f"{len(overlap)} texts appear in both train and test"
    exact_overlap = set(train["text"]) & set(test["text"])
    assert not exact_overlap

    cols = ["id", "text", "label", "kind", "source", "split", "eval_set"]
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    train[cols].to_csv(config.DATA_PROCESSED / "train.csv", index=False)
    test[cols].to_csv(config.DATA_PROCESSED / "test.csv", index=False)

    # ---- Report ----
    print("\nRemoved rows:")
    removed = pd.DataFrame(log, columns=["reason", "source", "removed"])
    print(removed.to_string(index=False) if len(removed) else "  none")
    print(f"total removed: {removed['removed'].sum() if len(removed) else 0:,}")

    summary = (df.groupby(["source", "kind", "split", "eval_set"]).size()
                 .reset_index(name="rows"))
    summary["_p"] = summary["source"].map(PRIORITY.index)
    summary = summary.sort_values(["_p", "kind"]).drop(columns="_p")
    print("\nSummary:")
    print(summary[["source", "kind", "rows", "split", "eval_set"]].to_string(index=False))
    print(f"\ntrain.csv: {len(train):,} rows  (legit {int((train.label == 0).sum()):,}, "
          f"malicious {int((train.label == 1).sum()):,})")
    print(f"test.csv:  {len(test):,} rows  (legit {int((test.label == 0).sum()):,}, "
          f"malicious {int((test.label == 1).sum()):,})")
    print("leakage check: 0 texts shared between train and test  [PASS]")
    capped = (df["text"].str.len() >= config.MAX_EMAIL_CHARS).sum()
    print(f"emails cut to {config.MAX_EMAIL_CHARS} chars: {capped:,} of {len(df):,}")

    write_data_md(raw_counts, summary, removed, train, test, capped, len(df))


def write_data_md(raw_counts, summary, removed, train, test, capped, total):
    meaning = {
        "Enron": "0 = ham (Enron business email), 1 = spam",
        "SpamAssassin": "0 = ham (mailing lists / personal), 1 = spam",
        "Ling": "0 = ham (Linguist mailing list), 1 = spam",
        "Nazario": "all 1 = phishing (credential-theft emails)",
        "Nigerian_Fraud": "all 1 = advance-fee (419) fraud",
        "CEAS_08": "0 = ham (developer mailing lists), 1 = spam",
    }
    use = {
        "Enron": "train", "SpamAssassin": "train", "Ling": "train",
        "Nazario": "test: `test_phishing` (main held-out phishing set, attack pool for Step 5)",
        "Nigerian_Fraud": "test: `test_phishing_nigerian` (second held-out set, reported separately)",
        "CEAS_08": "test: ham -> `test_ham`; spam -> `test_spam_transfer` (reported separately)",
    }
    kept = summary.groupby("source")["rows"].sum()
    lines = [
        "# Data",
        "",
        "Source: Kaggle dataset `naserabdullahalam/phishing-email-dataset` (text-only CSVs),",
        "downloaded with `kagglehub` into `data/raw/kaggle_phishing/` (not committed to Git).",
        "Built by `python -m src.prepare_data`. All numbers below are printed by that script.",
        "",
        "## Sources",
        "",
        "| Source | File | Raw rows | Kept rows | Label meaning (checked on samples) | Used for |",
        "|---|---|---|---|---|---|",
    ]
    for name, (fname, *_rest) in SOURCES.items():
        lines.append(f"| {name} | {fname} | {raw_counts[name]:,} | {int(kept.get(name, 0)):,} "
                     f"| {meaning[name]} | {use[name]} |")
    lines += [
        "",
        "`phishing_email.csv` from the same Kaggle dataset is **not used**: it is a merged,",
        "preprocessed copy of the files above, so using it would leak test emails into training.",
        "",
        "## Rows per split",
        "",
        "| Source | Kind | Rows | Split | Eval set |",
        "|---|---|---|---|---|",
    ]
    for r in summary.itertuples():
        lines.append(f"| {r.source} | {r.kind} | {r.rows:,} | {r.split} | {r.eval_set} |")
    lines += [
        "",
        f"- train.csv: {len(train):,} rows ({int((train.label == 0).sum()):,} legitimate, "
        f"{int((train.label == 1).sum()):,} malicious)",
        f"- test.csv: {len(test):,} rows ({int((test.label == 0).sum()):,} legitimate, "
        f"{int((test.label == 1).sum()):,} malicious)",
        "",
        "## Columns",
        "",
        "- `id`: source name + row number in the raw file",
        "- `text`: subject + newline + body, HTML removed, whitespace collapsed, "
        f"cut to {config.MAX_EMAIL_CHARS} characters",
        "- `label`: 0 = legitimate, 1 = malicious",
        "- `kind`: ham, spam or phishing",
        "- `source`, `split` (train/test), `eval_set` (which test set the row belongs to)",
        "",
        "## Cleaning",
        "",
        "Removed in this order: near-empty text (under 20 letters/digits) and mailbox",
        "placeholders; texts with conflicting labels (all copies); duplicates within a source;",
        "duplicates across sources (the train copy is kept, so test stays unseen).",
        "Duplicates are matched on lowercase letters and digits only, so formatting differences",
        "such as `$ 39 . 85` vs `$39.85` do not hide a copy.",
        "",
        "| Reason | Source | Removed |",
        "|---|---|---|",
    ]
    for r in removed.itertuples():
        lines.append(f"| {r.reason} | {r.source} | {r.removed:,} |")
    lines += [
        "",
        f"{capped:,} of {total:,} emails were cut at {config.MAX_EMAIL_CHARS} characters.",
        "Check run by the script: no text (normalised or exact) appears in both train and test.",
        "",
        "## Caveats",
        "",
        "- **Spam was treated as malicious in training.** The training sets contain ham and spam",
        "  but no phishing. The classifier therefore learns \"spam-like vs legitimate\", and the",
        "  phishing test sets measure how well that transfers to phishing. Spam is not phishing;",
        "  results on `test_phishing` and `test_spam_transfer` are reported separately.",
        "- Enron and Ling were stored lowercased with spaces around punctuation by the dataset",
        "  author; the other sources keep normal casing. Style differences between sources can",
        "  help or hurt a classifier, which is why per-source accuracy is reported (Step 3).",
        "- CEAS_08 ham is mostly technical mailing-list mail, so `test_ham` is not a sample of",
        "  everyday personal or business email.",
        "- Emails are capped at 2000 characters; injected payloads in Step 5 must sit before the cut.",
        "",
    ]
    (config.ROOT / "data" / "DATA.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote data/DATA.md")


if __name__ == "__main__":
    main()
