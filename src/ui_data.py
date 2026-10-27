"""Read-only helpers for the Streamlit UI. Everything here reads files; nothing runs a model.

The UI may only show numbers that exist in results/. Missing files -> None, and the UI says so.
"""
import json
import re
from functools import lru_cache

import pandas as pd

from src import config
from src.metrics import METRICS_DIR

DEMO_PATH = config.RESULTS_DIR / "demo_examples.json"


def list_result_sets():
    """Result sets are named by the prefix of results/metrics/<name>_seed_spread.csv
    (one such file per set; *_overall.csv would also match the <name>_de_overall.csv tables)."""
    if not METRICS_DIR.exists():
        return []
    names = sorted(p.name[: -len("_seed_spread.csv")] for p in METRICS_DIR.glob("*_seed_spread.csv"))
    order = {"final": 0, "defences": 1, "pilot": 2}
    return sorted(names, key=lambda n: (order.get(n, 9), n))


def load_table(name, table):
    path = METRICS_DIR / f"{name}_{table}.csv"
    return pd.read_csv(path) if path.exists() else None


def load_summary_json(name):
    path = config.RESULTS_DIR / f"{name}_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@lru_cache(maxsize=1)
def test_emails():
    return pd.read_csv(config.DATA_PROCESSED / "test.csv").set_index("id")


def email_text(email_id):
    return str(test_emails().loc[email_id, "text"])


def subject(email_id, n=80):
    return email_text(email_id).split("\n", 1)[0][:n]


@lru_cache(maxsize=1)
def sample_choices(n=30):
    """A fixed list of attack-pool emails for the dropdown (same seed as the experiments)."""
    from src.harness import sample_pool  # imported here: needs the classifier
    pool = sample_pool(n)
    return [(r.id, str(r.text).split("\n", 1)[0][:80]) for r in pool.itertuples()]


def demo_examples():
    if not DEMO_PATH.exists():
        return None
    return json.loads(DEMO_PATH.read_text(encoding="utf-8"))


def markdown_section(path, heading):
    """Return the text of '## heading' from a markdown file, or None."""
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    m = re.search(rf"^## {re.escape(heading)}\s*$(.*?)(?=^## |\Z)", text, re.M | re.S)
    return m.group(1).strip() if m else None
