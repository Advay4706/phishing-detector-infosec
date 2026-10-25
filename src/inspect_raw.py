"""Step 2, task 3: look at each raw Kaggle CSV before using it.

Prints columns, row count, label counts and 2 sample rows per label,
so label meanings come from the real data, not from assumptions.
Run from the project root:  python -m src.inspect_raw
"""
import pandas as pd

from src import config

KAGGLE_DIR = config.DATA_RAW / "kaggle_phishing"


def short(x, n=200):
    """Flatten newlines and cut to n characters so samples fit on screen."""
    s = " ".join(str(x).split())
    return s[:n] + ("..." if len(s) > n else "")


print("Files in data/raw/:")
for p in sorted(config.DATA_RAW.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(config.DATA_RAW)}  ({p.stat().st_size:,} bytes)")

for csv in sorted(KAGGLE_DIR.glob("*.csv")):
    df = pd.read_csv(csv, on_bad_lines="skip", encoding_errors="replace")
    print("\n" + "=" * 80)
    print(f"{csv.name}: {len(df):,} rows")
    print(f"columns: {list(df.columns)}")
    if "label" in df.columns:
        print("label counts:", df["label"].value_counts(dropna=False).to_dict())
        for lab in sorted(df["label"].dropna().unique()):
            print(f"\n  --- 2 samples with label={lab}")
            for _, row in df[df["label"] == lab].head(2).iterrows():
                print(f"  subject: {short(row.get('subject', ''), 100)}")
                print(f"  body:    {short(row.get('body', row.get('text_combined', '')))}")
