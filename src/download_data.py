"""Download the text-only Kaggle dataset into data/raw/kaggle_phishing/ (no login needed).

    python -m src.download_data
"""
import kagglehub

from src import config

OUT = config.DATA_RAW / "kaggle_phishing"

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    path = kagglehub.dataset_download("naserabdullahalam/phishing-email-dataset", output_dir=str(OUT))
    print("saved to", path)
