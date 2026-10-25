"""Stage 1: TF-IDF + classic ML classifier. Labels an email 0 (legitimate) or 1 (malicious).

Other steps only need:
    from src.classifier import predict
    predict("Subject\\nBody text...")  ->  {"label": 1, "label_name": "malicious", "prob_malicious": 0.97}
"""
from functools import lru_cache

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config

MODEL_PATH = config.MODELS_DIR / "stage1_classifier.joblib"
LABEL_NAMES = {0: "legitimate", 1: "malicious"}


def make_tfidf():
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),     # single words and word pairs ("verify account")
        min_df=2,               # ignore words seen in only 1 email (typos, IDs)
        max_df=0.95,            # ignore words in >95% of emails (carry no signal)
        max_features=50_000,    # keep the 50k most frequent terms
        sublinear_tf=True,      # 10 repeats of a word count less than 10x one
    )


def build_pipeline(name):
    """A Pipeline glues TF-IDF and the model together, so raw text goes in and a label comes out."""
    if name == "logreg":
        model = LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=config.SEED)
    elif name == "random_forest":
        model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                       n_jobs=-1, random_state=config.SEED)
    else:
        raise ValueError(f"unknown model {name}")
    return Pipeline([("tfidf", make_tfidf()), ("model", model)])


@lru_cache(maxsize=1)
def load_model():
    """Load the saved model once and reuse it (cache)."""
    bundle = joblib.load(MODEL_PATH)
    return bundle["pipeline"]


def predict(text):
    """Classify one email text. Returns label (0/1), its name, and P(malicious)."""
    pipe = load_model()
    prob = float(pipe.predict_proba([text])[0][1])
    label = int(prob >= 0.5)
    return {"label": label, "label_name": LABEL_NAMES[label], "prob_malicious": round(prob, 4)}


def predict_many(texts):
    """Classify a list of emails at once (much faster than calling predict() in a loop)."""
    pipe = load_model()
    probs = pipe.predict_proba(list(texts))[:, 1]
    return (probs >= 0.5).astype(int), probs
