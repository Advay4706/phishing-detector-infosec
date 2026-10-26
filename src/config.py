"""Shared settings for the whole project.

Every script imports from here, so changing a value once changes it everywhere.
"""
from pathlib import Path

# ---- Paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"              # original datasets (never committed)
DATA_PROCESSED = ROOT / "data" / "processed"  # cleaned CSVs made in Step 2
MODELS_DIR = ROOT / "models"                  # saved classifier (Step 3)
PAYLOADS_DIR = ROOT / "payloads"              # payloads.json (Step 5)
RESULTS_DIR = ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"               # one CSV row per LLM run
FIGURES_DIR = RESULTS_DIR / "figures"

# ---- Local LLM (Ollama) ---------------------------------------------------
OLLAMA_URL = "http://localhost:11434"         # local only, no cloud API
MODEL_NAME = "llama3.2:3b"                    # may change in Step 4 if JSON is unreliable
TEMPERATURE = 0                               # 0 = always pick the most likely word
SEED = 42                                     # fixed seed for repeatable output
REQUEST_TIMEOUT = 120                         # seconds to wait for one LLM reply
NUM_CTX = 4096                                # context window in tokens (2000-char email ~ 600 tokens)
MAX_OUTPUT_TOKENS = 256                       # cap on reply length
JSON_MODE = True                              # ask Ollama to return JSON (format="json")
VALID_VERDICTS = ("phishing", "legitimate")

# ---- Data -----------------------------------------------------------------
MAX_EMAIL_CHARS = 2000                        # cap email length (see truncation pitfall)

# ---- Experiments ----------------------------------------------------------
SEEDS = [42, 43, 44]                          # 3 seeds for repeated runs in Step 6
