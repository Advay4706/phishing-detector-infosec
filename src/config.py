"""Shared settings for the whole project.

Every script imports from here, so changing a value once changes it everywhere.
"""
import os
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
# Default model for all existing results. To run another model without touching them:
#   $env:PHISH_MODEL = "llama3.1:8b"   (PowerShell, before running any script)
DEFAULT_MODEL = "llama3.2:3b"
MODEL_NAME = os.environ.get("PHISH_MODEL", DEFAULT_MODEL)
MODEL_SUFFIX = "" if MODEL_NAME == DEFAULT_MODEL else "_" + MODEL_NAME.replace(":", "_").replace("/", "_").replace(".", "")
MODEL_TAG = MODEL_NAME.replace(":", "_").replace("/", "_").replace(".", "")  # for file names
TEMPERATURE = 0                               # 0 = always pick the most likely word
SEED = 42                                     # fixed seed for repeatable output
REQUEST_TIMEOUT = 300                         # seconds to wait for one LLM reply (larger models are slower)
NUM_CTX = 4096                                # context window in tokens (2000-char email ~ 600 tokens)
MAX_OUTPUT_TOKENS = 512                       # cap on reply length (256 cut off replies that quoted long URLs)
JSON_MODE = True                              # ask Ollama to return JSON (format="json")
VALID_VERDICTS = ("phishing", "legitimate")

# ---- Data -----------------------------------------------------------------
MAX_EMAIL_CHARS = 2000                        # cap email length (see truncation pitfall)

# ---- Experiments ----------------------------------------------------------
SEEDS = [42, 43, 44]                          # 3 seeds for repeated runs in Step 6
WORKERS = 4                                   # parallel LLM requests (measured: 3x faster, same verdicts)

# Sample sizes (the attack pool has 1,338 emails; each attacked email costs payloads x 5 calls)
PILOT_N_EMAILS = 50                           # Step 5/6 pilot: 50 emails x 3 payloads
PILOT_PAYLOADS = ["DO-1", "FV-1", "OH-1"]
FULL_N_EMAILS = 10                            # Step 8 full run: 10 emails x all 20 payloads x 5 locations
                                              # (measured ~1.1 s/call baseline, ~2.4 s/call defended,
                                              # 4 workers -> ~15,600 calls ~ 9 h for 5 conditions x 3 seeds)
HAM_N_EMAILS = 50                             # pilot: clean legitimate (CEAS ham) emails per condition,
FULL_HAM_N_EMAILS = 30                        # full run: same, to measure false alarms of a defence

# ---- Defences (Step 7) ----------------------------------------------------
DEFENCE_N_EMAILS = 15                         # Step 7: first 15 of the 50 pilot emails (and of the
                                              # 50 ham), all 3 pilot payloads x 5 locations, seed 42
                                              # (~720 LLM calls, ~30 min at 2.4 s/call)
DEFENCES = ["none", "delimiting", "hierarchy", "validation", "combined"]
DEFENCE = "none"                              # default condition for the harness
# Output validation overrides "legitimate" when the classifier is at least this sure the email
# is malicious. Fixed before any defence run (not tuned on test data).
VALIDATION_PROB_THRESHOLD = 0.90
