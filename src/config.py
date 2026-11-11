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
# Default model. Version 1 of the project (branch v1) used llama3.2:3b; its results are kept
# in results/ as the "pilot" and "defences" sets for comparison.
# To run another model without touching the default-model logs:
#   $env:PHISH_MODEL = "llama3.2:3b"   (PowerShell, before running any script)
DEFAULT_MODEL = "llama3.1:8b"
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
SEEDS = [42]                                  # one seed: at temperature 0 the v1 runs with seeds
                                              # 42, 43, 44 differed by at most 0.8 points
WORKERS = 4                                   # parallel LLM requests (same verdicts as 1 worker; about
                                              # 3x faster on llama3.2:3b, 1.5x on llama3.1:8b)

# Sample sizes (the attack pool has 1,338 emails; each attacked email costs payloads x 5 calls)
PILOT_N_EMAILS = 50                           # Step 5/6 pilot: 50 emails x 3 payloads
PILOT_PAYLOADS = ["DO-1", "FV-1", "OH-1"]
FULL_N_EMAILS = 15                            # full run: the first 15 of the 50 pilot emails (the same
                                              # 15 as the v1 defence run) x all 20 payloads x 5 locations.
                                              # Measured on llama3.1:8b, 4 workers: ~3.7 s/call, so
                                              # 1,530 calls ~ 1.6 h per condition, ~6.5 h for 4 conditions
HAM_N_EMAILS = 50                             # pilot: clean legitimate (CEAS ham) emails per condition,
FULL_HAM_N_EMAILS = 15                        # full run: the first 15 of the 50 pilot ham emails,

# ---- Defences (Step 7) ----------------------------------------------------
DEFENCE_N_EMAILS = 15                         # Step 7: first 15 of the 50 pilot emails (and of the
                                              # 50 ham), all 3 pilot payloads x 5 locations, seed 42
                                              # (~720 LLM calls, ~30 min at 2.4 s/call)
DEFENCES = ["none", "delimiting", "hierarchy", "validation", "combined"]
DEFENCE = "none"                              # default condition for the harness
# Output validation overrides "legitimate" when the classifier is at least this sure the email
# is malicious. Fixed before any defence run (not tuned on test data).
VALIDATION_PROB_THRESHOLD = 0.90
