"""Step 1 smoke test: checks that every tool the project needs is working.

Run from the project root:  python -m src.smoke_test
"""
import importlib
import json
import sys

import requests

from src import config

failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


# 1. Python is running from the project's .venv
in_venv = sys.prefix != sys.base_prefix
check("Python inside .venv", in_venv, sys.executable)

# 2. Every library in requirements.txt imports
for lib in ["numpy", "pandas", "sklearn", "joblib", "bs4", "requests",
            "statsmodels", "matplotlib", "pytest"]:
    try:
        mod = importlib.import_module(lib)
        check(f"import {lib}", True, getattr(mod, "__version__", "ok"))
    except ImportError as e:
        check(f"import {lib}", False, str(e))

# 3. Config values are what the project plan requires
check("config TEMPERATURE == 0", config.TEMPERATURE == 0)
check("config SEED == 42", config.SEED == 42)
check("data/raw folder exists", config.DATA_RAW.is_dir(), str(config.DATA_RAW))

# 4. Ollama server is running and the model is downloaded
try:
    tags = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=10).json()
    names = [m["name"] for m in tags.get("models", [])]
    check("Ollama server reachable", True, config.OLLAMA_URL)
    check(f"model '{config.MODEL_NAME}' downloaded", config.MODEL_NAME in names, ", ".join(names))
except requests.RequestException as e:
    check("Ollama server reachable", False, str(e))


# 5. The model answers in JSON, and gives the SAME answer twice (temp 0 + seed)
def ask():
    r = requests.post(
        f"{config.OLLAMA_URL}/api/generate",
        json={
            "model": config.MODEL_NAME,
            "prompt": 'Reply with JSON only: {"status": "ok", "number": <any integer 1-10>}',
            "format": "json",
            "stream": False,
            "options": {"temperature": config.TEMPERATURE, "seed": config.SEED},
        },
        timeout=config.REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["response"]


if "Ollama server reachable" not in failures:
    try:
        a, b = ask(), ask()
        try:
            parsed = json.loads(a)
            check("model returns valid JSON", True, a.strip())
        except json.JSONDecodeError:
            check("model returns valid JSON", False, a.strip())
        check("same prompt -> same answer (deterministic)", a == b)
    except requests.RequestException as e:
        check("model generates a reply", False, str(e))

print()
if failures:
    print(f"SMOKE TEST FAILED: {len(failures)} problem(s): {failures}")
    sys.exit(1)
print("SMOKE TEST PASSED: environment is ready.")
