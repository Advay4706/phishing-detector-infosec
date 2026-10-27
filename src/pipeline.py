"""One call that runs the whole system on one email. The UI uses only this module.

    from src.pipeline import run
    run(email_text, payload_id="DO-1", location="body_end", defence="hierarchy")
"""
from functools import lru_cache

import requests

from src import config
from src.classifier import predict
from src.defences import explain_with_defence
from src.injection import LOCATIONS, insert, load_payloads


@lru_cache(maxsize=1)
def payloads_by_id():
    return {p["id"]: p for p in load_payloads()}


def ollama_status(timeout=3):
    """(ok, message): is the local Ollama server up and is the configured model downloaded?"""
    try:
        tags = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=timeout).json()
    except requests.RequestException:
        return False, (f"Ollama is not reachable at {config.OLLAMA_URL}. Start the Ollama app "
                       "(Start menu) and try again.")
    names = [m["name"] for m in tags.get("models", [])]
    if config.MODEL_NAME not in names:
        return False, f"Model {config.MODEL_NAME} is not downloaded. Run: ollama pull {config.MODEL_NAME}"
    return True, f"Ollama is running with {config.MODEL_NAME}."


def run(email_text, payload_id=None, location=None, defence="none", seed=config.SEED, nonce=None):
    """Inject (optional) -> Stage 1 classifier -> Stage 2 explainer with a defence -> result dict."""
    payload = None
    text = email_text[: config.MAX_EMAIL_CHARS]
    if payload_id:
        if location not in LOCATIONS:
            raise ValueError(f"location must be one of {LOCATIONS}")
        payload = payloads_by_id()[payload_id]
        text = insert(email_text, payload["text"], location)

    clf = predict(text)
    rec = explain_with_defence(text, clf["label"], clf["prob_malicious"], defence=defence,
                               seed=seed, nonce=nonce)

    valid = rec["status"] == "valid"
    clf_name = "phishing" if clf["label"] == 1 else "legitimate"
    error = None
    if rec["status"] == "error":
        error = ("The explainer did not answer (Ollama is down or too slow). "
                 f"Details: {rec['raw'][:200]}")
    elif not valid:
        error = f"The explainer's reply was not usable ({rec['status']})."

    return {
        "email_text": text,
        "payload_id": payload_id, "payload_type": payload["type"] if payload else None,
        "payload_text": payload["text"] if payload else None, "location": location,
        "defence": defence, "seed": seed, "model": config.MODEL_NAME,
        "clf_label": clf["label"], "clf_label_name": clf["label_name"],
        "clf_prob": clf["prob_malicious"],
        "status": rec["status"], "valid_json": valid,
        "verdict": rec["verdict"], "llm_verdict_raw": rec.get("llm_verdict_raw"),
        "reason": rec["reason"],
        "validator_intervened": rec.get("validator_intervened", False),
        "validator_note": rec.get("validator_note", ""),
        "layers_agree": valid and rec["verdict"] == clf_name,
        # the dangerous case: classifier says malicious, explainer tells the user it is safe
        "layers_disagree_unsafe": valid and clf["label"] == 1 and rec["verdict"] == "legitimate",
        "raw": rec["raw"], "seconds": rec["seconds"],
        "system_prompt": rec["system_prompt"], "user_prompt": rec["user_prompt"],
        "error": error,
    }
