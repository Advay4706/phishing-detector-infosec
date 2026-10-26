"""Payload loading and insertion at 5 locations.

Email text format (from Step 2): first line = subject, the rest = body.
insert() always keeps the payload whole and trims the BODY instead, so the result
fits in config.MAX_EMAIL_CHARS (truncation pitfall: payload must sit before the cutoff).
"""
import json

from src import config

LOCATIONS = ("subject", "body_start", "body_end", "signature", "html_comment")
SIGNATURE_BLOCK = "\n\n-- \nKind regards,\nCustomer Support Team\n"


def load_payloads(path=config.PAYLOADS_DIR / "payloads.json"):
    data = json.loads(path.read_text(encoding="utf-8"))
    payloads = data["payloads"]
    ids = [p["id"] for p in payloads]
    assert len(ids) == len(set(ids)), "duplicate payload ids"
    assert all(p["type"] in data["types"] for p in payloads), "unknown payload type"
    return payloads


def split_email(text):
    subject, _, body = text.partition("\n")
    return subject, body


def _cut(body, room):
    """Trim the body to `room` characters (never negative)."""
    return body[: max(room, 0)]


def insert(text, payload, location, max_chars=config.MAX_EMAIL_CHARS):
    """Return the email with `payload` placed at `location`, at most max_chars long."""
    if location not in LOCATIONS:
        raise ValueError(f"unknown location {location}")
    subject, body = split_email(text)

    if location == "subject":
        subject = f"{subject} {payload}".strip()
        body = _cut(body, max_chars - len(subject) - 1)
        out = f"{subject}\n{body}"

    elif location == "body_start":
        head = f"{subject}\n{payload}\n\n"
        out = head + _cut(body, max_chars - len(head))

    elif location == "body_end":
        tail = f"\n\n{payload}"
        out = f"{subject}\n" + _cut(body, max_chars - len(subject) - 1 - len(tail)) + tail

    elif location == "signature":
        tail = SIGNATURE_BLOCK + payload
        out = f"{subject}\n" + _cut(body, max_chars - len(subject) - 1 - len(tail)) + tail

    else:  # html_comment: hidden comment in the middle of the body
        comment = f" <!-- {payload} --> "
        body = _cut(body, max_chars - len(subject) - 1 - len(comment))
        mid = len(body) // 2
        space = body.rfind(" ", 0, mid)          # split at a word boundary, not mid-word
        mid = space if space > 0 else mid
        out = f"{subject}\n{body[:mid]}{comment}{body[mid:]}"

    assert payload in out, "payload was cut off"
    assert len(out) <= max_chars, f"email too long: {len(out)}"
    return out
