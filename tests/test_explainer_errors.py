"""A reply that Ollama aborts for repeating itself is invalid model output, not a retryable error."""
import requests

from src import explainer


class FakeResponse:
    def __init__(self, code, text):
        self.status_code, self.text = code, text

    def raise_for_status(self):
        raise requests.HTTPError(response=self)


def fake_post(text):
    return lambda *a, **k: FakeResponse(500, text)


def test_repeat_limit_is_degenerate_output(monkeypatch):
    monkeypatch.setattr(explainer.requests, "post",
                        fake_post('{"error":"prediction aborted, token repeat limit reached"}'))
    rec = explainer.explain("email", 1, 0.95)
    assert rec["status"] == "degenerate_output"
    assert rec["verdict"] is None


def test_other_server_errors_stay_retryable(monkeypatch):
    monkeypatch.setattr(explainer.requests, "post", fake_post('{"error":"model not loaded"}'))
    assert explainer.explain("email", 1, 0.95)["status"] == "error"
