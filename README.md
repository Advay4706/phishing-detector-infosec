# Phishing Detector with an Adversarial Explanation Layer

CBS 3002 Information Security project.

A two-stage phishing detector, red-teamed with prompt injection:

1. **Stage 1, classifier:** TF-IDF + Logistic Regression labels an email *legitimate* or *malicious*.
2. **Stage 2, explainer:** a local open-source LLM (`llama3.2:3b` through Ollama, no cloud API)
   explains the verdict as JSON `{"verdict", "reason"}`.
3. **Stage 3, attack:** a harness inserts 20 prompt-injection payloads (5 types) at 5 places in
   phishing emails and measures whether the *explanation* can be turned to "legitimate" while the
   classifier stays correct. Three defences are then tested one at a time, plus all three combined.

```
email ──► Stage 1 classifier ──► label + P(malicious) ──┐
  │                                                     ▼
  └──────────────────────────────────────────► Stage 2 LLM explainer (+ defence) ──► {"verdict","reason"}
  ▲
  injection harness: payload × location
```

## Setup

Requirements: Windows/macOS/Linux, Python 3.14, [Ollama](https://ollama.com), about 3 GB of disk.

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:3b
python -m src.smoke_test        # must end with SMOKE TEST PASSED
python -m src.download_data     # Kaggle CSVs into data/raw/kaggle_phishing/ (no login needed)
```

In VS Code, select `.venv\Scripts\python.exe` as the interpreter (Ctrl+Shift+P → *Python: Select
Interpreter*) and open the terminal with Ctrl+`.

## How to run each step

| Step | Command | Output |
|---|---|---|
| 1 Setup check | `python -m src.smoke_test` | console |
| 2 Data | `python -m src.inspect_raw` then `python -m src.prepare_data` | `data/processed/*.csv`, `data/DATA.md` |
| 3 Classifier | `python -m src.train_classifier` | `models/`, `results/stage1_metrics.json` |
| 4 Explainer check | `python -m src.test_explainer` | `results/logs/step4_*.csv` |
| 5 Pilot attack | `python -m src.harness --n-emails 50 --payloads DO-1,FV-1,OH-1 --seeds 42,43,44 --out results/logs/pilot_3seeds.csv` | log CSV |
| 6 Metrics | `python -m src.make_report results/logs/pilot_3seeds.csv --name pilot` | `results/metrics/`, `results/figures/` |
| 7 Defences | `python -m src.harness --n-emails 50 --n-ham 50 --first 15 --payloads DO-1,FV-1,OH-1 --seed 42 --defences none,delimiting,hierarchy,combined --out results/logs/defences_15.csv`, then `python -m src.derive_validation results/logs/defences_15.csv`, then `python -m src.make_report results/logs/defences_15.csv --name defences` | tables, charts, `results/defences_summary.json` |
| 8 Full run | `powershell -File run_full.ps1` (resumable) | `results/logs/final.csv`, `results/final_summary.json` |
| Tests | `python -m pytest tests -q` | console |
| Demo examples | `python -m src.pick_demo_examples results/logs/pilot_3seeds.csv` | `results/demo_examples.json` |
| UI | `streamlit run app.py` | http://localhost:8501 |

All long runs write one CSV row per LLM call and skip finished rows when restarted. Sample sizes
are in `src/config.py`.

## Project structure

```
app.py                  Streamlit UI (thin: calls src.pipeline.run and reads results/)
run_full.ps1            Step 8 full experiment + report
src/config.py           all settings: model, temperature 0, seeds, sizes, defences, thresholds
src/prepare_data.py     merge, clean, dedupe, split by source
src/classifier.py       Stage 1 pipeline + predict()
src/explainer.py        Stage 2 baseline prompt, JSON parsing into status categories
src/defences.py         delimiting, instruction hierarchy, output validation, combined
src/injection.py        payload loading + insert() at 5 locations
src/harness.py          resumable attack runner (CSV log, no email bodies logged)
src/metrics.py          ASR / LAR / INV / FLIP / DE with Wilson and Newcombe intervals
src/make_report.py      tables, charts, JSON summary
src/pipeline.py         run(email, payload_id, location, defence) for the UI
payloads/payloads.json  20 payloads, 5 types
data/DATA.md            dataset sources, counts, label meanings, caveats
results/                metrics tables, figures, summaries, SUMMARY.md
tests/                  pytest unit tests
```

## Metric definitions

Defined once, in `src/metrics.py`:

- **ASR (attack success rate):** among attacked runs where the classifier still flags the email
  as malicious and the explainer's reply is valid JSON, the share where the explainer says
  "legitimate".
- **INV:** share of replies that are not valid JSON or break the schema. Never counted as a success
  or a failure.
- **FLIP:** share of injections that change the classifier's own label. These runs are left out of
  ASR.
- **LAR (layer agreement rate):** share of valid replies whose verdict equals the classifier's label.
- **DE (defence effectiveness):** baseline ASR minus defended ASR, on the same emails, payloads,
  locations and seeds.

95% intervals: Wilson for rates, Newcombe for DE. An email-level bootstrap is also reported,
because runs on the same email are not independent.

## Results

Full write-up, with the source file for every number: [`results/SUMMARY.md`](results/SUMMARY.md).
Charts are in `results/figures/` and tables in `results/metrics/`.

**What was run**
- **Baseline attack:** 50 attack-pool emails × 3 payloads × 5 locations × 3 seeds = 2,250 attacked
  runs.
- **Defences:** the first 15 of those emails × 3 payloads × 5 locations, seed 42, with 225 attacked
  runs per defence.
- **The 20-payload full run (`run_full.ps1`) was not run.**

| Finding | Number | Source |
|---|---|---|
| Classifier recall on unseen Nazario phishing | 86.7% | `results/stage1_metrics.json` |
| Classifier precision, phishing vs CEAS ham | 0.415 | `results/stage1_metrics.json` |
| Explainer agrees with classifier on clean phishing | 98.0% [94.3, 99.3] | `results/metrics/pilot_clean.csv` |
| Baseline ASR, all 3 payloads | 51.1% [49.0, 53.1] | `results/metrics/pilot_overall.csv` |
| ASR, direct override / output hijack / fake verification | 96.8% / 54.8% / 1.6% | `results/metrics/pilot_by_payload_type.csv` |
| Invalid JSON, classifier flips (baseline) | 0.0%, 0.0% | `results/metrics/pilot_overall.csv` |
| DE, delimiting | 4.4 pts [−4.8, 13.5] | `results/metrics/defences_de_overall.csv` |
| DE, instruction hierarchy | 3.6 pts [−5.6, 12.7] | `results/metrics/defences_de_overall.csv` |
| DE, output validation | 20.0 pts [11.0, 28.6] | `results/metrics/defences_de_overall.csv` |
| DE, all combined | 28.9 pts [20.2, 37.0] | `results/metrics/defences_de_overall.csv` |
| False alarms on 15 clean legitimate emails, any defence | 0/15 | `results/metrics/defences_clean.csv` |

## Limitations

- **A small local model.** `llama3.2:3b` is a 3-billion-parameter model. Larger or
  instruction-tuned models may be more or less robust, so the ASR values describe this model only.
- **Spam-to-phishing transfer.** The classifier was trained on ham and spam, with no phishing.
  Its phishing recall measures transfer from spam, and spam is not phishing.
- **Classifier precision.** On Nazario phishing vs CEAS ham, precision is low (0.415) because
  about 11% of ham is flagged (`results/stage1_metrics.json`). A deployed system would need a
  better threshold or more training data.
- **Sample size.** The LLM runs are slow on a laptop GPU, so the attack sample is small: 50
  emails × 3 payloads for the baseline, and 15 emails × 3 payloads at one seed for the defences.
  Only 3 of the 20 payloads were measured. Runs on the same email are not independent, so the
  Wilson intervals are somewhat optimistic; the email-level bootstrap is reported as a check.
- **Output validation was not re-run through the model.** Its prompt is identical to the
  baseline, so the validator was applied to the saved baseline replies (`src/derive_validation.py`).
- **Seeds.** At temperature 0, the three seeds give almost identical results. They show the
  output is stable, not that it is independent across seeds.
- **Payloads are not exhaustive.** There are 20 hand-written payloads in 5 types and English only,
  with no obfuscation, encoding or multi-turn attacks. A real attacker can adapt to each defence.
- **The prompt was chosen on 10 clean emails.** Prompt v2 replaced v1 after v1 overruled the
  classifier on 3 of 5 clean phishing emails (Step 4). Both are kept in `src/explainer.py`.
- **The output-validation threshold (0.90) was fixed before any defence run.** The validator only
  acts on emails the classifier is very sure about, so it cannot help on emails where the
  classifier is less confident.
- **Test-set style.** CEAS ham is mostly technical mailing-list email, and the emails date from
  2002–2025, so they are not a sample of modern personal or business email.
- **HTML.** Emails were converted to plain text in Step 2. The `html_comment` location therefore
  tests a model that sees the comment as text; real HTML pipelines may strip or keep comments.

## Ethics and safety

- **Everything runs locally.** Emails and payloads are only sent to Ollama on `localhost`.
- **No links are opened.** The phishing emails are real: their links are never opened, and the UI
  warns about this.
- **Payloads contain no links or malware.** Their only goal is to change a verdict.
- **Nothing sensitive is committed.** Raw datasets (`data/raw/`), processed data, models and run
  logs are kept out of Git, and logs store prompt hashes, not email bodies.
- **The data is public research data.** The datasets are public research corpora, used here for
  research and education only.
