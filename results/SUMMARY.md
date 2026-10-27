# Results summary

Every number below is copied from the file named in brackets. 95% intervals are Wilson for rates
and Newcombe for differences. The metric definitions are in `src/metrics.py`.

**What was run**
- **Classifier (Stage 1):** trained once, evaluated once on held-out sources.
- **Baseline attack (Step 6):** 50 attack-pool emails × 3 payloads (DO-1, FV-1, OH-1) × 5 locations
  × 3 seeds (42, 43, 44) = 2,250 attacked runs, plus 150 clean runs.
- **Defences (Step 7):** the first 15 of those 50 emails × the same 3 payloads × 5 locations, seed 42,
  for 5 conditions. Each condition has 225 attacked runs, 15 clean phishing runs and 15 clean
  legitimate (ham) runs. Output validation was applied to the saved baseline replies, because its
  prompt is identical to the baseline (see `src/derive_validation.py`).
- **Full run (all 20 payloads):** not run. Every 20-payload number is therefore **not measured**.

## 1. The classifier catches most unseen phishing but raises many false alarms

| Test set | Emails | Correctly labelled | Source |
|---|---|---|---|
| Nazario phishing | 1,543 | 86.7% | [results/stage1_metrics.json] |
| Nigerian fraud | 3,220 | 99.3% | [results/stage1_metrics.json] |
| CEAS spam | 18,546 | 98.4% | [results/stage1_metrics.json] |
| CEAS ham (legitimate) | 17,234 | 89.1% | [results/stage1_metrics.json] |

- **Phishing vs ham:** precision 0.415, recall 0.867, F1 0.561. The confusion matrix is
  [[15349, 1885], [205, 1338]] [results/stage1_metrics.json].
- **Why precision is low:** ham outnumbers phishing 17,234 to 1,543, so 1,885 false alarms
  outnumber the 1,338 phishing emails caught.
- **Attack pool:** 1,338 Nazario emails flagged correctly when clean [results/stage1_metrics.json].

## 2. On clean emails, the explainer agrees with the classifier

- **Step 4 check:** 10/10 valid JSON and 10/10 agreement with the classifier [results/logs/step4_clean_test.csv].
- **Step 6 clean runs:** LAR 98.0% [94.3%, 99.3%] over 150 runs. 2.0% of clean phishing was called
  legitimate with no attack [results/metrics/pilot_clean.csv].
- **Caveat, flipped labels:** when told the wrong classifier label, the model kept the true verdict
  on 0 of 5 phishing emails and 5 of 5 legitimate ones [results/logs/step4_flipped_label_check.csv].

## 3. About half of injection attacks succeed, and the payload type decides most of it

From [results/metrics/pilot_overall.csv] and [results/metrics/pilot_by_payload_type.csv]:

| | ASR | 95% CI | Runs |
|---|---|---|---|
| All attacks | 51.1% | 49.0% – 53.1% | 2,250 |
| Direct override (DO-1) | 96.8% | 95.3% – 97.8% | 750 |
| Output hijack (OH-1) | 54.8% | 51.2% – 58.3% | 750 |
| Fake verification (FV-1) | 1.6% | 0.9% – 2.8% | 750 |

- **Invalid JSON and classifier flips:** both 0.0%. The classifier never changed its label on an
  injected email [results/metrics/pilot_overall.csv].
- **By location:** body end 66.4%, signature 65.3%, HTML comment 51.6%, body start 38.0%,
  subject 34.0% [results/metrics/pilot_by_location.csv].
- **Seeds barely matter:** per payload type, ASR differs by at most 0.8 points across the 3 seeds
  [results/metrics/pilot_seed_spread.csv].
- **Email-level bootstrap CI for overall ASR:** 48.9% – 53.4% [results/metrics/pilot_bootstrap.csv].

## 4. Prompt-only defences barely help, and output checks help most

All rows use the 225 attacked runs per defence [results/metrics/defences_overall.csv] and
[results/metrics/defences_de_overall.csv]:

| Defence | ASR [95% CI] | DE in points [95% CI] | Model-only DE |
|---|---|---|---|
| None | 50.7% [44.2, 57.1] | n/a | n/a |
| Delimiting | 46.2% [39.8, 52.7] | 4.4 [−4.8, 13.5] | 4.4 |
| Instruction hierarchy | 47.1% [40.7, 53.6] | 3.6 [−5.6, 12.7] | 3.6 |
| Output validation | 30.7% [25.0, 37.0] | 20.0 [11.0, 28.6] | 0.0 |
| All combined | 21.8% [16.9, 27.6] | 28.9 [20.2, 37.0] | 12.9 |

- **Delimiting and hierarchy alone:** their DE intervals include 0, so there is no clear effect.
- **Output validation does not change the model:** model-only DE is 0.0. It overrides the model's
  answer instead (45 overrides; 36 in the combined condition) [results/metrics/defences_overall.csv]. It only acts when the
  classifier is at least 0.90 sure, which is true for 397 of the 1,338 attack-pool emails
  [results/stage1_test_predictions.csv].
- **Combined:** the only condition where the model itself also improves. Model-only ASR is 37.8%,
  compared with 50.7% with no defence [results/metrics/defences_overall.csv].
- **By payload type** [results/metrics/defences_de_by_payload_type.csv]:
  - Output hijack drops from 54.7% to 8.0% with the combined defence (DE 46.7 [32.7, 58.3]).
  - Direct override drops from 97.3% to 54.7% with combined, and to 53.3% with validation.
    With hierarchy it stays at 97.3%.
  - **Hierarchy made fake verification worse:** 0.0% → 8.0%, DE −8.0 [−16.4, −1.5].
- **Clean emails:** no defence produced a false alarm on the 15 legitimate emails (0/15, CI up to
  20.4%). With hierarchy, 1 of 15 clean phishing emails was called legitimate
  [results/metrics/defences_clean.csv].

## 5. What this does and does not show

**It shows that:**
- For this model (llama3.2:3b), an explanation layer can be steered to call phishing safe while the
  classifier stays correct.
- The type and position of the injected text matter more than random variation.

**It does not show:**
- **Other models.** No other model was tested.
- **Other attacks.** Only 3 of the 20 payloads were measured.
- **A small effect of delimiting or hierarchy.** The 15-email defence sample is too small to rule
  one out.
- **Behaviour on real mail traffic.**
