# Data

Source: Kaggle dataset `naserabdullahalam/phishing-email-dataset` (text-only CSVs),
downloaded with `kagglehub` into `data/raw/kaggle_phishing/` (not committed to Git).
Built by `python -m src.prepare_data`. All numbers below are printed by that script.

## Sources

| Source | File | Raw rows | Kept rows | Label meaning (checked on samples) | Used for |
|---|---|---|---|---|---|
| Enron | Enron.csv | 29,767 | 29,411 | 0 = ham (Enron business email), 1 = spam | train |
| SpamAssassin | SpamAssasin.csv | 5,809 | 5,607 | 0 = ham (mailing lists / personal), 1 = spam | train |
| Ling | Ling.csv | 2,859 | 2,851 | 0 = ham (Linguist mailing list), 1 = spam | train |
| Nazario | Nazario.csv | 1,565 | 1,543 | all 1 = phishing (credential-theft emails) | test: `test_phishing` (main held-out phishing set, attack pool for Step 5) |
| Nigerian_Fraud | Nigerian_Fraud.csv | 3,332 | 3,220 | all 1 = advance-fee (419) fraud | test: `test_phishing_nigerian` (second held-out set, reported separately) |
| CEAS_08 | CEAS_08.csv | 39,154 | 35,780 | 0 = ham (developer mailing lists), 1 = spam | test: ham -> `test_ham`; spam -> `test_spam_transfer` (reported separately) |

`phishing_email.csv` from the same Kaggle dataset is **not used**: it is a merged,
preprocessed copy of the files above, so using it would leak test emails into training.

## Rows per split

| Source | Kind | Rows | Split | Eval set |
|---|---|---|---|---|
| Enron | ham | 15,498 | train | train |
| Enron | spam | 13,913 | train | train |
| SpamAssassin | ham | 4,090 | train | train |
| SpamAssassin | spam | 1,517 | train | train |
| Ling | ham | 2,396 | train | train |
| Ling | spam | 455 | train | train |
| Nazario | phishing | 1,543 | test | test_phishing |
| Nigerian_Fraud | phishing | 3,220 | test | test_phishing_nigerian |
| CEAS_08 | ham | 17,234 | test | test_ham |
| CEAS_08 | spam | 18,546 | test | test_spam_transfer |

- train.csv: 37,869 rows (21,984 legitimate, 15,885 malicious)
- test.csv: 40,543 rows (17,234 legitimate, 23,309 malicious)

## Columns

- `id`: source name + row number in the raw file
- `text`: subject + newline + body, HTML removed, whitespace collapsed, cut to 2000 characters
- `label`: 0 = legitimate, 1 = malicious
- `kind`: ham, spam or phishing
- `source`, `split` (train/test), `eval_set` (which test set the row belongs to)

## Cleaning

Removed in this order: near-empty text (under 20 letters/digits) and mailbox
placeholders; texts with conflicting labels (all copies); duplicates within a source;
duplicates across sources (the train copy is kept, so test stays unseen).
Duplicates are matched on lowercase letters and digits only, so formatting differences
such as `$ 39 . 85` vs `$39.85` do not hide a copy.

| Reason | Source | Removed |
|---|---|---|
| empty / near-empty text | CEAS_08 | 1 |
| empty / near-empty text | Enron | 30 |
| empty / near-empty text | Nigerian_Fraud | 1 |
| mailbox placeholder, not an email | Nazario | 2 |
| duplicate within same source | CEAS_08 | 3,373 |
| duplicate within same source | Enron | 326 |
| duplicate within same source | Ling | 8 |
| duplicate within same source | Nazario | 20 |
| duplicate within same source | Nigerian_Fraud | 109 |
| duplicate within same source | SpamAssassin | 27 |
| duplicate of an email in another source | Nigerian_Fraud | 2 |
| duplicate of an email in another source | SpamAssassin | 175 |

16,742 of 78,412 emails were cut at 2000 characters.
Check run by the script: no text (normalised or exact) appears in both train and test.

## Caveats

- **Spam was treated as malicious in training.** The training sets contain ham and spam
  but no phishing. The classifier therefore learns "spam-like vs legitimate", and the
  phishing test sets measure how well that transfers to phishing. Spam is not phishing;
  results on `test_phishing` and `test_spam_transfer` are reported separately.
- Enron and Ling were stored lowercased with spaces around punctuation by the dataset
  author; the other sources keep normal casing. Style differences between sources can
  help or hurt a classifier, which is why per-source accuracy is reported (Step 3).
- CEAS_08 ham is mostly technical mailing-list mail, so `test_ham` is not a sample of
  everyday personal or business email.
- Emails are capped at 2000 characters; injected payloads in Step 5 must sit before the cut.
