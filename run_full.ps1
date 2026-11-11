# Step 8 full experiment. Resumable: re-run this script to continue after any interruption.
# Size comes from src/config.py (FULL_N_EMAILS, PILOT_N_EMAILS, HAM_N_EMAILS, SEEDS, DEFENCES).
$py = ".\.venv\Scripts\python.exe"
$n = & $py -c "from src import config; print(config.FULL_N_EMAILS)"
$seeds = & $py -c "from src import config; print(','.join(map(str, config.SEEDS)))"
$defs = & $py -c "from src import config; print(','.join(config.DEFENCES))"
$pilot = & $py -c "from src import config; print(config.PILOT_N_EMAILS)"
$pilotHam = & $py -c "from src import config; print(config.HAM_N_EMAILS)"
# The emails are the first $n of the same 50 pilot emails (and ham) that version 1 used.
# "validation" has the same prompt as "none", so it is derived from the saved baseline replies.
$run = ($defs.Split(",") | Where-Object { $_ -ne "validation" }) -join ","
& $py -u -m src.harness --n-emails $pilot --n-ham $pilotHam --first $n --payloads all --seeds $seeds --defences $run --out results/logs/full_8b.csv
if ($LASTEXITCODE -eq 0) { & $py -m src.derive_validation results/logs/full_8b.csv }
if ($LASTEXITCODE -eq 0) { & $py -m src.make_report results/logs/full_8b.csv --name full_8b }
