# Step 8 full experiment. Resumable: re-run this script to continue after any interruption.
# Size comes from src/config.py (FULL_N_EMAILS, FULL_HAM_N_EMAILS, SEEDS, DEFENCES).
$py = ".\.venv\Scripts\python.exe"
$n = & $py -c "from src import config; print(config.FULL_N_EMAILS)"
$ham = & $py -c "from src import config; print(config.FULL_HAM_N_EMAILS)"
$seeds = & $py -c "from src import config; print(','.join(map(str, config.SEEDS)))"
$defs = & $py -c "from src import config; print(','.join(config.DEFENCES))"
& $py -u -m src.harness --n-emails $n --payloads all --n-ham $ham --seeds $seeds --defences $defs --out results/logs/final.csv --workers 4
if ($LASTEXITCODE -eq 0) { & $py -m src.make_report results/logs/final.csv --name final }
