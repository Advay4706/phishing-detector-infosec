"""Build metrics tables, charts and a JSON summary for one experiment.

    python -m src.make_report results/logs/defences_pilot.csv --name defences_pilot
    python -m src.make_report results/logs/final.csv --name final

Writes:
    results/metrics/<name>_*.csv          every table (from src.metrics)
    results/figures/<name>_*.png          charts with 95% intervals
    results/<name>_summary.json           headline numbers + the file each came from
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.metrics import METRICS_DIR, load_runs, report  # noqa: E402

# Validated categorical palette (fixed order, one colour per defence; CVD checks pass)
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
DEF_COLORS = {"none": "#2a78d6", "delimiting": "#eb6834", "hierarchy": "#1baf7a",
              "validation": "#eda100", "combined": "#e87ba4"}
DEF_LABELS = {"none": "No defence", "delimiting": "Delimiting", "hierarchy": "Instr. hierarchy",
              "validation": "Output validation", "combined": "All combined"}
TYPE_LABELS = {"direct_override": "Direct override", "authority_impersonation": "Authority",
               "fake_verification": "Fake verification", "output_hijack": "Output hijack",
               "context_reframing": "Context reframing"}
LOC_LABELS = {"subject": "Subject", "body_start": "Body start", "body_end": "Body end",
              "signature": "Signature", "html_comment": "HTML comment"}


def style(ax, ylabel, ymax=100):
    ax.set_facecolor(SURFACE)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)


def grouped_bars(t, cat_col, cat_labels, title, path, value="ASR", n_col="n_asr"):
    """Bars per category, one bar per defence, Wilson CI whiskers, value printed on each bar."""
    defs = [d for d in config.DEFENCES if d in set(t.defence)]
    cats = [c for c in cat_labels if c in set(t[cat_col])]
    width = 0.8 / len(defs)
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(cats) * max(1, len(defs) / 2)), 4.4),
                           facecolor=SURFACE)
    for i, d in enumerate(defs):
        sub = t[t.defence == d].set_index(cat_col).reindex(cats)
        x = np.arange(len(cats)) + (i - (len(defs) - 1) / 2) * width
        y = sub[value].to_numpy() * 100
        lo, hi = sub[f"{value}_low"].to_numpy() * 100, sub[f"{value}_high"].to_numpy() * 100
        ax.bar(x, y, width, color=DEF_COLORS[d], edgecolor=SURFACE, linewidth=2,
               label=DEF_LABELS[d])
        ax.errorbar(x, y, yerr=np.clip([y - lo, hi - y], 0, None), fmt="none", ecolor=INK,
                    elinewidth=1, capsize=2)
        for xi, yi, hii in zip(x, y, hi):
            if not np.isnan(yi):
                ax.text(xi, hii + 1.5, f"{yi:.0f}", ha="center", va="bottom", fontsize=7,
                        color=INK)
    ax.set_xticks(np.arange(len(cats)), [cat_labels[c] for c in cats], color=INK)
    style(ax, f"{value} (%) with 95% Wilson CI", ymax=110)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, ncol=min(5, len(defs)), loc="upper left",
              bbox_to_anchor=(0, -0.1), labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def single_bars(labels, vals, lo, hi, colors, title, ylabel, path, ymin=0, ymax=100):
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=SURFACE)
    x = np.arange(len(labels))
    ax.bar(x, vals, 0.6, color=colors, edgecolor=SURFACE, linewidth=2)
    ax.errorbar(x, vals, yerr=np.clip([vals - lo, hi - vals], 0, None), fmt="none", ecolor=INK,
                elinewidth=1, capsize=3)
    for xi, v, h in zip(x, vals, hi):
        ax.text(xi, h + (ymax - ymin) * 0.015, f"{v:.1f}", ha="center", va="bottom",
                fontsize=8, color=INK)
    ax.set_xticks(x, labels, color=INK)
    style(ax, ylabel, ymax=ymax)
    ax.set_ylim(ymin, ymax)
    if ymin < 0:
        ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def charts(name, tables):
    fig_dir = config.FIGURES_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []
    t = tables["by_payload_type"].reset_index()
    p = fig_dir / f"{name}_asr_by_payload_type.png"
    grouped_bars(t, "payload_type", TYPE_LABELS, "Attack success rate by payload type", p)
    made.append(p)
    t = tables["by_location"].reset_index()
    p = fig_dir / f"{name}_asr_by_location.png"
    grouped_bars(t, "location", LOC_LABELS, "Attack success rate by insertion location", p)
    made.append(p)

    o = tables["overall"].reset_index()
    defs = [d for d in config.DEFENCES if d in set(o.defence)]
    o = o.set_index("defence").reindex(defs)
    labels, colors = [DEF_LABELS[d] for d in defs], [DEF_COLORS[d] for d in defs]
    p = fig_dir / f"{name}_asr_by_defence.png"
    single_bars(labels, o.ASR.to_numpy() * 100, o.ASR_low.to_numpy() * 100,
                o.ASR_high.to_numpy() * 100, colors, "Overall attack success rate by defence",
                "ASR (%) with 95% Wilson CI", p, ymax=110)
    made.append(p)
    for m, title in (("INV", "Invalid-JSON rate by defence"),
                     ("FLIP", "Classifier-flip rate by defence")):
        p = fig_dir / f"{name}_{m.lower()}_by_defence.png"
        hi = max(5.0, float(np.nanmax(o[f"{m}_high"].to_numpy() * 100)) * 1.3)
        single_bars(labels, o[m].to_numpy() * 100, o[f"{m}_low"].to_numpy() * 100,
                    o[f"{m}_high"].to_numpy() * 100, colors, title, f"{m} (%) with 95% CI", p,
                    ymax=hi)
        made.append(p)

    if "de_overall" in tables:
        de = tables["de_overall"].set_index("defence")
        dd = [d for d in defs if d in de.index]
        de = de.reindex(dd)
        lo_lim = min(-10.0, float(np.nanmin(de.DE_low)) * 100 - 5)
        p = fig_dir / f"{name}_de_by_defence.png"
        single_bars([DEF_LABELS[d] for d in dd], de.DE.to_numpy() * 100,
                    de.DE_low.to_numpy() * 100, de.DE_high.to_numpy() * 100,
                    [DEF_COLORS[d] for d in dd],
                    "Defence effectiveness (baseline ASR minus defended ASR)",
                    "DE (percentage points), 95% Newcombe CI", p, ymin=lo_lim, ymax=100)
        made.append(p)

    c = tables["clean"]
    for kind, m_title in (("phishing", "Clean phishing: explainer agrees with classifier"),
                          ("ham", "Clean legitimate email: explainer agrees with classifier")):
        ck = c[c.email_kind == kind].set_index("defence").reindex(
            [d for d in defs if d in set(c[c.email_kind == kind].defence)])
        if len(ck):
            p = fig_dir / f"{name}_clean_{kind}_lar.png"
            single_bars([DEF_LABELS[d] for d in ck.index], ck.LAR.to_numpy() * 100,
                        ck.LAR_low.to_numpy() * 100, ck.LAR_high.to_numpy() * 100,
                        [DEF_COLORS[d] for d in ck.index], m_title,
                        "LAR (%) with 95% Wilson CI", p, ymax=110)
            made.append(p)
    return made


def summary_json(name, logs, df, tables, figures):
    def f(x):
        return None if pd.isna(x) else round(float(x), 4)

    o = tables["overall"]
    out = {
        "name": name, "log_files": [str(p) for p in logs],
        "model": sorted(df.model.unique().tolist()), "seeds": sorted(int(s) for s in df.seed.unique()),
        "defences": [d for d in config.DEFENCES if d in set(df.defence)],
        "n_phishing_emails": int(df[df.email_kind == "phishing"].email_id.nunique()),
        "n_ham_emails": int(df[df.email_kind == "ham"].email_id.nunique()),
        "n_payloads": int(df[df.attacked].payload_id.nunique()), "n_runs": int(len(df)),
        "definitions": "see src/metrics.py docstring",
        "per_defence": {
            d: {"ASR": f(r.ASR), "ASR_ci": [f(r.ASR_low), f(r.ASR_high)],
                "ASR_raw": f(r.ASR_raw), "n_asr": int(r.n_asr), "LAR": f(r.LAR),
                "LAR_ci": [f(r.LAR_low), f(r.LAR_high)], "INV": f(r.INV), "FLIP": f(r.FLIP),
                "validator_interventions": int(r.validator_interventions),
                "source": f"results/metrics/{name}_overall.csv"}
            for d, r in o.iterrows()},
        "clean": {f"{r.defence}/{r.email_kind}": {
            "LAR": f(r.LAR), "LAR_ci": [f(r.LAR_low), f(r.LAR_high)],
            "wrong_rate": f(r.wrong_rate), "wrong_ci": [f(r.wrong_low), f(r.wrong_high)],
            "n": int(r.n_runs), "source": f"results/metrics/{name}_clean.csv"}
            for r in tables["clean"].itertuples()},
        "figures": [p.relative_to(config.ROOT).as_posix() for p in figures],
    }
    if "de_overall" in tables:
        out["defence_effectiveness"] = {
            r.defence: {"DE": f(r.DE), "DE_ci": [f(r.DE_low), f(r.DE_high)],
                        "DE_raw": f(r.DE_raw), "ASR_base": f(r.ASR_base), "ASR_def": f(r.ASR_def),
                        "source": f"results/metrics/{name}_de_overall.csv"}
            for r in tables["de_overall"].itertuples()}
    path = config.RESULTS_DIR / f"{name}_summary.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    df = load_runs(*args.logs)
    tables = report(df, args.name)
    figs = charts(args.name, tables)
    path = summary_json(args.name, args.logs, df, tables, figs)
    print(f"\nfigures: {[p.name for p in figs]}\nsummary: {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
