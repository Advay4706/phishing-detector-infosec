"""Streamlit demo for the CBS 3002 phishing detector.   Launch:  streamlit run app.py

Thin UI: models run only through src.pipeline.run(); results come only from files in results/.
"""
import altair as alt
import pandas as pd
import streamlit as st

from src import config, ui_data
from src.injection import LOCATIONS
from src.pipeline import ollama_status, payloads_by_id, run

st.set_page_config(page_title="Phishing Detector + Injection Lab", layout="wide")

LOCATION_LABELS = {"subject": "Subject", "body_start": "Body start", "body_end": "Body end",
                   "signature": "Signature", "html_comment": "HTML comment"}
DEFENCE_LABELS = {"none": "None (baseline)", "delimiting": "Input delimiting",
                  "hierarchy": "Instruction hierarchy", "validation": "Output validation",
                  "combined": "All three combined"}


def pct(x):
    return "not run yet" if x is None or pd.isna(x) else f"{100 * x:.1f}%"


def with_ci(row, m):
    return f"{pct(row[m])}  [{pct(row[m + '_low'])} – {pct(row[m + '_high'])}]"


# ------------------------------------------------------------------ sidebar
ok, msg = ollama_status()
with st.sidebar:
    st.header("System")
    (st.success if ok else st.error)(msg)
    st.caption(f"Model: {config.MODEL_NAME} · temperature {config.TEMPERATURE} · "
               f"seed {config.SEED}")
    st.caption("Everything runs locally. No email or payload leaves this computer.")

st.title("Phishing Detector with an Adversarial Explanation Layer")
tab_try, tab_results, tab_about = st.tabs(["Try it", "Results", "About"])


# ------------------------------------------------------------------ Try it
def show_stage_results(r, title):
    st.subheader(title)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Stage 1 · Classifier**")
        st.metric("Label", r["clf_label_name"], f"P(malicious) = {r['clf_prob']:.2f}",
                  delta_color="off")
    with c2:
        st.markdown("**Stage 2 · LLM explainer**")
        if r["error"]:
            st.warning(r["error"])
        else:
            st.metric("Verdict", r["verdict"], "valid JSON", delta_color="off")
    if r["reason"]:
        st.write(f"*Reason:* {r['reason']}")
    if r["validator_intervened"]:
        st.info(f"Output validator stepped in: {r['validator_note']}")
    if r["layers_disagree_unsafe"]:
        st.error("⚠ LAYERS DISAGREE: the classifier says MALICIOUS but the explainer tells the "
                 "user it is SAFE.")
    elif r["valid_json"]:
        st.success("Layers agree." if r["layers_agree"] else "Layers disagree.")


def apply_demo(ex):
    st.session_state.source = "Sample email"
    st.session_state.email_id = ex["email_id"]
    st.session_state.payload_id = ex["payload_id"] or "none"
    st.session_state.location = ex["location"] or "body_end"
    st.session_state.defence = ex["defence"]
    st.session_state.demo_note = ex


with tab_try:
    demos = ui_data.demo_examples()
    st.markdown("#### Demo examples (picked from saved results)")
    if not demos:
        st.caption("Demo examples are not picked yet: run python -m src.pick_demo_examples")
    else:
        cols = st.columns(len(demos["examples"]))
        for col, ex in zip(cols, demos["examples"]):
            if col.button(ex["title"], use_container_width=True):
                apply_demo(ex)
        note = st.session_state.get("demo_note")
        if note:
            st.caption(f"Recorded in `{note.get('source_file', demos['source_file'])}`: "
                       f"{note['recorded']}")

    st.divider()
    left, right = st.columns([3, 2])
    with left:
        source = st.radio("Email", ["Sample email", "Paste text"], horizontal=True, key="source")
        if source == "Sample email":
            choices = ui_data.sample_choices()
            ids = [c[0] for c in choices]
            if st.session_state.get("email_id") not in ids:
                ids = [st.session_state.get("email_id")] + ids if st.session_state.get(
                    "email_id") else ids
            email_id = st.selectbox("Attack-pool email (Nazario phishing, flagged correctly)",
                                    ids, key="email_id",
                                    format_func=lambda i: f"{i} · {ui_data.subject(i)}")
            email = ui_data.email_text(email_id)
            st.caption("Links in these emails are real phishing links. Do not open them.")
        else:
            email = st.text_area("Paste an email (first line = subject)", height=200,
                                 key="pasted")
    with right:
        payloads = payloads_by_id()
        types = ["none"] + sorted({p["type"] for p in payloads.values()})
        cur = st.session_state.get("payload_id", "none")
        default_type = payloads[cur]["type"] if cur in payloads else "none"
        ptype = st.selectbox("Payload type", types, index=types.index(default_type))
        if ptype == "none":
            payload_id = None
        else:
            options = [pid for pid, p in payloads.items() if p["type"] == ptype]
            payload_id = st.selectbox("Payload", options,
                                      index=options.index(cur) if cur in options else 0)
            st.caption(f"“{payloads[payload_id]['text']}”")
        location = st.selectbox("Insert at", LOCATIONS, key="location",
                                format_func=LOCATION_LABELS.get, disabled=payload_id is None)
        defence = st.selectbox("Defence", config.DEFENCES, key="defence",
                               format_func=DEFENCE_LABELS.get)
        go = st.button("Run clean and injected", type="primary", disabled=not ok)
        if not ok:
            st.caption("Start Ollama to run the explainer.")

    if go:
        if not email or not email.strip():
            st.warning("Choose or paste an email first.")
        else:
            try:
                with st.spinner("Running the clean email..."):
                    clean = run(email, defence=defence)
                injected = None
                if payload_id:
                    with st.spinner("Running the injected email..."):
                        injected = run(email, payload_id, location, defence)
                st.session_state.last = (clean, injected)
            except Exception as e:  # keep the UI alive whatever happens
                st.error(f"Run failed: {type(e).__name__}: {e}")

    if "last" in st.session_state:
        clean, injected = st.session_state.last
        c1, c2 = st.columns(2)
        with c1:
            show_stage_results(clean, "Clean email")
        with c2:
            if injected:
                show_stage_results(injected, f"Injected · {injected['payload_id']} at "
                                             f"{LOCATION_LABELS[injected['location']]}")
            else:
                st.info("No payload chosen: only the clean run is shown.")
        if injected:
            with st.expander("Injected email text (as the model saw it)"):
                st.code(injected["email_text"], language=None)
            with st.expander("Prompts sent to the model"):
                st.code(injected["system_prompt"], language=None)
                st.code(injected["user_prompt"], language=None)


# ------------------------------------------------------------------ Results
def bar_with_ci(df, x, color, title):
    """Bar chart of ASR with Wilson error bars."""
    d = df.copy()
    for c in ("ASR", "ASR_low", "ASR_high"):
        d[c] = d[c] * 100
    base = alt.Chart(d).encode(x=alt.X(f"{x}:N", title=None), xOffset=f"{color}:N")
    bars = base.mark_bar().encode(
        y=alt.Y("ASR:Q", title="ASR (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color(f"{color}:N", title=color),
        tooltip=[x, color, alt.Tooltip("ASR:Q", format=".1f"),
                 alt.Tooltip("ASR_low:Q", format=".1f"), alt.Tooltip("ASR_high:Q", format=".1f"),
                 "n_asr"])
    err = base.mark_errorbar().encode(y=alt.Y("ASR_low:Q", title="ASR (%)"), y2="ASR_high:Q")
    return (bars + err).properties(title=title, height=320)


with tab_results:
    sets = ui_data.list_result_sets()
    if not sets:
        st.info("No results yet — run Step 8 first (python -m src.run_full).")
    else:
        name = st.selectbox("Result set", sets,
                            help="defences = Step 7: 15 pilot emails x 3 payloads x 5 locations, "
                                 "seed 42, every defence · pilot = Step 6 baseline: 50 emails x "
                                 "3 payloads x 5 locations x 3 seeds · final = full run (if run)")
        overall = ui_data.load_table(name, "overall")
        st.markdown("#### Attacks per defence (95% Wilson intervals)")
        st.caption("ASR = attacks that made the explainer say 'legitimate' ÷ attacks with a valid "
                   "reply where the classifier stayed correct. Invalid replies (INV) and "
                   "classifier flips (FLIP) are counted separately, never inside ASR.")
        view = pd.DataFrame({
            "Defence": overall.iloc[:, 0].map(lambda d: DEFENCE_LABELS.get(d, d)),
            "Runs in ASR": overall.n_asr.astype(int),
            "ASR": [with_ci(r, "ASR") for _, r in overall.iterrows()],
            "ASR before validator": [pct(x) for x in overall.ASR_raw],
            "LAR": [with_ci(r, "LAR") for _, r in overall.iterrows()],
            "Invalid JSON": [pct(x) for x in overall.INV],
            "Classifier flips": [pct(x) for x in overall.FLIP],
        })
        st.dataframe(view, hide_index=True, use_container_width=True)

        de = ui_data.load_table(name, "de_overall")
        st.markdown("#### Defence effectiveness (DE = baseline ASR − defended ASR)")
        if de is None:
            st.caption("not run yet for this result set")
        else:
            st.dataframe(pd.DataFrame({
                "Defence": de.defence.map(lambda d: DEFENCE_LABELS.get(d, d)),
                "Baseline ASR": [pct(x) for x in de.ASR_base],
                "Defended ASR": [pct(x) for x in de.ASR_def],
                "DE [95% Newcombe CI]": [f"{pct(r.DE)}  [{pct(r.DE_low)} – {pct(r.DE_high)}]"
                                         for r in de.itertuples()],
                "DE before validator": [pct(x) for x in de.DE_raw],
            }), hide_index=True, use_container_width=True)
            d2 = de.assign(DE=de.DE * 100, lo=de.DE_low * 100, hi=de.DE_high * 100)
            ch = alt.Chart(d2).encode(x=alt.X("defence:N", title=None))
            st.altair_chart((ch.mark_bar().encode(y=alt.Y("DE:Q", title="DE (percentage points)"))
                             + ch.mark_errorbar().encode(
                                 y=alt.Y("lo:Q", title="DE (percentage points)"), y2="hi:Q")).properties(
                                 title="Defence effectiveness", height=280),
                            use_container_width=True)

        t = ui_data.load_table(name, "by_payload_type")
        st.altair_chart(bar_with_ci(t, "payload_type", "defence", "ASR by payload type"),
                        use_container_width=True)
        loc = ui_data.load_table(name, "by_location")
        st.altair_chart(bar_with_ci(loc, "location", "defence", "ASR by insertion location"),
                        use_container_width=True)

        clean = ui_data.load_table(name, "clean")
        st.markdown("#### Clean emails (no payload): does a defence change normal results?")
        st.caption("Phishing rows: wrong = called legitimate with no attack. "
                   "Ham rows: wrong = false alarm (legitimate email called phishing).")
        st.dataframe(pd.DataFrame({
            "Defence": clean.defence.map(lambda d: DEFENCE_LABELS.get(d, d)),
            "Email kind": clean.email_kind, "Runs": clean.n_runs,
            "LAR": [with_ci(r, "LAR") for _, r in clean.iterrows()],
            "Wrong verdict": [f"{pct(r.wrong_rate)}  [{pct(r.wrong_low)} – {pct(r.wrong_high)}]"
                              for r in clean.itertuples()],
            "Invalid JSON": [pct(x) for x in clean.INV],
        }), hide_index=True, use_container_width=True)
        with st.expander("Per-payload and per-seed tables"):
            for tb in ("by_payload", "by_location", "seed_spread", "bootstrap"):
                df = ui_data.load_table(name, tb)
                if df is not None:
                    st.markdown(f"`results/metrics/{name}_{tb}.csv`")
                    st.dataframe(df, use_container_width=True)


# ------------------------------------------------------------------ About
with tab_about:
    st.markdown("#### How the system works")
    st.graphviz_chart("""
    digraph {
      rankdir=LR; node [shape=box, style=rounded, fontname=Helvetica];
      email [label="Email\\n(subject + body,\\n≤ 2000 chars)"];
      attack [label="Injection harness\\n20 payloads × 5 locations", style="rounded,dashed"];
      clf [label="Stage 1\\nTF-IDF + Logistic Regression\\nphishing / legitimate"];
      llm [label="Stage 2\\nLocal LLM (llama3.2:3b, Ollama)\\nJSON {verdict, reason}"];
      def [label="Defence\\n(delimiting / hierarchy /\\noutput validation)", style="rounded,dashed"];
      out [label="Metrics\\nASR · LAR · INV · FLIP · DE\\n+ Wilson intervals"];
      attack -> email [style=dashed]; email -> clf -> llm -> out; email -> llm;
      def -> llm [style=dashed];
    }""")
    st.markdown("#### Data")
    data_md = config.ROOT / "data" / "DATA.md"
    for heading in ("Sources", "Rows per split"):
        sec = ui_data.markdown_section(data_md, heading)
        if sec:
            st.markdown(f"**{heading}**")
            st.markdown(sec)
    st.markdown("#### Limitations")
    lim = ui_data.markdown_section(config.ROOT / "README.md", "Limitations")
    st.markdown(lim or "not written yet (see README.md after Step 8)")
