# app.py — Performance Affinity Analysis (Enterprise, Py 3.11+)
# -----------------------------------------------------------------
# A production-grade Streamlit app for Contact Center Performance Affinity Analysis
# (Market Basket Analysis for KPI co-occurrence) with executive-ready UI/UX.
#
# ✅ Professional light theme + corporate palette
# ✅ Tabbed navigation (Data ▸ Setup ▸ Results ▸ Actions ▸ Export)
# ✅ Executive Summary with impact quantification & confidence
# ✅ Presets, progressive disclosure, context-aware help
# ✅ Drill-down to agent-level, cross-filter controls
# ✅ Shareable URLs via query params; benchmark comparisons
# ✅ Downloadable outputs incl. print-optimized executive report (HTML)
# ✅ Cached/stateful; Python 3.11+

from __future__ import annotations
import io
import json
import textwrap
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# ---------------------------------
# Page config & color system
# ---------------------------------
st.set_page_config(
    page_title="Performance Affinity Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "primary": "#1e40af",   # Navy
    "accent": "#3b82f6",    # Bright Blue
    "bg": "#ffffff",
    "muted": "#64748b",     # slate-500
    "pos": "#10b981",       # green
    "warn": "#f59e0b",      # amber
    "neg": "#ef4444",       # red
    # chart set
    "c1": "#1e40af",
    "c2": "#3b82f6",
    "c3": "#16a34a",
    "c4": "#ef4444",
}

# Professional light CSS + print styles
CUSTOM_CSS = f"""
<style>
:root {{
  --primary: {PALETTE['primary']};
  --accent: {PALETTE['accent']};
  --muted: {PALETTE['muted']};
  --pos: {PALETTE['pos']};
  --warn: {PALETTE['warn']};
  --neg: {PALETTE['neg']};
}}
.stApp {{ background: #f8fafc; color: #0f172a; }}
h1,h2,h3 {{ color: #0f172a; }}
.card {{ background:#ffffff; border:1px solid #e5e7eb; border-radius:14px; padding:16px 18px; box-shadow:0 4px 12px rgba(15,23,42,0.06); }}
.helper {{ color: var(--muted); font-size:0.92rem; }}
.pill {{ display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid #e2e8f0; background:#f8fafc; margin-right:8px; color:#0f172a; }}
.dataframe tbody tr:hover {{ background:#f1f5f9; }}
a, .stDownloadButton button {{ color:#fff !important; background: var(--primary) !important; border-color: var(--primary) !important; }}
.stButton button[kind="secondary"] {{ background:#e2e8f0; color:#0f172a; }}
/* Print styles for Executive Report */
@media print {{
  .stApp, body {{ background:#fff !important; }}
  .stMainBlockContainer {{ padding:0 !important; }}
  .no-print {{ display:none !important; }}
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------
# Helpers: data generation, binning, basket, mining
# ---------------------------------

def generate_sample_contact_center_data(n_agents: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    agents = [f"AG{1000+i}" for i in range(n_agents)]

    call_types = rng.choice(["Billing", "TechSupport", "Retention", "Sales", "General"], size=n_agents, p=[0.25,0.25,0.15,0.2,0.15])
    aht = rng.normal(loc=420, scale=110, size=n_agents).clip(120, 1200)        # seconds
    acw = rng.normal(loc=110, scale=40, size=n_agents).clip(20, 420)           # seconds
    hold = rng.normal(loc=65, scale=35, size=n_agents).clip(0, 600)            # seconds
    csat = rng.normal(loc=4.0, scale=0.7, size=n_agents).clip(1.0, 5.0)        # 1-5
    fcr = rng.choice([0,1], size=n_agents, p=[0.35, 0.65])
    adherence = rng.normal(loc=0.91, scale=0.05, size=n_agents).clip(0.6, 1.0) # 0-1
    qa_compliance = rng.normal(loc=0.88, scale=0.08, size=n_agents).clip(0.4, 1.0)
    escalations = rng.poisson(lam=1.3, size=n_agents)
    repeat_contacts_7d = rng.poisson(lam=0.9, size=n_agents)
    absenteeism = rng.normal(loc=0.04, scale=0.03, size=n_agents).clip(0.0, 0.25)

    df = pd.DataFrame({
        "AgentID": agents,
        "CallType": call_types,
        "AHT_sec": aht.round(0).astype(int),
        "ACW_sec": acw.round(0).astype(int),
        "Hold_sec": hold.round(0).astype(int),
        "CSAT": csat.round(2),
        "FCR": fcr.astype(int),
        "Adherence": adherence.round(3),
        "QA_Compliance": qa_compliance.round(3),
        "Escalations": escalations,
        "RepeatContacts7d": repeat_contacts_7d,
        "Absenteeism": absenteeism.round(3),
    })

    # Inject correlations
    mask_hi_acw = df["ACW_sec"] > df["ACW_sec"].quantile(0.75)
    df.loc[mask_hi_acw, "CSAT"] = (df.loc[mask_hi_acw, "CSAT"] - np.abs(np.random.normal(0.35, 0.2, mask_hi_acw.sum()))).clip(1.0, 5.0)

    mask_hi_hold = df["Hold_sec"] > df["Hold_sec"].quantile(0.75)
    df.loc[mask_hi_hold, "RepeatContacts7d"] += np.random.poisson(0.8, mask_hi_hold.sum())

    mask_low_qa = df["QA_Compliance"] < df["QA_Compliance"].quantile(0.25)
    df.loc[mask_low_qa, "Escalations"] += np.random.poisson(1.2, mask_low_qa.sum())

    return df


def bin_kpis(df: pd.DataFrame, strategy: str = "quantile") -> pd.DataFrame:
    binned = df.copy()
    q_features = ["AHT_sec", "ACW_sec", "Hold_sec", "CSAT", "Adherence", "QA_Compliance", "Escalations", "RepeatContacts7d", "Absenteeism"]
    for col in q_features:
        if strategy == "quantile":
            try:
                q = pd.qcut(binned[col], 3, labels=["Low", "Med", "High"])
            except ValueError:
                q = pd.qcut(binned[col], 2, labels=["Low", "High"])            
            binned[col+"_BIN"] = q.astype(str)
        else:
            med = binned[col].median()
            binned[col+"_BIN"] = np.where(binned[col] >= med, "High", "Low")
    if "FCR" in binned.columns:
        binned["FCR_BIN"] = np.where(binned["FCR"] == 1, "FCR_Yes", "FCR_No")
    return binned


def to_basket(df_binned: pd.DataFrame) -> pd.DataFrame:
    item_cols = [c for c in df_binned.columns if c.endswith("_BIN")]
    baskets = []
    for col in item_cols:
        dummies = pd.get_dummies(df_binned[col], prefix=col.replace("_BIN", ""), dtype=int)
        baskets.append(dummies)
    basket_df = pd.concat(baskets, axis=1)
    if "CallType" in df_binned.columns:
        basket_df = pd.concat([basket_df, pd.get_dummies(df_binned["CallType"], prefix="CallType", dtype=int)], axis=1)
    return basket_df


def mine_rules(
    basket_df: pd.DataFrame,
    algo: str = "apriori",
    min_support: float = 0.08,
    metric: str = "lift",
    min_threshold: float = 1.05,
    max_len: int | None = 3,
) -> pd.DataFrame:
    if algo == "fpgrowth":
        freq = fpgrowth(basket_df, min_support=min_support, use_colnames=True, max_len=max_len)
    else:
        freq = apriori(basket_df, min_support=min_support, use_colnames=True, max_len=max_len)
    if freq.empty:
        return pd.DataFrame()
    rules = association_rules(freq, metric=metric, min_threshold=min_threshold)
    rules = rules.sort_values("lift", ascending=False)

    def set_to_str(s: Any) -> str:
        return ", ".join(sorted(list(s))) if isinstance(s, (set, frozenset)) else str(s)

    rules["antecedents_str"] = rules["antecedents"].apply(set_to_str)
    rules["consequents_str"] = rules["consequents"].apply(set_to_str)

    cols = ["antecedents_str", "consequents_str", "support", "confidence", "lift", "leverage", "conviction"]
    rules = rules[cols]
    rules[["support","confidence","lift","leverage","conviction"]] = rules[["support","confidence","lift","leverage","conviction"]].round(4)

    # Confidence scoring (enterprise hint): combine confidence, lift, conviction, support
    # Score 0-100
    rules["confidence_score"] = (
        (rules["confidence"]*50) + ((rules["lift"]-1).clip(lower=0)*30) + (rules["conviction"].clip(0,5)/5*10) + (rules["support"]*10)
    ).clip(0,100).round(1)

    return rules.reset_index(drop=True)


def build_pyvis_network(rules: pd.DataFrame, height: str = "560px") -> str:
    net = Network(height=height, width="100%", bgcolor="#ffffff", font_color="#0f172a")
    net.barnes_hut(gravity=-12000, central_gravity=0.3, spring_length=120, spring_strength=0.02, damping=0.9)

    nodes = set()
    for _, r in rules.iterrows():
        a = r["antecedents_str"].split(", ")
        c = r["consequents_str"].split(", ")
        nodes.update(a + c)

    for node in nodes:
        color = PALETTE["c2"] if "High" in node else (PALETTE["c3"] if "FCR_Yes" in node else (PALETTE["c1"] if "CallType_" in node else PALETTE["c4"]))
        net.add_node(node, label=node, color=color)

    for _, r in rules.iterrows():
        for a in r["antecedents_str"].split(", "):
            for c in r["consequents_str"].split(", "):
                title = f"lift: {r['lift']}, conf: {r['confidence']}, support: {r['support']}"
                net.add_edge(a, c, title=title, value=float(r["lift"]))

    return net.generate_html()

# ---------------------------------
# Session state
# ---------------------------------
for key, default in {
    "data": None,
    "binned": None,
    "basket": None,
    "rules": None,
    "network_html": None,
    "selected_rule": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------
# Query params (shareable URLs)
# ---------------------------------
qp = st.experimental_get_query_params()

def set_query_params(**kwargs):
    st.experimental_set_query_params(**{**qp, **kwargs})

# ---------------------------------
# Sidebar: Presets, help, benchmarks
# ---------------------------------
with st.sidebar:
    st.header("Settings")

    # Presets
    st.markdown("**Presets**")
    preset = st.radio(
        label="Choose scenario",
        options=["Balanced", "Exploratory (broader)", "Strict (high confidence)"],
        index=int(qp.get("preset", [0])[0]) if qp.get("preset") else 0,
        help="One-click parameter presets"
    )

    # Map presets to params
    if preset == "Balanced":
        default_params = dict(algo="apriori", min_support=0.08, metric="lift", min_threshold=1.1, max_len=3)
    elif preset == "Exploratory (broader)":
        default_params = dict(algo="fpgrowth", min_support=0.05, metric="lift", min_threshold=1.0, max_len=4)
    else:  # Strict
        default_params = dict(algo="apriori", min_support=0.12, metric="confidence", min_threshold=0.7, max_len=3)

    # Context help
    st.markdown("""
    <div class='helper'>Select a preset to quickly tune support/thresholds. Use <b>Exploratory</b> to discover more patterns, then switch to <b>Strict</b> for executive-ready insights.</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Benchmarks (editable)**")
    bm_csat = st.number_input("Industry CSAT (1-5)", 1.0, 5.0, float(qp.get("bm_csat", [4.2])[0]))
    bm_fcr = st.number_input("Industry FCR (0-1)", 0.0, 1.0, float(qp.get("bm_fcr", [0.72])[0]))
    bm_aht = st.number_input("Industry AHT (sec)", 60, 1800, int(qp.get("bm_aht", [420])[0]))

# ---------------------------------
# Header & Executive Summary card placeholder
# ---------------------------------
st.title("Performance Affinity Analysis")

summary_container = st.container()

# ---------------------------------
# Tabs: Data ▸ Setup ▸ Results ▸ Actions ▸ Export
# ---------------------------------
TAB_DATA, TAB_SETUP, TAB_RESULTS, TAB_ACTIONS, TAB_EXPORT = st.tabs([
    "Data", "Setup", "Results", "Actions", "Export"
])

# ---------------- Data Tab -----------------
with TAB_DATA:
    st.subheader("Data Ingestion")

    col_a, col_b = st.columns([2,1])
    with col_a:
        uploaded = st.file_uploader("Upload KPI CSV", type=["csv"], accept_multiple_files=False)
        st.markdown("""
        <div class='helper'>Expected columns: AgentID, CallType, AHT_sec, ACW_sec, Hold_sec, CSAT, FCR, Adherence, QA_Compliance, Escalations, RepeatContacts7d, Absenteeism.</div>
        """, unsafe_allow_html=True)
    with col_b:
        if st.button("Generate Sample Dataset", use_container_width=True):
            st.session_state.data = generate_sample_contact_center_data()
            st.success("Sample data ready.")

    if uploaded is not None:
        try:
            st.session_state.data = pd.read_csv(uploaded)
            st.success("Uploaded data loaded.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    if st.session_state.data is None:
        st.info("Upload a CSV or generate a sample to proceed.")
    else:
        st.dataframe(st.session_state.data.head(25), use_container_width=True)
        # Quick KPI vs benchmarks
        df = st.session_state.data
        kcol1, kcol2, kcol3, kcol4 = st.columns(4)
        with kcol1: st.metric("Avg CSAT", round(df["CSAT"].mean(),2), delta=round(df["CSAT"].mean()-bm_csat,2))
        with kcol2: st.metric("FCR", round(df["FCR"].mean(),2), delta=round(df["FCR"].mean()-bm_fcr,2))
        with kcol3: st.metric("AHT (sec)", int(df["AHT_sec"].mean()), delta=int(df["AHT_sec"].mean()-bm_aht))
        with kcol4: st.metric("Agents", df["AgentID"].nunique())

# ---------------- Setup Tab -----------------
with TAB_SETUP:
    st.subheader("Preprocessing & Mining Parameters")

    col1, col2, col3, col4, col5 = st.columns(5)
    algo = col1.selectbox("Algorithm", ["apriori", "fpgrowth"], index=["apriori","fpgrowth"].index(default_params["algo"]))
    min_support = col2.slider("Min Support", 0.01, 0.3, float(default_params["min_support"]), 0.01)
    metric = col3.selectbox("Rule Metric", ["lift", "confidence", "support"], index=["lift","confidence","support"].index(default_params["metric"]))
    min_threshold = col4.slider("Min Threshold", 0.5, 3.0, float(default_params["min_threshold"]), 0.05)
    max_len = col5.slider("Max Items", 2, 5, int(default_params["max_len"]))

    # Progressive disclosure
    with st.expander("Advanced: Binning Strategy & Itemization"):
        bin_strategy = st.radio("Binning", ["quantile", "median"], horizontal=True)
        st.markdown("<div class='helper'>Use quantiles for balanced Low/Med/High buckets; median for simpler High/Low.</div>", unsafe_allow_html=True)

    # Real-time preview (if data available)
    if st.session_state.data is not None:
        with st.spinner("Preparing preview…"):
            @st.cache_data(show_spinner=False)
            def _bin_and_basket(_df: pd.DataFrame, _strategy: str):
                binned = bin_kpis(_df, strategy=_strategy)
                basket = to_basket(binned)
                return binned, basket
            binned, basket = _bin_and_basket(st.session_state.data, bin_strategy)
            st.session_state.binned, st.session_state.basket = binned, basket
            st.caption("Preview: Basket width (items) → {}".format(basket.shape[1]))

# ---------------- Results Tab -----------------
with TAB_RESULTS:
    st.subheader("Results & Visualizations")
    if st.session_state.basket is None:
        st.info("Please configure in the Setup tab.")
    else:
        @st.cache_data(show_spinner=False)
        def _mine(_basket: pd.DataFrame, algo: str, min_support: float, metric: str, min_threshold: float, max_len: int):
            return mine_rules(_basket, algo=algo, min_support=min_support, metric=metric, min_threshold=min_threshold, max_len=max_len)
        rules = _mine(st.session_state.basket, algo, float(min_support), metric, float(min_threshold), int(max_len))
        st.session_state.rules = rules

        if rules.empty:
            st.warning("No rules found. Lower Min Support/Threshold or increase Max Items.")
        else:
            # Executive Summary
            top = rules.head(5)
            avg_lift = float(rules["lift"].median())
            avg_conf = float(rules["confidence"].median())
            avg_support = float(rules["support"].median())
            avg_score = float(rules["confidence_score"].median())

            with summary_container:
                st.markdown(f"""
                <div class='card'>
                <h3>Executive Summary</h3>
                <ul>
                  <li><b>Top patterns discovered:</b> {len(rules)}</li>
                  <li><b>Typical association strength (lift):</b> {avg_lift:.2f}</li>
                  <li><b>Reliability (confidence):</b> {avg_conf:.2f}</li>
                  <li><b>Prevalence (support):</b> {avg_support:.2f}</li>
                  <li><b>Insight quality score:</b> {avg_score:.1f}/100</li>
                </ul>
                <div class='helper'>Impact example: If rules linking <i>High Hold & High ACW → Low CSAT</i> are actioned for the top decile of agents, expected CSAT lift could be ~0.1–0.3 (model-dependent). Use the Actions tab to create targeted bundles.</div>
                </div>
                """, unsafe_allow_html=True)

            # Cross-filter controls
            cf1, cf2 = st.columns(2)
            with cf1:
                filter_contains = st.text_input("Filter items containing… (e.g., CSAT, Hold, CallType_Sales)")
            with cf2:
                min_conf_score = st.slider("Min Insight Quality Score", 0.0, 100.0, 40.0, 1.0)

            view_rules = rules.copy()
            if filter_contains:
                mask = view_rules["antecedents_str"].str.contains(filter_contains, case=False) | view_rules["consequents_str"].str.contains(filter_contains, case=False)
                view_rules = view_rules[mask]
            view_rules = view_rules[view_rules["confidence_score"] >= min_conf_score]

            # Table
            st.dataframe(view_rules.head(100), use_container_width=True)
            st.caption("Select a rule below to drill down.")

            # Drilldown select
            rule_labels = [f"{a} → {c} (lift {l:.2f}, conf {conf:.2f})" for a,c,l,conf in zip(view_rules["antecedents_str"], view_rules["consequents_str"], view_rules["lift"], view_rules["confidence"])]
            selected_idx = st.selectbox("Rule", options=list(range(len(rule_labels))), format_func=lambda i: rule_labels[i] if len(rule_labels)>0 else "", index=0 if len(rule_labels)>0 else 0)
            if len(view_rules) > 0:
                selected_rule = view_rules.iloc[selected_idx]
                st.session_state.selected_rule = selected_rule

                # Visuals
                vc1, vc2 = st.columns(2)
                with vc1:
                    st.markdown("**Top Rules by Lift**")
                    fig = px.bar(view_rules.head(20).sort_values("lift", ascending=True), x="lift", y="antecedents_str", orientation="h", color_discrete_sequence=[PALETTE['c1']])
                    fig.update_layout(height=460, margin=dict(l=10,r=10,t=10,b=10))
                    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                with vc2:
                    st.markdown("**Antecedent → Consequent (Sankey)**")
                    ants = view_rules.head(20)["antecedents_str"].tolist()
                    cons = view_rules.head(20)["consequents_str"].tolist()
                    labels = list(sorted(set(ants + cons)))
                    idx = {lab:i for i,lab in enumerate(labels)}
                    sources = [idx[a] for a in ants]
                    targets = [idx[c] for c in cons]
                    values = (view_rules.head(20)["confidence"]*100).tolist()
                    sankey = go.Figure(data=[go.Sankey(node=dict(label=labels, pad=12, thickness=16), link=dict(source=sources, target=targets, value=values))])
                    sankey.update_layout(height=460, margin=dict(l=10,r=10,t=10,b=10))
                    st.plotly_chart(sankey, use_container_width=True, theme="streamlit")

                st.markdown("**Interactive Network**")
                net_html = build_pyvis_network(view_rules.head(40))
                st.session_state.network_html = net_html
                st.components.v1.html(net_html, height=560, scrolling=True)

                # Agent-level drilldown: show agents matching antecedents of selected rule
                st.markdown("**Drill-down: Agents matching antecedents**")
                if st.session_state.binned is not None:
                    ants_items = [i.strip() for i in selected_rule["antecedents_str"].split(",")]
                    # Map back to binned columns by matching prefixes
                    b = st.session_state.binned
                    basket = st.session_state.basket
                    mask = np.ones(len(b), dtype=bool)
                    for it in ants_items:
                        if it in basket.columns:
                            mask &= (basket[it] == 1).values
                    agent_view = b.loc[mask, ["AgentID","CallType","AHT_sec","ACW_sec","Hold_sec","CSAT","FCR","QA_Compliance","Escalations","RepeatContacts7d","Adherence","Absenteeism"]]
                    st.dataframe(agent_view.head(200), use_container_width=True)

# ---------------- Actions Tab -----------------
with TAB_ACTIONS:
    st.subheader("Actionable Recommendations")
    if st.session_state.rules is None or st.session_state.rules.empty:
        st.info("Run analysis in Results tab first.")
    else:
        rules = st.session_state.rules
        top_rules = rules.head(20)
        # Auto-generate recommendations with impact + confidence bands
        recs = []
        for _, r in top_rules.iterrows():
            ants, cons = r["antecedents_str"], r["consequents_str"]
            conf = r["confidence"]
            lift = r["lift"]
            support = r["support"]
            score = r["confidence_score"]
            # Simple impact heuristic on CSAT delta if CSAT appears in consequents
            impact = "~0.1–0.3 CSAT" if "CSAT" in cons else "Operational efficiency gain"
            band = "High" if score >= 70 else ("Medium" if score >= 45 else "Emerging")
            recs.append({
                "Rule": f"{ants} → {cons}",
                "Confidence": round(conf,3),
                "Lift": round(lift,2),
                "Support": round(support,3),
                "InsightScore": score,
                "ExpectedImpact": impact,
                "ConfidenceBand": band,
                "Next Steps": "Targeted coaching; process check; KB refresh; monitor next 4 weeks"
            })
        recs_df = pd.DataFrame(recs)
        st.dataframe(recs_df, use_container_width=True)
        st.markdown("<div class='helper'>InsightScore blends confidence, lift, conviction and support to indicate reliability for action.</div>", unsafe_allow_html=True)

# ---------------- Export Tab -----------------
with TAB_EXPORT:
    st.subheader("Export & Share")
    if st.session_state.rules is None or st.session_state.rules.empty:
        st.info("No outputs yet. Run the analysis first.")
    else:
        rules = st.session_state.rules
        binned = st.session_state.binned
        net_html = st.session_state.network_html or ""

        # CSVs
        rules_csv = rules.to_csv(index=False).encode("utf-8")
        proc_csv = binned.to_csv(index=False).encode("utf-8")
        network_html_bytes = net_html.encode("utf-8")

        c1, c2, c3, c4 = st.columns(4)
        c1.download_button("Rules (CSV)", rules_csv, file_name="affinity_rules.csv", mime="text/csv")
        c2.download_button("Processed Data (CSV)", proc_csv, file_name="processed_kpis_binned.csv", mime="text/csv")
        c3.download_button("Network (HTML)", network_html_bytes, file_name="affinity_network.html", mime="text/html")

        # Executive report (print-optimized)
        report_html = f"""
        <html><head><meta charset='utf-8'><title>Executive Report — Performance Affinity Analysis</title>
        <style>body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;background:#fff;color:#0f172a;line-height:1.5;margin:24px;}}
        h1,h2{{color:#0f172a}} table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #e5e7eb;padding:8px}} .muted{{color:#64748b}}</style>
        </head><body>
        <h1>Executive Report</h1>
        <p class='muted'>Auto-generated from the Performance Affinity Analysis app.</p>
        <h2>Summary</h2>
        <ul>
          <li>Total rules: {len(rules)}</li>
          <li>Median lift: {rules['lift'].median():.2f} | Median confidence: {rules['confidence'].median():.2f} | Median support: {rules['support'].median():.2f}</li>
        </ul>
        <h2>Top Rules</h2>
        {rules.head(20).to_html(index=False)}
        </body></html>
        """
        report_bytes = report_html.encode("utf-8")
        c4.download_button("Executive Report (HTML)", report_bytes, file_name="executive_report.html", mime="text/html")

        # ZIP bundle
        import zipfile
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("affinity_rules.csv", rules_csv)
            zf.writestr("processed_kpis_binned.csv", proc_csv)
            zf.writestr("affinity_network.html", network_html_bytes)
            zf.writestr("executive_report.html", report_bytes)
        st.download_button("Download All (ZIP)", data=zip_buffer.getvalue(), file_name="affinity_outputs.zip", mime="application/zip", use_container_width=True)

# ---------------------------------
# Footer helper
# ---------------------------------
st.markdown("""
<div class='helper'>Tip: Share the current parameters with colleagues using URL query params. Set presets/benchmarks, then copy the URL.</div>
""", unsafe_allow_html=True)
