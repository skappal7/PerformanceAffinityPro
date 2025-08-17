# app.py — Performance Affinity Analysis (Production-Ready)
# -------------------------------------------------------
# A modern, animated, and interactive Streamlit app that applies
# Market Basket Analysis (Apriori/FP-Growth) to Contact Center KPIs
# to discover co-occurring performance patterns ("affinities").
#
# ✅ Python 3.11+
# ✅ Stateful & cached
# ✅ Rich UI + custom CSS
# ✅ Interactive visuals (Plotly + PyVis)
# ✅ Downloadable outputs (rules, processed data, charts, network HTML, requirements)
# ✅ Sample dataset generator & walkthrough guidance
#
# Suggested requirements (auto-downloadable from within the app):
#   streamlit>=1.36
#   pandas>=2.2
#   numpy>=1.26
#   plotly>=5.22
#   mlxtend>=0.23
#   networkx>=3.2
#   pyvis>=0.3.2
#
# Run:  streamlit run app.py

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

# -----------------------------
# Global App Config & Theming
# -----------------------------
st.set_page_config(
    page_title="Performance Affinity Analysis (Contact Center)",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Custom CSS (cards, animations, glassmorphism, tooltips)
# -----------------------------
CUSTOM_CSS = """
<style>
:root {
  --acc-1: #4f46e5; /* indigo-600 */
  --acc-2: #06b6d4; /* cyan-500 */
  --acc-3: #10b981; /* emerald-500 */
  --bg-grad-1: #0f172a; /* slate-900 */
  --bg-grad-2: #111827; /* gray-900 */
  --card-bg: rgba(255,255,255,0.06);
  --card-brd: rgba(255,255,255,0.15);
  --muted: #94a3b8; /* slate-400 */
}

/* Background gradient */
.stApp {
  background: radial-gradient(1200px 600px at 20% -10%, rgba(79,70,229,0.18), transparent 60%),
              radial-gradient(1000px 800px at 100% 0%, rgba(6,182,212,0.18), transparent 50%),
              linear-gradient(120deg, var(--bg-grad-1), var(--bg-grad-2));
  color: #e5e7eb;
}

/* Headings */
h1,h2,h3 { text-shadow: 0 1px 12px rgba(0,0,0,0.25); }

/* Glass cards */
.card { 
  background: var(--card-bg); 
  border: 1px solid var(--card-brd); 
  border-radius: 18px; 
  padding: 16px 18px; 
  box-shadow: 0 8px 24px rgba(0,0,0,0.25);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}
.card:hover { transform: translateY(-2px); box-shadow: 0 16px 32px rgba(0,0,0,0.35); border-color: rgba(255,255,255,0.25); }

/* Subtle fade-in */
.fadein { animation: fadeIn .6s ease both; }
@keyframes fadeIn { from { opacity:0; transform: translateY(4px);} to {opacity:1; transform: translateY(0);} }

/* Metric pills */
.pill { display:inline-block; padding:6px 10px; border-radius:999px; border:1px solid var(--card-brd); background:rgba(255,255,255,0.06); margin-right:8px; }

/* Tables */
.dataframe tbody tr:hover { background: rgba(255,255,255,0.05); }

/* Sidebar tweak */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(17,24,39,0.9), rgba(2,6,23,0.9)); }

/* Tooltip-ish helper */
.helper { font-size:0.92rem; color: var(--muted); }

.download-bar { display:flex; gap:10px; flex-wrap:wrap; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------
# Helpers: Sample Data, Binning, Basket Transform, Mining, Visualization
# -----------------------------

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

    # Inject realistic correlations (e.g., high ACW -> lower CSAT, longer AHT -> more escalations)
    mask_hi_acw = df["ACW_sec"] > df["ACW_sec"].quantile(0.75)
    df.loc[mask_hi_acw, "CSAT"] = (df.loc[mask_hi_acw, "CSAT"] - np.abs(np.random.normal(0.35, 0.2, mask_hi_acw.sum()))).clip(1.0, 5.0)

    mask_hi_hold = df["Hold_sec"] > df["Hold_sec"].quantile(0.75)
    df.loc[mask_hi_hold, "RepeatContacts7d"] += np.random.poisson(0.8, mask_hi_hold.sum())

    mask_low_qa = df["QA_Compliance"] < df["QA_Compliance"].quantile(0.25)
    df.loc[mask_low_qa, "Escalations"] += np.random.poisson(1.2, mask_low_qa.sum())

    return df


def bin_kpis(df: pd.DataFrame, strategy: str = "quantile") -> pd.DataFrame:
    """Create High/Medium/Low (or True/False) bins for KPIs to form items."""
    binned = df.copy()

    # Numeric features to bin
    q_features = ["AHT_sec", "ACW_sec", "Hold_sec", "CSAT", "Adherence", "QA_Compliance", "Escalations", "RepeatContacts7d", "Absenteeism"]
    for col in q_features:
        if strategy == "quantile":
            # 3-quantiles: Low/Med/High
            try:
                q = pd.qcut(binned[col], 3, labels=["Low", "Med", "High"])
            except ValueError:
                # Fallback to 2 bins if not enough unique values
                q = pd.qcut(binned[col], 2, labels=["Low", "High"])            
            binned[col+"_BIN"] = q.astype(str)
        else:  # median high/low
            med = binned[col].median()
            binned[col+"_BIN"] = np.where(binned[col] >= med, "High", "Low")

    # Binary features preserve semantics
    if "FCR" in binned.columns:
        binned["FCR_BIN"] = np.where(binned["FCR"] == 1, "FCR_Yes", "FCR_No")

    return binned


def to_basket(df_binned: pd.DataFrame) -> pd.DataFrame:
    """Convert binned columns to one-hot basket (wide boolean) for MBA."""
    item_cols = [c for c in df_binned.columns if c.endswith("_BIN")]

    # One-hot encode each binned feature
    baskets = []
    for col in item_cols:
        dummies = pd.get_dummies(df_binned[col], prefix=col.replace("_BIN", ""), dtype=int)
        baskets.append(dummies)
    basket_df = pd.concat(baskets, axis=1)

    # Also include categorical call type as items
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
    # Sort by descending lift to prioritize strong/interesting rules
    rules = rules.sort_values("lift", ascending=False)

    # Convert frozensets to readable strings
    def set_to_str(s: Any) -> str:
        return ", ".join(sorted(list(s))) if isinstance(s, (set, frozenset)) else str(s)

    rules["antecedents_str"] = rules["antecedents"].apply(set_to_str)
    rules["consequents_str"] = rules["consequents"].apply(set_to_str)

    # Friendly columns
    cols = [
        "antecedents_str", "consequents_str", "support", "confidence", "lift",
        "leverage", "conviction",
    ]
    rules = rules[cols]

    # Round for presentation
    rules[["support","confidence","lift","leverage","conviction"]] = rules[["support","confidence","lift","leverage","conviction"]].round(4)

    return rules.reset_index(drop=True)


def build_pyvis_network(rules: pd.DataFrame, height: str = "600px") -> str:
    """Build interactive network (antecedent -> consequent) and return HTML string."""
    net = Network(height=height, width="100%", bgcolor="#0b1220", font_color="#e5e7eb")
    net.barnes_hut(gravity=-15000, central_gravity=0.3, spring_length=120, spring_strength=0.02, damping=0.9)

    # Add nodes and edges
    nodes = set()
    for _, r in rules.iterrows():
        a = r["antecedents_str"].split(", ")
        c = r["consequents_str"].split(", ")
        for antecedent in a:
            nodes.add(antecedent)
        for consequent in c:
            nodes.add(consequent)

    for node in nodes:
        color = "#4f46e5" if "High" in node else ("#10b981" if "FCR_Yes" in node else ("#06b6d4" if "CallType_" in node else "#eab308"))
        net.add_node(node, label=node, color=color)

    for _, r in rules.iterrows():
        for a in r["antecedents_str"].split(", "):
            for c in r["consequents_str"].split(", "):
                title = f"lift: {r['lift']}, conf: {r['confidence']}, support: {r['support']}"
                net.add_edge(a, c, title=title, value=float(r["lift"]))

    return net.generate_html()


# -----------------------------
# Stateful Containers
# -----------------------------
if "data" not in st.session_state:
    st.session_state.data = None
if "binned" not in st.session_state:
    st.session_state.binned = None
if "basket" not in st.session_state:
    st.session_state.basket = None
if "rules" not in st.session_state:
    st.session_state.rules = None
if "network_html" not in st.session_state:
    st.session_state.network_html = None


# -----------------------------
# Sidebar — Controls & Guidance
# -----------------------------
with st.sidebar:
    st.markdown("## 🧭 Performance Affinity Analysis")
    st.markdown(
        """
        Discover **co-occurring KPI patterns** (affinities) across agents, call types, and outcomes. Use this to bundle coaching themes, spot multi-KPI risks, and prioritize fixes.
        """
    )

    st.markdown("### 📥 Data Source")
    uploaded = st.file_uploader("Upload CSV (sample schema below)", type=["csv"], accept_multiple_files=False)

    st.markdown("**or**")
    if st.button("Generate Sample Dataset", type="primary", use_container_width=True):
        with st.spinner("Synthesizing realistic contact center data…"):
            st.session_state.data = generate_sample_contact_center_data()
            st.toast("Sample data ready ✓")

    st.markdown("---")
    st.markdown("### ⚙️ Preprocessing")
    bin_strategy = st.radio("Binning Strategy", ["quantile", "median"], help="How to convert continuous KPIs into categorical items.")

    st.markdown("### 🧮 Mining Settings")
    algo = st.selectbox("Algorithm", ["apriori", "fpgrowth"], index=0)
    min_support = st.slider("Min Support", 0.01, 0.3, 0.08, 0.01)
    metric = st.selectbox("Rule Metric", ["lift", "confidence", "support"], index=0)
    min_threshold = st.slider("Min Threshold", 0.5, 3.0, 1.05, 0.05, help="Minimum metric threshold for rule generation.")
    max_len = st.slider("Max Items per Rule", 2, 5, 3)

    st.markdown("---")
    st.markdown("### 💾 Exports")
    want_zip = st.checkbox("Bundle all outputs into a ZIP for download", value=True)


# -----------------------------
# Main — Header
# -----------------------------
st.title("Performance Affinity Analysis (Contact Center)")

st.markdown(
    """
    <div class="card fadein">
    <b>What is this?</b> This app mines <span class="pill">frequent associations</span> between your KPIs and behaviors — for example, <i>High Hold + High ACW → Low CSAT</i> — so you can bundle coaching, target process fixes, and watch how patterns evolve.
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Data Loading & Schema Helper
# -----------------------------
SAMPLE_SCHEMA = [
    ("AgentID", "string"), ("CallType", "category"), ("AHT_sec", "int"), ("ACW_sec", "int"),
    ("Hold_sec", "int"), ("CSAT", "float [1-5]"), ("FCR", "0/1"), ("Adherence", "float [0-1]"),
    ("QA_Compliance", "float [0-1]"), ("Escalations", "int"), ("RepeatContacts7d", "int"), ("Absenteeism", "float [0-1]")
]

schema_df = pd.DataFrame(SAMPLE_SCHEMA, columns=["Column", "Type/Range"])
with st.expander("👀 Expected Schema & Sample Download"):
    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(schema_df, use_container_width=True)
    with c2:
        sample = generate_sample_contact_center_data(n_agents=300)
        csv_bytes = sample.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Sample CSV",
            data=csv_bytes,
            file_name="sample_contact_center_kpis.csv",
            mime="text/csv",
            help="Use this sample to try the app end-to-end."
        )

# Load uploaded data if present
if uploaded is not None:
    try:
        df_in = pd.read_csv(uploaded)
        st.session_state.data = df_in
        st.toast("Uploaded data loaded ✓")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

# Guard: ensure we have data
if st.session_state.data is None:
    st.info("Upload a CSV or click **Generate Sample Dataset** in the sidebar to begin.")
    st.stop()

# -----------------------------
# Preprocess & Basketize (Cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def _bin_and_basket(_df: pd.DataFrame, _strategy: str):
    binned = bin_kpis(_df, strategy=_strategy)
    basket = to_basket(binned)
    return binned, basket

binned, basket = _bin_and_basket(st.session_state.data, bin_strategy)
st.session_state.binned = binned
st.session_state.basket = basket

# -----------------------------
# Mine Rules (Cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def _mine(_basket: pd.DataFrame, algo: str, min_support: float, metric: str, min_threshold: float, max_len: int):
    return mine_rules(_basket, algo=algo, min_support=min_support, metric=metric, min_threshold=min_threshold, max_len=max_len)

with st.spinner("Mining performance affinities…"):
    rules = _mine(basket, algo, float(min_support), metric, float(min_threshold), int(max_len))

st.session_state.rules = rules

# -----------------------------
# Results Overview
# -----------------------------
if rules.empty:
    st.warning("No rules found with the current thresholds. Try lowering **Min Support** or **Min Threshold**, or increase **Max Items per Rule**.")
    st.stop()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1: st.metric("Rules Found", len(rules))
with kpi2: st.metric("Median Lift", float(rules["lift"].median()))
with kpi3: st.metric("Top Confidence", float(rules["confidence"].max()))
with kpi4: st.metric("Median Support", float(rules["support"].median()))

st.markdown("<div class='helper'>Higher <b>lift</b> (>1) suggests the pattern occurs more often than by chance. Use <b>support</b> to ensure patterns are common enough to matter.</div>", unsafe_allow_html=True)

# -----------------------------
# Top Rules Table + Commentary
# -----------------------------
st.subheader("Top Affinity Rules")

left, right = st.columns([1.7, 1.3], vertical_alignment="top")

with left:
    top_n = st.slider("Show Top-N by Lift", 5, min(100, len(rules)), min(20, len(rules)))
    top_rules = rules.head(top_n)
    st.dataframe(top_rules, use_container_width=True)

with right:
    st.markdown(
        """
        <div class="card fadein">
        <b>How to read:</b><br>
        • <b>Antecedents → Consequents</b>: if the left-hand items occur, the right-hand items are likely to occur, too.<br>
        • <b>Support</b>: how frequently this rule appears in your data.<br>
        • <b>Confidence</b>: probability of consequents given antecedents.<br>
        • <b>Lift</b>: how much more often the pattern occurs vs random chance (\>1 is interesting).<br>
        <br>
        <b>Tip:</b> Sort by <i>lift</i> to find non-obvious insights, then filter by <i>support</i> for operational significance.
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Visuals: Bar (Top Lift) + Sankey + Network (PyVis)
# -----------------------------
vis1, vis2 = st.columns([1,1])

with vis1:
    st.markdown("### 📈 Top Rules by Lift")
    fig = px.bar(
        top_rules.sort_values("lift", ascending=True),
        x="lift", y="antecedents_str",
        orientation="h",
        hover_data=["consequents_str", "support", "confidence"],
        title="Higher lift = stronger-than-chance association"
    )
    fig.update_layout(height=480, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

with vis2:
    st.markdown("### 🔀 Antecedent → Consequent Flow (Sankey)")
    # Build Sankey sources/targets
    ants = top_rules["antecedents_str"].tolist()
    cons = top_rules["consequents_str"].tolist()
    labels = list(sorted(set(ants + cons)))
    idx = {lab:i for i,lab in enumerate(labels)}
    sources = [idx[a] for a in ants]
    targets = [idx[c] for c in cons]
    values = (top_rules["confidence"]*100).tolist()

    sankey = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=12, thickness=16),
        link=dict(source=sources, target=targets, value=values, hovertemplate='Confidence %{value:.1f}%')
    )])
    sankey.update_layout(height=480, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(sankey, use_container_width=True, theme="streamlit")

st.markdown("### 🌐 Interactive Network of KPI Affinities")

with st.spinner("Building interactive network…"):
    net_html = build_pyvis_network(top_rules)
    st.session_state.network_html = net_html

# Embed PyVis network
st.components.v1.html(st.session_state.network_html, height=620, scrolling=True)

st.markdown(
    """
    <div class="helper">Nodes are items (e.g., <i>AHT High</i>, <i>CSAT Low</i>), and directed edges represent rules. Hover to see <i>lift</i>, <i>confidence</i>, and <i>support</i>.</div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Co-occurrence Heatmap (bonus context)
# -----------------------------
st.subheader("Co-occurrence Heatmap (Quick Context)")

# Build a compact co-occurrence matrix among selected items from top rules
selected_items = sorted(set(
    ", ".join(top_rules["antecedents_str"]).split(", ") + ", ".join(top_rules["consequents_str"]).split(", ")
))

# Limit size for readability
selected_items = selected_items[:30]

cooc = pd.DataFrame(0, index=selected_items, columns=selected_items, dtype=float)
for item in selected_items:
    if item in basket.columns:
        mask_i = basket[item] == 1
        for item2 in selected_items:
            if item2 in basket.columns:
                cooc.loc[item, item2] = float(((basket[item2] == 1) & mask_i).mean())

heat = px.imshow(cooc, aspect="auto", title="Item Co-occurrence (as fraction of rows)")
heat.update_layout(height=520, margin=dict(l=10,r=10,t=40,b=10))
st.plotly_chart(heat, use_container_width=True, theme="streamlit")

# -----------------------------
# Suggested Coaching Bundles (auto-explained)
# -----------------------------
st.subheader("🎯 Suggested Coaching Bundles (Generated)")

suggestions = []
for _, r in top_rules.iterrows():
    ants = r["antecedents_str"]
    cons = r["consequents_str"]
    if "CSAT" in cons or "FCR_No" in cons:
        suggestions.append(f"When {ants} occurs, {cons} is likely. Coach on handle efficiency and empathy; review hold/ACW drivers.")
    elif "Escalations" in cons or "RepeatContacts7d" in cons:
        suggestions.append(f"{ants} → {cons}. Emphasize root-cause resolution, knowledge base navigation, and first-contact closure.")
    else:
        suggestions.append(f"{ants} → {cons}. Review scripts & process steps driving this linkage; consider quick-hit refreshers.")

if suggestions:
    s_df = pd.DataFrame({"Recommendation": suggestions[:12]})
    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.dataframe(s_df, use_container_width=True)
    with c2:
        st.markdown(
            """
            <div class="card fadein">
            These are <i>auto-generated</i> based on the strongest rules you selected above. Use them to create <b>bundled coaching themes</b> where one action addresses multiple KPIs at once.
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("No coaching suggestions available for the current rule set.")

# -----------------------------
# Downloads — Rules CSV, Processed Data, Network HTML, Requirements, ZIP
# -----------------------------
st.subheader("⬇️ Download Outputs")

# Rules CSV
rules_csv = rules.to_csv(index=False).encode("utf-8")
proc_csv = binned.to_csv(index=False).encode("utf-8")
network_html_bytes = st.session_state.network_html.encode("utf-8") if st.session_state.network_html else b""

colz = st.columns(4)
colz[0].download_button("Rules (CSV)", rules_csv, file_name="affinity_rules.csv", mime="text/csv")
colz[1].download_button("Processed Data (CSV)", proc_csv, file_name="processed_kpis_binned.csv", mime="text/csv")
colz[2].download_button("Network (HTML)", network_html_bytes, file_name="affinity_network.html", mime="text/html")

# Requirements builder
requirements = textwrap.dedent(
    """
    streamlit>=1.36
    pandas>=2.2
    numpy>=1.26
    plotly>=5.22
    mlxtend>=0.23
    networkx>=3.2
    pyvis>=0.3.2
    """
).strip().encode("utf-8")
colz[3].download_button("requirements.txt", data=requirements, file_name="requirements.txt")

# Optional ZIP bundle
if want_zip:
    import zipfile
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("affinity_rules.csv", rules_csv)
        zf.writestr("processed_kpis_binned.csv", proc_csv)
        zf.writestr("affinity_network.html", network_html_bytes)
        zf.writestr("requirements.txt", requirements)
        # Export figures as stand-alone HTML for portability
        zf.writestr("top_rules_lift.html", fig.to_html(full_html=True, include_plotlyjs="cdn"))
        zf.writestr("sankey.html", sankey.to_html(full_html=True, include_plotlyjs="cdn"))
        zf.writestr("cooccurrence_heatmap.html", heat.to_html(full_html=True, include_plotlyjs="cdn"))
    st.download_button(
        "Download All (ZIP)",
        data=zip_buffer.getvalue(),
        file_name="performance_affinity_outputs.zip",
        mime="application/zip",
        use_container_width=True,
    )

# -----------------------------
# Footer Guidance
# -----------------------------
st.markdown(
    """
    <br>
    <div class="card fadein">
    <b>Next steps:</b>
    <ol>
      <li>Filter and tag high-lift, adequate-support rules as <i>Actionable</i>.</li>
      <li>Turn suggestions into <b>coaching bundles</b> and track KPI shifts monthly.</li>
      <li>Automate data refresh (daily/weekly) and alert when <i>new high-lift patterns</i> emerge.</li>
    </ol>
    </div>
    """,
    unsafe_allow_html=True,
)
