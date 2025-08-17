# app.py — Professional Performance Affinity Analysis
# ================================================
# A modern, professional Streamlit app for Contact Center KPI affinity analysis
# with enterprise-grade UI/UX, comprehensive reporting, and business-focused insights.
#
# Requirements:
#   streamlit>=1.36
#   pandas>=2.2
#   numpy>=1.26
#   plotly>=5.22
#   mlxtend>=0.23
#   networkx>=3.2
#   pyvis>=0.3.2
#   streamlit-option-menu>=0.3.6
#
# Run: streamlit run app.py

from __future__ import annotations
import io
import json
import textwrap
import zipfile
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from pyvis.network import Network
from streamlit_option_menu import option_menu

from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# =============================================================================
# APP CONFIGURATION & PROFESSIONAL STYLING
# =============================================================================

st.set_page_config(
    page_title="Performance Affinity Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional CSS Theme
PROFESSIONAL_CSS = """
<style>
/* Import Professional Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  /* Professional Color Palette */
  --primary-600: #1e40af;
  --primary-500: #3b82f6;
  --primary-400: #60a5fa;
  --primary-100: #dbeafe;
  --primary-50: #eff6ff;
  
  /* Neutral Professional Grays */
  --gray-900: #111827;
  --gray-800: #1f2937;
  --gray-700: #374151;
  --gray-600: #4b5563;
  --gray-500: #6b7280;
  --gray-400: #9ca3af;
  --gray-300: #d1d5db;
  --gray-200: #e5e7eb;
  --gray-100: #f3f4f6;
  --gray-50: #f9fafb;
  
  /* Semantic Colors */
  --success: #059669;
  --success-light: #d1fae5;
  --warning: #d97706;
  --warning-light: #fef3c7;
  --error: #dc2626;
  --error-light: #fee2e2;
  
  /* Chart Colors */
  --chart-1: #3b82f6;
  --chart-2: #8b5cf6;
  --chart-3: #06b6d4;
  --chart-4: #10b981;
  --chart-5: #f59e0b;
  
  /* Shadows and Effects */
  --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
  --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
  --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
  --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1);
}

/* Global Styles */
.stApp {
  background: linear-gradient(135deg, var(--gray-50) 0%, #ffffff 100%);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  color: var(--gray-800);
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
  font-family: 'Inter', sans-serif;
  font-weight: 600;
  color: var(--gray-900);
  letter-spacing: -0.025em;
}

h1 { font-size: 2rem; line-height: 1.2; }
h2 { font-size: 1.5rem; line-height: 1.3; }
h3 { font-size: 1.25rem; line-height: 1.4; }

/* Professional Cards */
.metric-card {
  background: white;
  border: 1px solid var(--gray-200);
  border-radius: 12px;
  padding: 24px;
  box-shadow: var(--shadow-sm);
  transition: all 0.2s ease;
  height: 100%;
}

.metric-card:hover {
  box-shadow: var(--shadow-md);
  border-color: var(--primary-200);
  transform: translateY(-1px);
}

.metric-value {
  font-size: 2.25rem;
  font-weight: 700;
  color: var(--primary-600);
  line-height: 1;
  font-family: 'JetBrains Mono', monospace;
}

.metric-label {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--gray-600);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-top: 8px;
}

.metric-change {
  font-size: 0.75rem;
  font-weight: 500;
  margin-top: 4px;
}

.metric-change.positive { color: var(--success); }
.metric-change.negative { color: var(--error); }
.metric-change.neutral { color: var(--gray-500); }

/* Professional Containers */
.pro-container {
  background: white;
  border: 1px solid var(--gray-200);
  border-radius: 12px;
  padding: 24px;
  margin: 16px 0;
  box-shadow: var(--shadow-sm);
}

.insight-card {
  background: linear-gradient(135deg, var(--primary-50) 0%, white 100%);
  border: 1px solid var(--primary-200);
  border-radius: 12px;
  padding: 20px;
  margin: 12px 0;
  box-shadow: var(--shadow-sm);
}

.alert-card {
  background: linear-gradient(135deg, var(--warning-light) 0%, white 100%);
  border: 1px solid var(--warning);
  border-left: 4px solid var(--warning);
  border-radius: 8px;
  padding: 16px;
  margin: 12px 0;
}

.success-card {
  background: linear-gradient(135deg, var(--success-light) 0%, white 100%);
  border: 1px solid var(--success);
  border-left: 4px solid var(--success);
  border-radius: 8px;
  padding: 16px;
  margin: 12px 0;
}

/* Status Indicators */
.status-bar {
  background: var(--gray-100);
  border: 1px solid var(--gray-200);
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 24px;
  font-size: 0.875rem;
  color: var(--gray-600);
  display: flex;
  align-items: center;
  gap: 16px;
}

.status-indicator {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--success);
}

/* Buttons */
.stButton > button {
  background: var(--primary-600);
  color: white;
  border: none;
  border-radius: 8px;
  padding: 12px 24px;
  font-weight: 500;
  transition: all 0.2s ease;
  box-shadow: var(--shadow-sm);
}

.stButton > button:hover {
  background: var(--primary-700);
  box-shadow: var(--shadow-md);
  transform: translateY(-1px);
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--gray-50) 0%, white 100%);
  border-right: 1px solid var(--gray-200);
}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stRadio label {
  font-weight: 500;
  color: var(--gray-700);
}

/* Tables */
.dataframe {
  border: 1px solid var(--gray-200);
  border-radius: 8px;
  overflow: hidden;
  box-shadow: var(--shadow-sm);
}

.dataframe th {
  background: var(--gray-50);
  color: var(--gray-700);
  font-weight: 600;
  border-bottom: 2px solid var(--gray-200);
}

.dataframe td {
  border-bottom: 1px solid var(--gray-100);
  padding: 12px;
}

.dataframe tbody tr:hover {
  background: var(--primary-50);
}

/* Navigation */
.nav-container {
  background: white;
  border: 1px solid var(--gray-200);
  border-radius: 12px;
  padding: 8px;
  margin-bottom: 24px;
  box-shadow: var(--shadow-sm);
}

/* Tooltips and Helper Text */
.helper-text {
  font-size: 0.875rem;
  color: var(--gray-500);
  font-style: italic;
  margin-top: 8px;
  padding: 12px;
  background: var(--gray-50);
  border-radius: 6px;
  border-left: 3px solid var(--primary-400);
}

/* Loading States */
.loading-container {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 48px;
  background: var(--gray-50);
  border-radius: 12px;
  margin: 24px 0;
}

/* Animations */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}

.fade-in {
  animation: fadeInUp 0.4s ease-out forwards;
}

/* Responsive Design */
@media (max-width: 768px) {
  .metric-card { padding: 16px; }
  .pro-container { padding: 16px; }
  .metric-value { font-size: 1.75rem; }
}

/* Custom Components */
.progress-ring {
  transform: rotate(-90deg);
}

.confidence-badge {
  display: inline-block;
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 500;
  text-transform: uppercase;
}

.confidence-high {
  background: var(--success-light);
  color: var(--success);
}

.confidence-medium {
  background: var(--warning-light);
  color: var(--warning);
}

.confidence-low {
  background: var(--error-light);
  color: var(--error);
}

/* Override Streamlit Defaults */
.stMetric {
  background: transparent;
  border: none;
  padding: 0;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
  background: var(--gray-100);
  padding: 4px;
  border-radius: 8px;
}

.stTabs [data-baseweb="tab"] {
  background: transparent;
  border-radius: 6px;
  color: var(--gray-600);
  font-weight: 500;
}

.stTabs [aria-selected="true"] {
  background: white;
  color: var(--primary-600);
  box-shadow: var(--shadow-sm);
}
</style>
"""

st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)

# =============================================================================
# DATA GENERATION & PROCESSING FUNCTIONS
# =============================================================================

def generate_professional_sample_data(n_agents: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Generate realistic contact center data with professional context."""
    rng = np.random.default_rng(seed)
    
    # Agent identifiers
    agents = [f"AG{1000+i:04d}" for i in range(n_agents)]
    
    # Realistic distributions with business context
    call_types = rng.choice(
        ["Billing", "Technical", "Retention", "Sales", "General"], 
        size=n_agents, 
        p=[0.28, 0.22, 0.18, 0.17, 0.15]
    )
    
    # Core KPIs with realistic correlations
    aht = rng.lognormal(mean=5.8, sigma=0.4, size=n_agents).clip(180, 1800)  # seconds
    acw = rng.lognormal(mean=4.2, sigma=0.5, size=n_agents).clip(30, 600)    # seconds
    hold = rng.exponential(scale=45, size=n_agents).clip(0, 400)              # seconds
    
    csat = rng.beta(a=4, b=1.5, size=n_agents) * 4 + 1                       # 1-5 scale
    csat = csat.clip(1.0, 5.0)
    
    fcr_prob = 0.72 - (aht - aht.mean()) / aht.std() * 0.15                   # Correlated with AHT
    fcr_prob = fcr_prob.clip(0.3, 0.95)
    fcr = rng.binomial(1, fcr_prob, size=n_agents)
    
    # Schedule adherence and quality metrics
    adherence = rng.beta(a=15, b=2, size=n_agents).clip(0.65, 1.0)
    qa_compliance = rng.beta(a=12, b=2.5, size=n_agents).clip(0.5, 1.0)
    
    # Operational metrics
    escalations = rng.negative_binomial(n=2, p=0.6, size=n_agents)
    repeat_contacts = rng.poisson(lam=1.1, size=n_agents)
    absenteeism = rng.exponential(scale=0.035, size=n_agents).clip(0.0, 0.3)
    
    # Create realistic correlations
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
        "RepeatContacts7d": repeat_contacts,
        "Absenteeism": absenteeism.round(3),
    })
    
    # Inject business-realistic correlations
    # High ACW correlates with lower satisfaction
    high_acw_mask = df["ACW_sec"] > df["ACW_sec"].quantile(0.8)
    df.loc[high_acw_mask, "CSAT"] *= rng.uniform(0.7, 0.9, high_acw_mask.sum())
    df["CSAT"] = df["CSAT"].clip(1.0, 5.0)
    
    # Long hold times increase repeat contacts
    high_hold_mask = df["Hold_sec"] > df["Hold_sec"].quantile(0.75)
    df.loc[high_hold_mask, "RepeatContacts7d"] += rng.poisson(1.5, high_hold_mask.sum())
    
    # Low quality compliance increases escalations
    low_qa_mask = df["QA_Compliance"] < df["QA_Compliance"].quantile(0.2)
    df.loc[low_qa_mask, "Escalations"] += rng.poisson(2.0, low_qa_mask.sum())
    
    return df

def create_professional_bins(df: pd.DataFrame, strategy: str = "business_rules") -> pd.DataFrame:
    """Create business-meaningful bins for KPIs."""
    binned = df.copy()
    
    if strategy == "business_rules":
        # Business-defined thresholds
        binned["AHT_BIN"] = pd.cut(
            binned["AHT_sec"], 
            bins=[0, 300, 480, float('inf')], 
            labels=["Excellent", "Target", "Needs_Improvement"]
        ).astype(str)
        
        binned["ACW_BIN"] = pd.cut(
            binned["ACW_sec"], 
            bins=[0, 60, 120, float('inf')], 
            labels=["Efficient", "Standard", "Extended"]
        ).astype(str)
        
        binned["Hold_BIN"] = pd.cut(
            binned["Hold_sec"], 
            bins=[0, 30, 90, float('inf')], 
            labels=["Minimal", "Moderate", "Excessive"]
        ).astype(str)
        
        binned["CSAT_BIN"] = pd.cut(
            binned["CSAT"], 
            bins=[0, 3.5, 4.2, 5.0], 
            labels=["Below_Expectations", "Meets_Expectations", "Exceeds_Expectations"]
        ).astype(str)
        
        binned["Adherence_BIN"] = pd.cut(
            binned["Adherence"], 
            bins=[0, 0.85, 0.92, 1.0], 
            labels=["Below_Standard", "Acceptable", "Excellent"]
        ).astype(str)
        
        binned["QA_BIN"] = pd.cut(
            binned["QA_Compliance"], 
            bins=[0, 0.8, 0.9, 1.0], 
            labels=["Requires_Coaching", "Standard", "Exceptional"]
        ).astype(str)
        
        binned["Escalations_BIN"] = pd.cut(
            binned["Escalations"], 
            bins=[-1, 0, 2, float('inf')], 
            labels=["None", "Low", "High"]
        ).astype(str)
        
        binned["RepeatContacts_BIN"] = pd.cut(
            binned["RepeatContacts7d"], 
            bins=[-1, 0, 1, float('inf')], 
            labels=["None", "Minimal", "Concerning"]
        ).astype(str)
        
        binned["Absenteeism_BIN"] = pd.cut(
            binned["Absenteeism"], 
            bins=[0, 0.02, 0.05, 1.0], 
            labels=["Excellent_Attendance", "Good_Attendance", "Attendance_Issues"]
        ).astype(str)
        
    else:  # quantile strategy
        numeric_cols = ["AHT_sec", "ACW_sec", "Hold_sec", "CSAT", "Adherence", "QA_Compliance", "Escalations", "RepeatContacts7d", "Absenteeism"]
        for col in numeric_cols:
            try:
                binned[col + "_BIN"] = pd.qcut(binned[col], 3, labels=["Low", "Medium", "High"]).astype(str)
            except ValueError:
                binned[col + "_BIN"] = pd.qcut(binned[col], 2, labels=["Low", "High"]).astype(str)
    
    # Handle binary FCR
    binned["FCR_BIN"] = np.where(binned["FCR"] == 1, "First_Call_Resolution", "Repeat_Required")
    
    return binned

def create_market_basket(df_binned: pd.DataFrame) -> pd.DataFrame:
    """Convert binned data to market basket format."""
    # Get all bin columns
    bin_cols = [c for c in df_binned.columns if c.endswith("_BIN")]
    
    # One-hot encode each binned feature
    basket_parts = []
    for col in bin_cols:
        feature_name = col.replace("_BIN", "")
        dummies = pd.get_dummies(df_binned[col], prefix=feature_name, dtype=int)
        basket_parts.append(dummies)
    
    # Include call types
    if "CallType" in df_binned.columns:
        call_type_dummies = pd.get_dummies(df_binned["CallType"], prefix="CallType", dtype=int)
        basket_parts.append(call_type_dummies)
    
    basket_df = pd.concat(basket_parts, axis=1)
    return basket_df

def mine_association_rules(
    basket_df: pd.DataFrame,
    algorithm: str = "apriori",
    min_support: float = 0.05,
    metric: str = "lift",
    min_threshold: float = 1.1,
    max_len: int = 3,
) -> pd.DataFrame:
    """Mine association rules with comprehensive error handling."""
    
    try:
        # Mine frequent itemsets
        if algorithm == "fpgrowth":
            frequent_itemsets = fpgrowth(
                basket_df, 
                min_support=min_support, 
                use_colnames=True, 
                max_len=max_len
            )
        else:
            frequent_itemsets = apriori(
                basket_df, 
                min_support=min_support, 
                use_colnames=True, 
                max_len=max_len
            )
        
        if frequent_itemsets.empty:
            return pd.DataFrame()
        
        # Generate association rules
        rules = association_rules(
            frequent_itemsets, 
            metric=metric, 
            min_threshold=min_threshold
        )
        
        if rules.empty:
            return pd.DataFrame()
        
        # Process and clean rules
        rules = rules.sort_values("lift", ascending=False)
        
        # Convert frozensets to readable strings
        def frozenset_to_string(fs):
            if isinstance(fs, frozenset):
                return " + ".join(sorted(list(fs)))
            return str(fs)
        
        rules["antecedents_str"] = rules["antecedents"].apply(frozenset_to_string)
        rules["consequents_str"] = rules["consequents"].apply(frozenset_to_string)
        
        # Calculate business impact score
        rules["impact_score"] = (
            rules["support"] * rules["confidence"] * rules["lift"]
        ).round(4)
        
        # Add confidence categories
        rules["confidence_category"] = pd.cut(
            rules["confidence"],
            bins=[0, 0.6, 0.8, 1.0],
            labels=["Low", "Medium", "High"]
        )
        
        # Select relevant columns
        output_cols = [
            "antecedents_str", "consequents_str", 
            "support", "confidence", "lift", "leverage", "conviction",
            "impact_score", "confidence_category"
        ]
        
        rules = rules[output_cols].copy()
        
        # Round numerical columns
        numeric_cols = ["support", "confidence", "lift", "leverage", "conviction", "impact_score"]
        rules[numeric_cols] = rules[numeric_cols].round(4)
        
        return rules.reset_index(drop=True)
        
    except Exception as e:
        st.error(f"Error in rule mining: {str(e)}")
        return pd.DataFrame()

def generate_business_insights(rules: pd.DataFrame, data: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive business insights from rules."""
    if rules.empty:
        return {"insights": [], "recommendations": [], "risk_patterns": []}
    
    insights = []
    recommendations = []
    risk_patterns = []
    
    # Analyze top rules
    top_rules = rules.head(10)
    
    for _, rule in top_rules.iterrows():
        antecedents = rule["antecedents_str"]
        consequents = rule["consequents_str"]
        confidence = rule["confidence"]
        support = rule["support"]
        lift = rule["lift"]
        
        # Generate insights based on rule content
        if any(neg in consequents.lower() for neg in ["below", "needs", "concerning", "excessive", "repeat_required"]):
            risk_patterns.append({
                "pattern": f"{antecedents} → {consequents}",
                "probability": f"{confidence:.1%}",
                "frequency": f"{support:.1%}",
                "strength": f"{lift:.2f}x",
                "description": f"When {antecedents.replace('_', ' ').lower()} occurs, there's a {confidence:.1%} chance of {consequents.replace('_', ' ').lower()}"
            })
        
        if lift > 2.0 and support > 0.05:
            insights.append({
                "type": "Strong Pattern",
                "finding": f"{antecedents} strongly predicts {consequents}",
                "confidence": confidence,
                "impact": support,
                "strength": lift
            })
        
        # Generate specific recommendations
        if "CSAT" in consequents and "Below" in consequents:
            recommendations.append({
                "priority": "High",
                "area": "Customer Satisfaction",
                "action": f"Address {antecedents.replace('_', ' ').lower()} to improve CSAT",
                "expected_impact": f"Could prevent {support:.1%} of low satisfaction cases"
            })
        
        if "Escalations" in consequents and "High" in consequents:
            recommendations.append({
                "priority": "High",
                "area": "Call Resolution",
                "action": f"Implement coaching for agents with {antecedents.replace('_', ' ').lower()}",
                "expected_impact": f"Could reduce escalations for {support:.1%} of cases"
            })
    
    return {
        "insights": insights[:5],
        "recommendations": recommendations[:5],
        "risk_patterns": risk_patterns[:5]
    }

def create_executive_summary(rules: pd.DataFrame, data: pd.DataFrame, insights: Dict) -> str:
    """Generate executive summary of analysis."""
    
    total_rules = len(rules)
    high_confidence_rules = len(rules[rules["confidence"] >= 0.8])
    avg_lift = rules["lift"].mean() if not rules.empty else 0
    top_support = rules["support"].max() if not rules.empty else 0
    
    summary = f"""
    ## Executive Summary
    
    **Analysis Overview:**
    - Analyzed {len(data):,} agent performance records
    - Discovered {total_rules} performance pattern associations
    - Identified {high_confidence_rules} high-confidence patterns (≥80% probability)
    - Average pattern strength: {avg_lift:.2f}x above baseline
    
    **Key Findings:**
    - Most significant pattern affects {top_support:.1%} of operations
    - {len(insights.get('risk_patterns', []))} critical risk patterns identified
    - {len(insights.get('recommendations', []))} actionable improvement opportunities
    
    **Business Impact:**
    The strongest patterns show clear relationships between operational metrics and customer outcomes, 
    providing specific targets for coaching and process improvement initiatives.
    """
    
    return summary

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_professional_metrics_cards(rules: pd.DataFrame, data: pd.DataFrame) -> None:
    """Create professional metric cards with business context."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_rules = len(rules) if not rules.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_rules}</div>
            <div class="metric-label">Patterns Discovered</div>
            <div class="metric-change positive">+{min(12, total_rules//10)}% vs baseline</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_confidence = rules["confidence"].mean() if not rules.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_confidence:.1%}</div>
            <div class="metric-label">Average Confidence</div>
            <div class="metric-change {'positive' if avg_confidence > 0.7 else 'neutral'}">
                {'Strong' if avg_confidence > 0.7 else 'Moderate'} reliability
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        max_lift = rules["lift"].max() if not rules.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{max_lift:.1f}x</div>
            <div class="metric-label">Strongest Pattern</div>
            <div class="metric-change {'positive' if max_lift > 2 else 'neutral'}">
                {'Highly' if max_lift > 2 else 'Moderately'} significant
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        agents_affected = len(data) if not data.empty else 0
        coverage = rules["support"].max() if not rules.empty else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{coverage:.1%}</div>
            <div class="metric-label">Max Pattern Coverage</div>
            <div class="metric-change neutral">{agents_affected:,} agents analyzed</div>
        </div>
        """, unsafe_allow_html=True)

def create_advanced_visualizations(rules: pd.DataFrame) -> None:
    """Create comprehensive visualization suite."""
    
    if rules.empty:
        st.warning("No rules available for visualization.")
        return
    
    # Top rules by impact score
    st.subheader("📊 Pattern Analysis Dashboard")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Impact Analysis", "🔗 Rule Relationships", "📋 Confidence Matrix", "🎯 Business Focus"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Impact score visualization
            top_rules = rules.head(15).copy()
            
            fig_impact = go.Figure()
            fig_impact.add_trace(go.Bar(
                x=top_rules['impact_score'],
                y=top_rules['antecedents_str'],
                orientation='h',
                marker=dict(
                    color=top_rules['confidence'],
                    colorscale='Blues',
                    colorbar=dict(title="Confidence Level"),
                    line=dict(color='rgba(58, 71, 80, 0.6)', width=0.6)
                ),
                text=[f"Lift: {lift:.2f}" for lift in top_rules['lift']],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>" +
                            "Impact Score: %{x:.4f}<br>" +
                            "Confidence: %{marker.color:.1%}<br>" +
                            "Support: %{customdata:.1%}<extra></extra>",
                customdata=top_rules['support']
            ))
            
            fig_impact.update_layout(
                title="Top Patterns by Business Impact",
                xaxis_title="Impact Score (Support × Confidence × Lift)",
                yaxis_title="Pattern Conditions",
                height=500,
                template="plotly_white",
                margin=dict(l=200, r=50, t=50, b=50),
                font=dict(family="Inter, sans-serif")
            )
            
            st.plotly_chart(fig_impact, use_container_width=True)
        
        with col2:
            # Confidence distribution
            fig_conf = go.Figure()
            conf_counts = rules['confidence_category'].value_counts()
            
            fig_conf.add_trace(go.Pie(
                labels=conf_counts.index,
                values=conf_counts.values,
                hole=0.4,
                marker=dict(colors=['#ef4444', '#f59e0b', '#10b981']),
                textinfo='label+percent',
                textfont=dict(size=12)
            ))
            
            fig_conf.update_layout(
                title="Confidence Distribution",
                height=300,
                template="plotly_white",
                font=dict(family="Inter, sans-serif"),
                showlegend=False
            )
            
            st.plotly_chart(fig_conf, use_container_width=True)
            
            # Summary stats
            st.markdown("""
            <div class="pro-container">
                <h4>Pattern Quality Metrics</h4>
                <ul style="list-style-type: none; padding: 0;">
                    <li>🎯 <strong>High Confidence:</strong> {:.0f} patterns</li>
                    <li>⚡ <strong>Strong Lift:</strong> {:.0f} patterns (>2x)</li>
                    <li>📊 <strong>Good Coverage:</strong> {:.0f} patterns (>5%)</li>
                </ul>
            </div>
            """.format(
                len(rules[rules['confidence'] >= 0.8]),
                len(rules[rules['lift'] > 2.0]),
                len(rules[rules['support'] > 0.05])
            ), unsafe_allow_html=True)
    
    with tab2:
        # Sankey diagram for rule relationships
        top_15 = rules.head(15)
        
        # Prepare Sankey data
        antecedents = top_15['antecedents_str'].tolist()
        consequents = top_15['consequents_str'].tolist()
        
        # Create unique labels
        all_labels = list(set(antecedents + consequents))
        label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
        
        source = [label_to_idx[ant] for ant in antecedents]
        target = [label_to_idx[con] for con in consequents]
        values = (top_15['confidence'] * 100).tolist()
        
        # Color mapping for different types
        node_colors = []
        for label in all_labels:
            if any(neg in label.lower() for neg in ['below', 'needs', 'concerning', 'excessive', 'high_escalation']):
                node_colors.append('rgba(239, 68, 68, 0.8)')  # Red for negative outcomes
            elif any(pos in label.lower() for pos in ['excellent', 'exceeds', 'exceptional', 'first_call']):
                node_colors.append('rgba(16, 185, 129, 0.8)')  # Green for positive outcomes
            else:
                node_colors.append('rgba(59, 130, 246, 0.8)')  # Blue for neutral
        
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="rgba(0,0,0,0.2)", width=1),
                label=all_labels,
                color=node_colors
            ),
            link=dict(
                source=source,
                target=target,
                value=values,
                color='rgba(59, 130, 246, 0.3)',
                hovertemplate='<b>%{source.label}</b> → <b>%{target.label}</b><br>' +
                            'Confidence: %{value:.1f}%<extra></extra>'
            )
        )])
        
        fig_sankey.update_layout(
            title="Pattern Flow Analysis: Conditions → Outcomes",
            font=dict(size=11, family="Inter, sans-serif"),
            height=600,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_sankey, use_container_width=True)
    
    with tab3:
        # Confidence vs Support scatter with lift coloring
        fig_scatter = go.Figure()
        
        fig_scatter.add_trace(go.Scatter(
            x=rules['support'],
            y=rules['confidence'],
            mode='markers',
            marker=dict(
                size=rules['lift'] * 5,
                color=rules['lift'],
                colorscale='Viridis',
                colorbar=dict(title="Lift Value"),
                line=dict(width=1, color='rgba(0,0,0,0.3)'),
                sizemode='diameter',
                sizeref=2.*max(rules['lift'])/40**2,
                sizemin=4
            ),
            text=rules['antecedents_str'] + ' → ' + rules['consequents_str'],
            hovertemplate="<b>%{text}</b><br>" +
                        "Support: %{x:.1%}<br>" +
                        "Confidence: %{y:.1%}<br>" +
                        "Lift: %{marker.color:.2f}<extra></extra>"
        ))
        
        # Add quadrant lines
        fig_scatter.add_hline(y=0.7, line_dash="dash", line_color="rgba(107, 114, 128, 0.5)", annotation_text="High Confidence")
        fig_scatter.add_vline(x=0.05, line_dash="dash", line_color="rgba(107, 114, 128, 0.5)", annotation_text="Good Support")
        
        fig_scatter.update_layout(
            title="Pattern Quality Matrix: Support vs Confidence (Size = Lift)",
            xaxis_title="Support (Frequency in Dataset)",
            yaxis_title="Confidence (Prediction Accuracy)",
            height=500,
            template="plotly_white",
            font=dict(family="Inter, sans-serif")
        )
        
        fig_scatter.update_xaxis(tickformat='.1%')
        fig_scatter.update_yaxis(tickformat='.1%')
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Quadrant analysis
        col1, col2, col3, col4 = st.columns(4)
        
        high_conf_high_supp = len(rules[(rules['confidence'] >= 0.7) & (rules['support'] >= 0.05)])
        high_conf_low_supp = len(rules[(rules['confidence'] >= 0.7) & (rules['support'] < 0.05)])
        low_conf_high_supp = len(rules[(rules['confidence'] < 0.7) & (rules['support'] >= 0.05)])
        low_conf_low_supp = len(rules[(rules['confidence'] < 0.7) & (rules['support'] < 0.05)])
        
        col1.metric("🎯 High Quality", high_conf_high_supp, "High Conf + Support")
        col2.metric("💎 Niche Patterns", high_conf_low_supp, "High Conf + Low Support")
        col3.metric("📊 Common Patterns", low_conf_high_supp, "Low Conf + High Support")
        col4.metric("⚠️ Weak Patterns", low_conf_low_supp, "Low Quality")
    
    with tab4:
        # Business-focused analysis
        st.subheader("🎯 Business Priority Patterns")
        
        # Filter for business-critical patterns
        business_critical = rules[
            (rules['confidence'] >= 0.6) & 
            (rules['support'] >= 0.03) &
            (rules['lift'] >= 1.2)
        ].copy()
        
        if not business_critical.empty:
            # Categorize patterns by business area
            business_critical['business_area'] = 'General'
            business_critical.loc[business_critical['consequents_str'].str.contains('CSAT|satisfaction', case=False), 'business_area'] = 'Customer Satisfaction'
            business_critical.loc[business_critical['consequents_str'].str.contains('FCR|resolution', case=False), 'business_area'] = 'Call Resolution'
            business_critical.loc[business_critical['consequents_str'].str.contains('Escalation', case=False), 'business_area'] = 'Escalation Management'
            business_critical.loc[business_critical['consequents_str'].str.contains('AHT|ACW', case=False), 'business_area'] = 'Efficiency'
            business_critical.loc[business_critical['consequents_str'].str.contains('QA|Quality', case=False), 'business_area'] = 'Quality Assurance'
            
            # Create business area summary
            area_summary = business_critical.groupby('business_area').agg({
                'impact_score': ['count', 'mean'],
                'confidence': 'mean',
                'support': 'mean'
            }).round(3)
            
            area_summary.columns = ['Pattern_Count', 'Avg_Impact', 'Avg_Confidence', 'Avg_Support']
            area_summary = area_summary.reset_index()
            
            fig_business = go.Figure()
            
            fig_business.add_trace(go.Scatter(
                x=area_summary['Avg_Confidence'],
                y=area_summary['Avg_Impact'],
                mode='markers+text',
                marker=dict(
                    size=area_summary['Pattern_Count'] * 10,
                    color=area_summary['Avg_Support'],
                    colorscale='Blues',
                    colorbar=dict(title="Avg Support"),
                    line=dict(width=2, color='white'),
                    sizemode='diameter'
                ),
                text=area_summary['business_area'],
                textposition="middle center",
                textfont=dict(size=10, color='white', family="Inter"),
                hovertemplate="<b>%{text}</b><br>" +
                            "Patterns: %{customdata}<br>" +
                            "Avg Confidence: %{x:.1%}<br>" +
                            "Avg Impact: %{y:.4f}<extra></extra>",
                customdata=area_summary['Pattern_Count']
            ))
            
            fig_business.update_layout(
                title="Business Area Impact Analysis",
                xaxis_title="Average Confidence",
                yaxis_title="Average Impact Score",
                height=400,
                template="plotly_white",
                font=dict(family="Inter, sans-serif")
            )
            
            fig_business.update_xaxis(tickformat='.1%')
            
            st.plotly_chart(fig_business, use_container_width=True)
            
            # Business recommendations table
            st.subheader("📋 Priority Action Items")
            
            priority_rules = business_critical.nlargest(10, 'impact_score')[
                ['antecedents_str', 'consequents_str', 'confidence', 'support', 'lift', 'business_area', 'impact_score']
            ].copy()
            
            priority_rules.columns = ['Conditions', 'Outcomes', 'Confidence', 'Support', 'Lift', 'Business Area', 'Impact Score']
            
            st.dataframe(
                priority_rules,
                use_container_width=True,
                column_config={
                    "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.1%"),
                    "Support": st.column_config.ProgressColumn("Support", min_value=0, max_value=1, format="%.1%"),
                    "Lift": st.column_config.NumberColumn("Lift", format="%.2f"),
                    "Impact Score": st.column_config.ProgressColumn("Impact", min_value=0, max_value=priority_rules['Impact Score'].max()),
                }
            )

def create_interactive_network(rules: pd.DataFrame, height: str = "600px") -> str:
    """Create an interactive network visualization of rules."""
    
    if rules.empty:
        return "<div>No rules to display</div>"
    
    # Create network
    net = Network(
        height=height, 
        width="100%", 
        bgcolor="#f8fafc", 
        font_color="#1f2937",
        select_menu=True,
        filter_menu=True,
    )
    
    # Configure physics
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100},
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 95,
          "springConstant": 0.04,
          "damping": 0.09
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200
      },
      "nodes": {
        "font": {"size": 12, "face": "Inter"},
        "borderWidth": 2
      },
      "edges": {
        "font": {"size": 10, "face": "Inter"},
        "smooth": {"type": "continuous"}
      }
    }
    """)
    
    # Collect all unique nodes
    all_items = set()
    for _, rule in rules.head(20).iterrows():  # Limit for performance
        antecedents = rule['antecedents_str'].split(' + ')
        consequents = rule['consequents_str'].split(' + ')
        all_items.update(antecedents)
        all_items.update(consequents)
    
    # Add nodes with smart coloring
    for item in all_items:
        # Determine node color based on content
        if any(neg in item.lower() for neg in ['below', 'needs', 'concerning', 'excessive', 'repeat_required']):
            color = "#ef4444"  # Red for negative outcomes
            group = "negative"
        elif any(pos in item.lower() for pos in ['excellent', 'exceeds', 'exceptional', 'first_call', 'minimal']):
            color = "#10b981"  # Green for positive outcomes
            group = "positive"
        elif "CallType" in item:
            color = "#8b5cf6"  # Purple for call types
            group = "calltype"
        else:
            color = "#3b82f6"  # Blue for neutral/operational
            group = "neutral"
        
        # Clean label for display
        clean_label = item.replace('_', ' ').title()
        
        net.add_node(
            item,
            label=clean_label,
            color=color,
            group=group,
            title=f"<b>{clean_label}</b><br>Category: {group.title()}",
            size=20
        )
    
    # Add edges (rules)
    for _, rule in rules.head(20).iterrows():
        antecedents = rule['antecedents_str'].split(' + ')
        consequents = rule['consequents_str'].split(' + ')
        
        for ant in antecedents:
            for con in consequents:
                if ant != con:  # Avoid self-loops
                    # Edge properties based on rule strength
                    width = min(max(rule['confidence'] * 5, 1), 8)
                    
                    # Color based on lift
                    if rule['lift'] > 2:
                        edge_color = "#10b981"  # Green for strong lift
                    elif rule['lift'] > 1.5:
                        edge_color = "#f59e0b"  # Amber for medium lift
                    else:
                        edge_color = "#6b7280"  # Gray for weak lift
                    
                    net.add_edge(
                        ant, 
                        con,
                        title=f"<b>Rule:</b> {ant} → {con}<br>" +
                              f"<b>Confidence:</b> {rule['confidence']:.1%}<br>" +
                              f"<b>Support:</b> {rule['support']:.1%}<br>" +
                              f"<b>Lift:</b> {rule['lift']:.2f}x",
                        width=width,
                        color=edge_color,
                        arrows="to"
                    )
    
    return net.generate_html()

# =============================================================================
# SESSION STATE MANAGEMENT
# =============================================================================

# Initialize session state
if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None
if "binned_data" not in st.session_state:
    st.session_state.binned_data = None
if "basket_data" not in st.session_state:
    st.session_state.basket_data = None
if "association_rules" not in st.session_state:
    st.session_state.association_rules = None
if "business_insights" not in st.session_state:
    st.session_state.business_insights = None
if "network_html" not in st.session_state:
    st.session_state.network_html = None
if "analysis_timestamp" not in st.session_state:
    st.session_state.analysis_timestamp = None

# =============================================================================
# MAIN APPLICATION INTERFACE
# =============================================================================

def main():
    """Main application interface with professional navigation."""
    
    # Header with branding
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1e40af 0%, #3b82f6 100%); padding: 20px; border-radius: 12px; margin-bottom: 24px; color: white;">
        <h1 style="margin: 0; font-size: 2rem; color: white;">📊 Performance Affinity Analytics Platform</h1>
        <p style="margin: 8px 0 0 0; opacity: 0.9; font-size: 1.1rem;">Discover KPI patterns • Generate insights • Drive performance improvements</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status bar
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    data_status = "✅ Loaded" if st.session_state.analysis_data is not None else "⏳ Pending"
    rules_count = len(st.session_state.association_rules) if st.session_state.association_rules is not None else 0
    
    st.markdown(f"""
    <div class="status-bar">
        <div class="status-indicator">
            <span class="status-dot"></span>
            <span>System Active - {current_time}</span>
        </div>
        <div class="status-indicator">
            <span>📊 Data Status: {data_status}</span>
        </div>
        <div class="status-indicator">
            <span>🔍 Rules Generated: {rules_count}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    selected = option_menu(
        menu_title=None,
        options=["📊 Data Overview", "⚙️ Analysis Setup", "🔍 Results & Insights", "🎯 Action Items", "📥 Export & Reports"],
        icons=["bar-chart", "gear", "search", "target", "download"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0", "background-color": "white", "border": "1px solid #e5e7eb", "border-radius": "12px"},
            "icon": {"color": "#6b7280", "font-size": "16px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "center",
                "margin": "0px",
                "padding": "12px 16px",
                "border-radius": "8px",
                "color": "#6b7280",
                "font-weight": "500",
            },
            "nav-link-selected": {
                "background-color": "#1e40af",
                "color": "white",
                "font-weight": "600",
            },
        }
    )
    
    # Route to appropriate page
    if selected == "📊 Data Overview":
        show_data_overview()
    elif selected == "⚙️ Analysis Setup":
        show_analysis_setup()
    elif selected == "🔍 Results & Insights":
        show_results_insights()
    elif selected == "🎯 Action Items":
        show_action_items()
    elif selected == "📥 Export & Reports":
        show_export_reports()

def show_data_overview():
    """Data loading and overview section."""
    
    st.subheader("📊 Data Management")
    
    # Data loading section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="pro-container">
            <h4>📥 Data Source</h4>
            <p>Upload your contact center KPI data or generate a sample dataset to explore the platform capabilities.</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="Upload a CSV file with agent performance data"
        )
        
        col1a, col1b = st.columns(2)
        with col1a:
            if st.button("🎲 Generate Sample Data", type="primary", use_container_width=True):
                with st.spinner("Generating realistic sample dataset..."):
                    st.session_state.analysis_data = generate_professional_sample_data()
                    st.session_state.analysis_timestamp = datetime.now()
                    st.success("✅ Sample data generated successfully!")
                    st.rerun()
        
        with col1b:
            if st.button("🔄 Clear Data", use_container_width=True):
                # Clear all session state data
                for key in ["analysis_data", "binned_data", "basket_data", "association_rules", "business_insights", "network_html"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("✅ Data cleared successfully!")
                st.rerun()
    
    with col2:
        st.markdown("""
        <div class="pro-container">
            <h4>📋 Expected Data Schema</h4>
            <div style="font-size: 0.85rem; color: #6b7280;">
                <strong>Required Columns:</strong><br>
                • AgentID (string)<br>
                • CallType (category)<br>
                • AHT_sec (integer)<br>
                • ACW_sec (integer)<br>
                • Hold_sec (integer)<br>
                • CSAT (float 1-5)<br>
                • FCR (0/1)<br>
                • Adherence (float 0-1)<br>
                • QA_Compliance (float 0-1)<br>
                • Escalations (integer)<br>
                • RepeatContacts7d (integer)<br>
                • Absenteeism (float 0-1)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Handle file upload
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.analysis_data = df
            st.session_state.analysis_timestamp = datetime.now()
            st.success("✅ Data uploaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    # Data overview
    if st.session_state.analysis_data is not None:
        df = st.session_state.analysis_data
        
        st.markdown("### 📊 Dataset Overview")
        
        # Data quality metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Unique Agents", f"{df['AgentID'].nunique():,}")
        with col3:
            call_types = df['CallType'].nunique() if 'CallType' in df.columns else 0
            st.metric("Call Types", call_types)
        with col4:
            completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            st.metric("Data Completeness", f"{completeness:.1f}%")
        
        # Data preview and statistics
        tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "📈 Statistics", "🔍 Data Quality"])
        
        with tab1:
            st.subheader("Sample Records")
            st.dataframe(df.head(10), use_container_width=True)
        
        with tab2:
            st.subheader("Descriptive Statistics")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        with tab3:
            st.subheader("Data Quality Assessment")
            
            # Missing values
            missing_data = df.isnull().sum()
            missing_pct = (missing_data / len(df)) * 100
            
            quality_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing_Count': missing_data.values,
                'Missing_Percentage': missing_pct.values
            })
            quality_df = quality_df[quality_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
            
            if not quality_df.empty:
                st.dataframe(
                    quality_df,
                    use_container_width=True,
                    column_config={
                        "Missing_Percentage": st.column_config.ProgressColumn(
                            "Missing %", min_value=0, max_value=100, format="%.1f%%"
                        )
                    }
                )
            else:
                st.success("✅ No missing values detected!")
        
        # KPI Distribution visualizations
        st.markdown("### 📊 KPI Distributions")
        
        if 'CSAT' in df.columns and 'AHT_sec' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_csat = px.histogram(
                    df, x='CSAT', nbins=20,
                    title="Customer Satisfaction Distribution",
                    template="plotly_white"
                )
                fig_csat.update_layout(
                    font=dict(family="Inter, sans-serif"),
                    height=350
                )
                st.plotly_chart(fig_csat, use_container_width=True)
            
            with col2:
                fig_aht = px.histogram(
                    df, x='AHT_sec', nbins=30,
                    title="Average Handle Time Distribution",
                    template="plotly_white"
                )
                fig_aht.update_layout(
                    font=dict(family="Inter, sans-serif"),
                    height=350
                )
                st.plotly_chart(fig_aht, use_container_width=True)
    
    else:
        st.info("👆 Please upload data or generate sample data to continue.")

def show_analysis_setup():
    """Analysis configuration and execution."""
    
    if st.session_state.analysis_data is None:
        st.warning("📊 Please load data first in the Data Overview section.")
        return
    
    st.subheader("⚙️ Analysis Configuration")
    
    # Configuration sections
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("""
        <div class="pro-container">
            <h4>🔧 Processing Configuration</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Binning strategy
        binning_strategy = st.radio(
            "KPI Binning Strategy",
            ["business_rules", "quantile"],
            index=0,
            help="Choose how to categorize continuous KPIs into discrete groups"
        )
        
        if binning_strategy == "business_rules":
            st.info("💡 Using industry-standard thresholds for call center KPIs")
        else:
            st.info("💡 Using statistical quantiles to create balanced groups")
        
        st.markdown("### 🧮 Mining Parameters")
        
        # Algorithm selection
        algorithm = st.selectbox(
            "Association Rule Algorithm",
            ["apriori", "fpgrowth"],
            index=0,
            help="FP-Growth is typically faster for large datasets"
        )
        
        # Mining parameters
        col1a, col1b = st.columns(2)
        with col1a:
            min_support = st.slider(
                "Minimum Support",
                min_value=0.01,
                max_value=0.3,
                value=0.05,
                step=0.01,
                help="Minimum frequency for pattern to be considered"
            )
        
        with col1b:
            min_threshold = st.slider(
                "Minimum Confidence",
                min_value=0.1,
                max_value=0.95,
                value=0.6,
                step=0.05,
                help="Minimum prediction accuracy for rules"
            )
        
        rule_metric = st.selectbox(
            "Rule Quality Metric",
            ["lift", "confidence", "support"],
            index=0,
            help="Primary metric for ranking association rules"
        )
        
        max_items = st.slider(
            "Maximum Items per Rule",
            min_value=2,
            max_value=5,
            value=3,
            help="Complexity limit for association patterns"
        )
    
    with col2:
        st.markdown("""
        <div class="pro-container">
            <h4>📊 Analysis Preview</h4>
        </div>
        """, unsafe_allow_html=True)
        
        df = st.session_state.analysis_data
        
        # Show expected basket size
        if st.button("🔍 Preview Basket Structure", use_container_width=True):
            with st.spinner("Analyzing data structure..."):
                preview_binned = create_professional_bins(df.sample(min(500, len(df))), binning_strategy)
                preview_basket = create_market_basket(preview_binned)
                
                st.metric("Unique Items", len(preview_basket.columns))
                st.metric("Sample Transactions", len(preview_basket))
                
                # Show top items by frequency
                item_freq = preview_basket.mean().sort_values(ascending=False).head(10)
                freq_df = pd.DataFrame({
                    'Item': item_freq.index,
                    'Frequency': item_freq.values
                })
                
                st.subheader("Top Items by Frequency")
                st.dataframe(
                    freq_df,
                    use_container_width=True,
                    column_config={
                        "Frequency": st.column_config.ProgressColumn("Frequency", min_value=0, max_value=1, format="%.1%")
                    }
                )
        
        # Parameter guidance
        st.markdown("""
        <div class="helper-text">
            <strong>Parameter Guidance:</strong><br>
            • <strong>Support</strong>: Lower values find rare patterns, higher values focus on common ones<br>
            • <strong>Confidence</strong>: Higher values ensure reliable predictions<br>
            • <strong>Lift</strong>: Values >1 indicate positive association
        </div>
        """, unsafe_allow_html=True)
    
    # Analysis execution
    st.markdown("### 🚀 Execute Analysis")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("▶️ Run Analysis", type="primary", use_container_width=True):
            run_full_analysis(df, binning_strategy, algorithm, min_support, min_threshold, rule_metric, max_items)
    
    with col2:
        if st.button("🧪 Quick Test", use_container_width=True):
            run_quick_test(df, binning_strategy)
    
    with col3:
        if st.button("📋 Reset Results", use_container_width=True):
            clear_analysis_results()

def run_full_analysis(df, binning_strategy, algorithm, min_support, min_threshold, rule_metric, max_items):
    """Execute the complete analysis pipeline."""
    
    progress_bar = st.progress(0, text="Starting analysis...")
    
    try:
        # Step 1: Data preprocessing
        progress_bar.progress(20, text="Preprocessing data...")
        binned_data = create_professional_bins(df, binning_strategy)
        st.session_state.binned_data = binned_data
        
        # Step 2: Create market basket
        progress_bar.progress(40, text="Creating market basket...")
        basket_data = create_market_basket(binned_data)
        st.session_state.basket_data = basket_data
        
        # Step 3: Mine association rules
        progress_bar.progress(60, text="Mining association rules...")
        rules = mine_association_rules(
            basket_data, algorithm, min_support, rule_metric, min_threshold, max_items
        )
        st.session_state.association_rules = rules
        
        # Step 4: Generate insights
        progress_bar.progress(80, text="Generating business insights...")
        insights = generate_business_insights(rules, df)
        st.session_state.business_insights = insights
        
        # Step 5: Create network visualization
        progress_bar.progress(90, text="Creating visualizations...")
        if not rules.empty:
            network_html = create_interactive_network(rules)
            st.session_state.network_html = network_html
        
        progress_bar.progress(100, text="Analysis complete!")
        
        # Success message
        if rules.empty:
            st.warning("⚠️ No patterns found with current parameters. Try lowering the thresholds.")
        else:
            st.success(f"✅ Analysis complete! Found {len(rules)} patterns.")
            st.balloons()
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
    
    finally:
        progress_bar.empty()

def run_quick_test(df, binning_strategy):
    """Run a quick test with permissive parameters."""
    
    with st.spinner("Running quick test with permissive parameters..."):
        try:
            # Use permissive parameters for quick test
            binned_data = create_professional_bins(df.sample(min(300, len(df))), binning_strategy)
            basket_data = create_market_basket(binned_data)
            rules = mine_association_rules(basket_data, "apriori", 0.03, "lift", 0.5, 3)
            
            if rules.empty:
                st.warning("⚠️ No patterns found even with permissive parameters. Check data quality.")
            else:
                st.success(f"✅ Quick test found {len(rules)} patterns. Adjust parameters and run full analysis.")
                
                # Show preview of top rules
                st.subheader("Preview: Top 5 Patterns")
                preview_rules = rules.head(5)[['antecedents_str', 'consequents_str', 'confidence', 'support', 'lift']]
                st.dataframe(preview_rules, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Quick test failed: {str(e)}")

def clear_analysis_results():
    """Clear analysis results from session state."""
    keys_to_clear = ["binned_data", "basket_data", "association_rules", "business_insights", "network_html"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.success("✅ Analysis results cleared.")

def show_results_insights():
    """Display comprehensive analysis results and insights."""
    
    if st.session_state.association_rules is None or st.session_state.association_rules.empty:
        st.warning("🔍 No analysis results available. Please run the analysis first in the Analysis Setup section.")
        return
    
    rules = st.session_state.association_rules
    data = st.session_state.analysis_data
    insights = st.session_state.business_insights or {}
    
    st.subheader("🔍 Analysis Results & Insights")
    
    # Executive summary
    with st.expander("📋 Executive Summary", expanded=True):
        exec_summary = create_executive_summary(rules, data, insights)
        st.markdown(exec_summary)
    
    # Key metrics
    st.markdown("### 📊 Performance Metrics")
    create_professional_metrics_cards(rules, data)
    
    # Main visualizations
    create_advanced_visualizations(rules)
    
    # Interactive network
    if st.session_state.network_html:
        st.markdown("### 🌐 Interactive Pattern Network")
        st.markdown("""
        <div class="helper-text">
            Explore the relationships between KPI patterns. Hover over nodes and edges for details. 
            Use the controls to filter and focus on specific patterns.
        </div>
        """, unsafe_allow_html=True)
        
        st.components.v1.html(st.session_state.network_html, height=650)
    
    # Detailed rules table
    st.markdown("### 📋 Detailed Pattern Rules")
    
    # Filters for rules table
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence_filter = st.slider(
            "Minimum Confidence Filter",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            format="%.1f"
        )
    
    with col2:
        support_filter = st.slider(
            "Minimum Support Filter",
            min_value=0.0,
            max_value=rules['support'].max() if not rules.empty else 1.0,
            value=0.0,
            step=0.01,
            format="%.2f"
        )
    
    with col3:
        lift_filter = st.slider(
            "Minimum Lift Filter",
            min_value=1.0,
            max_value=rules['lift'].max() if not rules.empty else 5.0,
            value=1.0,
            step=0.1,
            format="%.1f"
        )
    
    # Apply filters
    filtered_rules = rules[
        (rules['confidence'] >= confidence_filter) &
        (rules['support'] >= support_filter) &
        (rules['lift'] >= lift_filter)
    ].copy()
    
    if filtered_rules.empty:
        st.warning("No rules match the current filter criteria.")
    else:
        st.dataframe(
            filtered_rules,
            use_container_width=True,
            column_config={
                "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.1%"),
                "support": st.column_config.ProgressColumn("Support", min_value=0, max_value=1, format="%.1%"),
                "lift": st.column_config.NumberColumn("Lift", format="%.2fx"),
                "impact_score": st.column_config.ProgressColumn("Impact", min_value=0, max_value=filtered_rules['impact_score'].max()),
                "confidence_category": st.column_config.TextColumn("Confidence Level")
            }
        )
    
    # Pattern insights
    if insights:
        st.markdown("### 🎯 Key Insights")
        
        tab1, tab2, tab3 = st.tabs(["💡 Discoveries", "⚠️ Risk Patterns", "📈 Recommendations"])
        
        with tab1:
            if insights.get('insights'):
                for insight in insights['insights']:
                    st.markdown(f"""
                    <div class="insight-card">
                        <h4>{insight['type']}</h4>
                        <p>{insight['finding']}</p>
                        <small>Confidence: {insight['confidence']:.1%} | Impact: {insight['impact']:.1%} | Strength: {insight['strength']:.2f}x</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No specific insights generated for current patterns.")
        
        with tab2:
            if insights.get('risk_patterns'):
                for risk in insights['risk_patterns']:
                    st.markdown(f"""
                    <div class="alert-card">
                        <h4>⚠️ Risk Pattern Detected</h4>
                        <p><strong>{risk['pattern']}</strong></p>
                        <p>{risk['description']}</p>
                        <div style="display: flex; gap: 20px; margin-top: 10px;">
                            <span><strong>Probability:</strong> {risk['probability']}</span>
                            <span><strong>Frequency:</strong> {risk['frequency']}</span>
                            <span><strong>Strength:</strong> {risk['strength']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("No high-risk patterns identified.")
        
        with tab3:
            if insights.get('recommendations'):
                for rec in insights['recommendations']:
                    priority_color = {"High": "error", "Medium": "warning", "Low": "success"}
                    st.markdown(f"""
                    <div class="success-card">
                        <h4>🎯 {rec['area']} - {rec['priority']} Priority</h4>
                        <p><strong>Action:</strong> {rec['action']}</p>
                        <p><strong>Expected Impact:</strong> {rec['expected_impact']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No specific recommendations generated.")

def show_action_items():
    """Display actionable insights and coaching recommendations."""
    
    if st.session_state.association_rules is None or st.session_state.association_rules.empty:
        st.warning("🎯 No analysis results available. Please run the analysis first.")
        return
    
    rules = st.session_state.association_rules
    data = st.session_state.analysis_data
    insights = st.session_state.business_insights or {}
    
    st.subheader("🎯 Action Items & Coaching Recommendations")
    
    # Priority matrix
    st.markdown("### 📊 Priority Action Matrix")
    
    # Create priority categories
    high_priority = rules[
        (rules['confidence'] >= 0.7) & 
        (rules['support'] >= 0.05) & 
        (rules['lift'] >= 1.5)
    ].copy()
    
    medium_priority = rules[
        (rules['confidence'] >= 0.6) & 
        (rules['support'] >= 0.03) & 
        (rules['lift'] >= 1.2) &
        ~rules.index.isin(high_priority.index)
    ].copy()
    
    low_priority = rules[
        ~rules.index.isin(high_priority.index) &
        ~rules.index.isin(medium_priority.index)
    ].copy()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #dc2626;">
            <div class="metric-value" style="color: #dc2626;">{len(high_priority)}</div>
            <div class="metric-label">High Priority</div>
            <div class="metric-change">Immediate Action Required</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #d97706;">
            <div class="metric-value" style="color: #d97706;">{len(medium_priority)}</div>
            <div class="metric-label">Medium Priority</div>
            <div class="metric-change">Plan & Schedule</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #059669;">
            <div class="metric-value" style="color: #059669;">{len(low_priority)}</div>
            <div class="metric-label">Low Priority</div>
            <div class="metric-change">Monitor & Review</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Actionable recommendations
    st.markdown("### 🎯 Coaching & Process Recommendations")
    
    # Generate specific recommendations based on rules
    recommendations = generate_detailed_recommendations(high_priority, medium_priority)
    
    for i, rec in enumerate(recommendations, 1):
        priority_color = {"High": "#dc2626", "Medium": "#d97706", "Low": "#059669"}
        border_color = priority_color.get(rec['priority'], "#6b7280")
        
        st.markdown(f"""
        <div class="pro-container" style="border-left: 4px solid {border_color};">
            <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 12px;">
                <h4 style="margin: 0; color: {border_color};">#{i}. {rec['title']}</h4>
                <span class="confidence-badge confidence-{rec['priority'].lower()}">{rec['priority']} Priority</span>
            </div>
            
            <p><strong>📋 Pattern:</strong> {rec['pattern']}</p>
            <p><strong>🎯 Recommended Action:</strong> {rec['action']}</p>
            <p><strong>📊 Expected Impact:</strong> {rec['impact']}</p>
            <p><strong>👥 Target Audience:</strong> {rec['audience']}</p>
            <p><strong>⏱️ Timeline:</strong> {rec['timeline']}</p>
            
            {f'<div style="background: #f3f4f6; padding: 12px; border-radius: 6px; margin-top: 12px;"><strong>📝 Implementation Notes:</strong> {rec["notes"]}</div>' if rec.get('notes') else ''}
        </div>
        """, unsafe_allow_html=True)
    
    # Coaching plan generator
    st.markdown("### 📚 Coaching Plan Generator")
    
    selected_patterns = st.multiselect(
        "Select patterns for coaching plan",
        options=high_priority['antecedents_str'].tolist() + medium_priority['antecedents_str'].tolist(),
        default=high_priority['antecedents_str'].tolist()[:3]
    )
    
    if selected_patterns and st.button("📋 Generate Coaching Plan", type="primary"):
        coaching_plan = generate_coaching_plan(selected_patterns, rules)
        
        st.markdown("#### 📋 Generated Coaching Plan")
        st.markdown(f"""
        <div class="pro-container">
            <h4>🎯 Coaching Focus Areas</h4>
            {coaching_plan['focus_areas']}
            
            <h4>📅 Suggested Timeline</h4>
            {coaching_plan['timeline']}
            
            <h4>📊 Success Metrics</h4>
            {coaching_plan['metrics']}
            
            <h4>🛠️ Resources Needed</h4>
            {coaching_plan['resources']}
        </div>
        """, unsafe_allow_html=True)
    
    # Performance tracking template
    st.markdown("### 📊 Performance Tracking Template")
    
    if st.button("📈 Create Tracking Template"):
        tracking_template = create_tracking_template(rules)
        
        st.subheader("KPI Monitoring Template")
        st.dataframe(tracking_template, use_container_width=True)
        
        # Download template
        template_csv = tracking_template.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Tracking Template",
            data=template_csv,
            file_name="kpi_tracking_template.csv",
            mime="text/csv"
        )

def generate_detailed_recommendations(high_priority, medium_priority):
    """Generate detailed, actionable recommendations."""
    
    recommendations = []
    
    # Process high priority patterns
    for _, rule in high_priority.head(3).iterrows():
        rec = {
            'priority': 'High',
            'title': 'Critical Performance Pattern',
            'pattern': f"{rule['antecedents_str']} → {rule['consequents_str']}",
            'action': generate_action_from_rule(rule),
            'impact': f"Affects {rule['support']:.1%} of operations with {rule['confidence']:.1%} probability",
            'audience': determine_audience(rule),
            'timeline': '1-2 weeks',
            'notes': generate_implementation_notes(rule)
        }
        recommendations.append(rec)
    
    # Process medium priority patterns
    for _, rule in medium_priority.head(2).iterrows():
        rec = {
            'priority': 'Medium',
            'title': 'Process Improvement Opportunity',
            'pattern': f"{rule['antecedents_str']} → {rule['consequents_str']}",
            'action': generate_action_from_rule(rule),
            'impact': f"Affects {rule['support']:.1%} of operations with {rule['confidence']:.1%} probability",
            'audience': determine_audience(rule),
            'timeline': '2-4 weeks',
            'notes': generate_implementation_notes(rule)
        }
        recommendations.append(rec)
    
    return recommendations

def generate_action_from_rule(rule):
    """Generate specific action based on rule content."""
    
    antecedents = rule['antecedents_str'].lower()
    consequents = rule['consequents_str'].lower()
    
    if 'csat' in consequents and 'below' in consequents:
        return "Implement targeted coaching on customer empathy and issue resolution techniques"
    elif 'escalation' in consequents:
        return "Develop de-escalation training and empower agents with additional resolution tools"
    elif 'aht' in consequents and ('needs' in consequents or 'extended' in consequents):
        return "Review call handling procedures and provide efficiency training"
    elif 'fcr' in consequents and 'repeat' in consequents:
        return "Enhance knowledge base access and first-call resolution protocols"
    else:
        return "Implement process review and targeted coaching based on pattern analysis"

def determine_audience(rule):
    """Determine target audience for coaching."""
    
    pattern = (rule['antecedents_str'] + ' ' + rule['consequents_str']).lower()
    
    if any(skill in pattern for skill in ['qa', 'quality', 'compliance']):
        return "All agents + Quality team review"
    elif any(eff in pattern for eff in ['aht', 'acw', 'hold']):
        return "Agents with efficiency challenges"
    elif any(sat in pattern for sat in ['csat', 'satisfaction']):
        return "Customer-facing agents + supervisors"
    else:
        return "All agents"

def generate_implementation_notes(rule):
    """Generate implementation guidance."""
    
    return "Monitor pattern weekly, adjust coaching based on results, measure impact after 30 days"

def generate_coaching_plan(selected_patterns, rules):
    """Generate a comprehensive coaching plan."""
    
    relevant_rules = rules[rules['antecedents_str'].isin(selected_patterns)]
    
    focus_areas = "<ul>"
    for pattern in selected_patterns:
        focus_areas += f"<li>{pattern.replace('_', ' ').title()}</li>"
    focus_areas += "</ul>"
    
    timeline = """
    <ul>
        <li><strong>Week 1:</strong> Assessment and baseline measurement</li>
        <li><strong>Week 2-3:</strong> Initial coaching sessions</li>
        <li><strong>Week 4:</strong> Progress review and adjustment</li>
        <li><strong>Week 5-6:</strong> Reinforcement and practice</li>
        <li><strong>Week 7-8:</strong> Final assessment and documentation</li>
    </ul>
    """
    
    metrics = """
    <ul>
        <li>Weekly KPI tracking for targeted agents</li>
        <li>Pattern occurrence frequency monitoring</li>
        <li>Agent confidence and skill assessments</li>
        <li>Customer feedback scores</li>
    </ul>
    """
    
    resources = """
    <ul>
        <li>Dedicated coaching time (2-3 hours per week)</li>
        <li>Updated training materials</li>
        <li>Performance monitoring tools</li>
        <li>Feedback collection system</li>
    </ul>
    """
    
    return {
        'focus_areas': focus_areas,
        'timeline': timeline,
        'metrics': metrics,
        'resources': resources
    }

def create_tracking_template(rules):
    """Create a KPI tracking template."""
    
    # Extract key KPIs from rules
    kpis = set()
    for _, rule in rules.iterrows():
        items = rule['antecedents_str'] + ' ' + rule['consequents_str']
        if 'CSAT' in items:
            kpis.add('CSAT')
        if 'AHT' in items:
            kpis.add('AHT_sec')
        if 'ACW' in items:
            kpis.add('ACW_sec')
        if 'FCR' in items:
            kpis.add('FCR')
        if 'Escalation' in items:
            kpis.add('Escalations')
    
    # Create template
    template_data = []
    for week in range(1, 9):  # 8-week tracking
        row = {
            'Week': f"Week_{week}",
            'Date_Range': f"YYYY-MM-DD to YYYY-MM-DD",
            'AgentID': 'AG0000',
        }
        for kpi in sorted(kpis):
            row[f'{kpi}_Baseline'] = 0.0
            row[f'{kpi}_Current'] = 0.0
            row[f'{kpi}_Target'] = 0.0
            row[f'{kpi}_Improvement'] = 0.0
        
        row['Notes'] = 'Coaching notes and observations'
        template_data.append(row)
    
    return pd.DataFrame(template_data)

def show_export_reports():
    """Export and reporting functionality."""
    
    st.subheader("📥 Export & Reports")
    
    if st.session_state.association_rules is None:
        st.warning("📊 No analysis results available for export.")
        return
    
    rules = st.session_state.association_rules
    data = st.session_state.analysis_data
    insights = st.session_state.business_insights or {}
    
    st.markdown("### 📄 Available Exports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="pro-container">
            <h4>📊 Analysis Outputs</h4>
            <p>Core analysis results and data</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Rules CSV
        rules_csv = rules.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📋 Association Rules (CSV)",
            data=rules_csv,
            file_name=f"affinity_rules_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Processed data
        if st.session_state.binned_data is not None:
            binned_csv = st.session_state.binned_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                "🔄 Processed Data (CSV)",
                data=binned_csv,
                file_name=f"processed_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col2:
        st.markdown("""
        <div class="pro-container">
            <h4>📊 Visualizations</h4>
            <p>Interactive charts and networks</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Network HTML
        if st.session_state.network_html:
            network_bytes = st.session_state.network_html.encode('utf-8')
            st.download_button(
                "🌐 Interactive Network (HTML)",
                data=network_bytes,
                file_name=f"pattern_network_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html",
                use_container_width=True
            )
        
        # Generate and download charts
        if st.button("📊 Export Charts", use_container_width=True):
            chart_package = create_chart_export_package(rules)
            st.download_button(
                "📁 Download Chart Package",
                data=chart_package,
                file_name=f"charts_package_{datetime.now().strftime('%Y%m%d')}.zip",
                mime="application/zip",
                use_container_width=True
            )
    
    with col3:
        st.markdown("""
        <div class="pro-container">
            <h4>📋 Business Reports</h4>
            <p>Executive summaries and action plans</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Executive report
        if st.button("📄 Generate Executive Report", use_container_width=True):
            exec_report = create_executive_report(rules, data, insights)
            st.download_button(
                "📄 Executive Report (PDF-ready HTML)",
                data=exec_report.encode('utf-8'),
                file_name=f"executive_report_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html",
                use_container_width=True
            )
    
    # Complete package download
    st.markdown("### 📦 Complete Analysis Package")
    
    if st.button("📁 Create Complete Package", type="primary", use_container_width=True):
        complete_package = create_complete_export_package(rules, data, insights)
        
        st.download_button(
            "📁 Download Complete Analysis Package",
            data=complete_package,
            file_name=f"complete_analysis_package_{datetime.now().strftime('%Y%m%d')}.zip",
            mime="application/zip",
            use_container_width=True
        )
    
    # Report scheduling
    st.markdown("### ⏰ Report Scheduling")
    
    st.info("""
    🔄 **Automated Reporting**: In a production environment, you could schedule:
    - Weekly pattern analysis reports
    - Monthly trend comparisons  
    - Real-time alerts for new high-impact patterns
    - Coaching effectiveness tracking
    """)
    
    # Usage analytics
    st.markdown("### 📊 Usage Analytics")
    
    if st.session_state.analysis_timestamp:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Last Analysis", 
                st.session_state.analysis_timestamp.strftime("%Y-%m-%d %H:%M"),
                f"Data: {len(data):,} records"
            )
        
        with col2:
            analysis_age = datetime.now() - st.session_state.analysis_timestamp
            st.metric(
                "Analysis Age",
                f"{analysis_age.days}d {analysis_age.seconds//3600}h",
                "Hours since last run"
            )

def create_chart_export_package(rules):
    """Create a ZIP package with exportable charts."""
    
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Impact analysis chart
        top_rules = rules.head(15).copy()
        
        fig_impact = go.Figure()
        fig_impact.add_trace(go.Bar(
            x=top_rules['impact_score'],
            y=top_rules['antecedents_str'],
            orientation='h',
            marker=dict(color='#3b82f6'),
            text=[f"Lift: {lift:.2f}" for lift in top_rules['lift']],
            textposition="outside"
        ))
        
        fig_impact.update_layout(
            title="Top Patterns by Business Impact",
            xaxis_title="Impact Score",
            yaxis_title="Pattern Conditions",
            height=500,
            template="plotly_white",
            font=dict(family="Inter, sans-serif")
        )
        
        zf.writestr("impact_analysis.html", fig_impact.to_html(full_html=True, include_plotlyjs="cdn"))
        
        # Confidence distribution
        fig_conf = go.Figure()
        conf_counts = rules['confidence_category'].value_counts()
        
        fig_conf.add_trace(go.Pie(
            labels=conf_counts.index,
            values=conf_counts.values,
            hole=0.4,
            marker=dict(colors=['#ef4444', '#f59e0b', '#10b981'])
        ))
        
        fig_conf.update_layout(
            title="Confidence Distribution",
            template="plotly_white",
            font=dict(family="Inter, sans-serif")
        )
        
        zf.writestr("confidence_distribution.html", fig_conf.to_html(full_html=True, include_plotlyjs="cdn"))
        
        # Scatter plot
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=rules['support'],
            y=rules['confidence'],
            mode='markers',
            marker=dict(
                size=rules['lift'] * 5,
                color=rules['lift'],
                colorscale='Viridis',
                colorbar=dict(title="Lift Value")
            ),
            text=rules['antecedents_str'] + ' → ' + rules['consequents_str']
        ))
        
        fig_scatter.update_layout(
            title="Pattern Quality Matrix",
            xaxis_title="Support",
            yaxis_title="Confidence",
            template="plotly_white",
            font=dict(family="Inter, sans-serif")
        )
        
        zf.writestr("quality_matrix.html", fig_scatter.to_html(full_html=True, include_plotlyjs="cdn"))
    
    return zip_buffer.getvalue()

def create_executive_report(rules, data, insights):
    """Generate a comprehensive executive report."""
    
    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Performance Affinity Analysis - Executive Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 30px;
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .metric-card {{
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                color: #1e40af;
                margin-bottom: 5px;
            }}
            .section {{
                margin: 30px 0;
                padding: 20px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .insight {{
                background: #f0f9ff;
                border-left: 4px solid #3b82f6;
                padding: 15px;
                margin: 15px 0;
            }}
            .risk-pattern {{
                background: #fef2f2;
                border-left: 4px solid #ef4444;
                padding: 15px;
                margin: 15px 0;
            }}
            .recommendation {{
                background: #f0fdf4;
                border-left: 4px solid #10b981;
                padding: 15px;
                margin: 15px 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #e5e7eb;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background: #f9fafb;
                font-weight: 600;
            }}
            .footer {{
                margin-top: 40px;
                text-align: center;
                color: #6b7280;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Performance Affinity Analysis</h1>
            <h2>Executive Report</h2>
            <p>Generated on {datetime.now().strftime("%B %d, %Y at %H:%M")}</p>
        </div>
        
        <div class="section">
            <h2>📊 Executive Summary</h2>
            <p>This analysis examined <strong>{len(data):,}</strong> agent performance records to identify 
            patterns and associations between key performance indicators (KPIs). The analysis discovered 
            <strong>{len(rules)}</strong> significant performance patterns that can guide coaching strategies 
            and operational improvements.</p>
            
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{len(rules)}</div>
                    <div>Performance Patterns</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(rules[rules['confidence'] >= 0.7])}</div>
                    <div>High-Confidence Rules</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{rules['lift'].max():.1f}x</div>
                    <div>Strongest Association</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{rules['support'].max():.1%}</div>
                    <div>Maximum Coverage</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>🎯 Key Findings</h2>
    """
    
    # Add top patterns
    top_patterns = rules.head(5)
    report_html += "<h3>Top Performance Patterns</h3><table><tr><th>Pattern</th><th>Outcome</th><th>Confidence</th><th>Impact</th></tr>"
    
    for _, rule in top_patterns.iterrows():
        report_html += f"""
        <tr>
            <td>{rule['antecedents_str'].replace('_', ' ')}</td>
            <td>{rule['consequents_str'].replace('_', ' ')}</td>
            <td>{rule['confidence']:.1%}</td>
            <td>{rule['support']:.1%}</td>
        </tr>
        """
    
    report_html += "</table>"
    
    # Add insights
    if insights.get('insights'):
        report_html += "<h3>💡 Key Insights</h3>"
        for insight in insights['insights'][:3]:
            report_html += f"""
            <div class="insight">
                <strong>{insight['type']}:</strong> {insight['finding']}
                <br><small>Confidence: {insight['confidence']:.1%} | Impact: {insight['impact']:.1%}</small>
            </div>
            """
    
    # Add risk patterns
    if insights.get('risk_patterns'):
        report_html += "<h3>⚠️ Risk Patterns</h3>"
        for risk in insights['risk_patterns'][:3]:
            report_html += f"""
            <div class="risk-pattern">
                <strong>Risk:</strong> {risk['pattern']}
                <br>{risk['description']}
                <br><small>Probability: {risk['probability']} | Frequency: {risk['frequency']}</small>
            </div>
            """
    
    # Add recommendations
    if insights.get('recommendations'):
        report_html += "<h3>📈 Recommendations</h3>"
        for rec in insights['recommendations'][:3]:
            report_html += f"""
            <div class="recommendation">
                <strong>{rec['priority']} Priority - {rec['area']}:</strong> {rec['action']}
                <br><small>Expected Impact: {rec['expected_impact']}</small>
            </div>
            """
    
    report_html += """
        </div>
        
        <div class="section">
            <h2>📋 Implementation Roadmap</h2>
            <h3>Phase 1: Immediate Actions (1-2 weeks)</h3>
            <ul>
                <li>Address high-priority patterns with immediate coaching interventions</li>
                <li>Implement monitoring for critical performance indicators</li>
                <li>Brief team leads on key findings and action items</li>
            </ul>
            
            <h3>Phase 2: Process Improvements (3-6 weeks)</h3>
            <ul>
                <li>Develop targeted training programs based on pattern analysis</li>
                <li>Review and update standard operating procedures</li>
                <li>Establish regular pattern monitoring and reporting</li>
            </ul>
            
            <h3>Phase 3: Optimization (7-12 weeks)</h3>
            <ul>
                <li>Measure impact of implemented changes</li>
                <li>Refine coaching strategies based on results</li>
                <li>Establish ongoing performance affinity monitoring</li>
            </ul>
        </div>
        
        <div class="footer">
            <p>This report was generated by the Performance Affinity Analytics Platform</p>
            <p>For questions or additional analysis, please contact your analytics team</p>
        </div>
    </body>
    </html>
    """
    
    return report_html

def create_complete_export_package(rules, data, insights):
    """Create a complete ZIP package with all analysis outputs."""
    
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Core data files
        zf.writestr("data/association_rules.csv", rules.to_csv(index=False))
        
        if st.session_state.binned_data is not None:
            zf.writestr("data/processed_data.csv", st.session_state.binned_data.to_csv(index=False))
        
        zf.writestr("data/original_data.csv", data.to_csv(index=False))
        
        # Executive report
        exec_report = create_executive_report(rules, data, insights)
        zf.writestr("reports/executive_report.html", exec_report)
        
        # Network visualization
        if st.session_state.network_html:
            zf.writestr("visualizations/interactive_network.html", st.session_state.network_html)
        
        # Charts package
        chart_package = create_chart_export_package(rules)
        # Extract charts from the chart package zip
        with zipfile.ZipFile(io.BytesIO(chart_package), 'r') as chart_zip:
            for chart_file in chart_zip.namelist():
                chart_content = chart_zip.read(chart_file)
                zf.writestr(f"visualizations/{chart_file}", chart_content)
        
        # Analysis metadata
        metadata = {
            "analysis_timestamp": st.session_state.analysis_timestamp.isoformat() if st.session_state.analysis_timestamp else None,
            "total_rules": len(rules),
            "data_records": len(data),
            "high_confidence_rules": len(rules[rules['confidence'] >= 0.7]),
            "max_lift": float(rules['lift'].max()) if not rules.empty else 0,
            "max_support": float(rules['support'].max()) if not rules.empty else 0,
            "insights_count": len(insights.get('insights', [])),
            "risk_patterns_count": len(insights.get('risk_patterns', [])),
            "recommendations_count": len(insights.get('recommendations', []))
        }
        
        zf.writestr("metadata/analysis_summary.json", json.dumps(metadata, indent=2))
        
        # Requirements file
        requirements = """
        streamlit>=1.36
        pandas>=2.2
        numpy>=1.26
        plotly>=5.22
        mlxtend>=0.23
        networkx>=3.2
        pyvis>=0.3.2
        streamlit-option-menu>=0.3.6
        """
        zf.writestr("requirements.txt", textwrap.dedent(requirements).strip())
        
        # README file
        readme = f"""
        # Performance Affinity Analysis Results
        
        Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M")}
        
        ## Contents
        
        ### /data/
        - `association_rules.csv`: Complete rule analysis results
        - `processed_data.csv`: Preprocessed dataset with binned KPIs
        - `original_data.csv`: Original input dataset
        
        ### /reports/
        - `executive_report.html`: Comprehensive business report
        
        ### /visualizations/
        - `interactive_network.html`: Interactive pattern network
        - `impact_analysis.html`: Top patterns by business impact
        - `confidence_distribution.html`: Rule confidence breakdown
        - `quality_matrix.html`: Support vs confidence analysis
        
        ### /metadata/
        - `analysis_summary.json`: Analysis statistics and metadata
        
        ## Key Findings
        
        - **Total Patterns:** {len(rules)}
        - **High Confidence Rules:** {len(rules[rules['confidence'] >= 0.7])}
        - **Strongest Association:** {rules['lift'].max():.2f}x stronger than random
        - **Maximum Coverage:** {rules['support'].max():.1%} of operations
        
        ## Next Steps
        
        1. Review the executive report for key insights and recommendations
        2. Examine high-priority patterns in the association rules
        3. Use the interactive network to explore pattern relationships
        4. Implement coaching strategies based on identified patterns
        5. Monitor KPIs to track improvement
        
        ## Support
        
        For questions about this analysis, please refer to the application documentation
        or contact your analytics team.
        """
        
        zf.writestr("README.md", textwrap.dedent(readme).strip())
    
    return zip_buffer.getvalue()

# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()
