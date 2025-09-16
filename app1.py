# enhanced_app.py — Enhanced Performance Affinity Analysis Platform
# ================================================================
# Fixed version with complete implementation
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
#   pyarrow>=15.0  # For Parquet support

import io
import json
import zipfile
from datetime import datetime
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from pyvis.network import Network

# =============================================================================
# SAMPLE DATA GENERATION
# =============================================================================

def generate_professional_sample_data(n_agents: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Generate realistic sample data for testing."""
    rng = np.random.default_rng(seed)
    
    agents = [f"AG{1000+i:04d}" for i in range(n_agents)]
    
    call_types = rng.choice([
        "Billing_Inquiry", "Technical_Support", "Account_Management", 
        "Sales_Consultation", "Complaint_Resolution", "Service_Activation"
    ], size=n_agents, p=[0.20, 0.18, 0.16, 0.15, 0.16, 0.15])
    
    aht = rng.lognormal(mean=5.8, sigma=0.4, size=n_agents)
    aht = aht.clip(120, 2400)
    
    acw = rng.lognormal(mean=4.0, sigma=0.5, size=n_agents)
    acw = acw.clip(15, 900)
    
    hold_time = rng.exponential(scale=30, size=n_agents)
    hold_time = hold_time.clip(0, 600)
    
    csat = rng.beta(a=4, b=1.5, size=n_agents) * 4 + 1
    csat = csat.clip(1.0, 5.0)
    
    fcr = rng.binomial(1, 0.75, size=n_agents)
    
    adherence = rng.beta(a=15, b=2, size=n_agents).clip(0.5, 1.0)
    qa_score = rng.beta(a=12, b=2.5, size=n_agents).clip(0.4, 1.0)
    escalations = rng.poisson(lam=0.8, size=n_agents)
    repeat_contacts = rng.poisson(lam=1.2, size=n_agents)
    absenteeism = rng.exponential(scale=0.03, size=n_agents).clip(0.0, 0.4)
    
    df = pd.DataFrame({
        "agent_id": agents,
        "call_type": call_types,
        "aht": aht.round(0).astype(int),
        "acw": acw.round(0).astype(int),
        "hold_time": hold_time.round(0).astype(int),
        "csat": csat.round(2),
        "fcr": fcr.astype(int),
        "adherence": adherence.round(3),
        "qa_score": qa_score.round(3),
        "escalations": escalations,
        "repeat_contacts": repeat_contacts,
        "absenteeism": absenteeism.round(3)
    })
    
    return df

# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def create_professional_bins(df, strategy="business_rules"):
    """Create professional bins for continuous variables."""
    binned = df.copy()
    
    # Bin numeric columns
    if 'aht' in df.columns:
        binned['aht_category'] = pd.cut(
            df['aht'],
            bins=[0, 300, 600, 900, float('inf')],
            labels=["AHT_Efficient", "AHT_Standard", "AHT_Extended", "AHT_NeedsReview"]
        )
    
    if 'csat' in df.columns:
        binned['csat_category'] = pd.cut(
            df['csat'],
            bins=[0, 3.0, 3.5, 4.0, 5.0],
            labels=["CSAT_BelowExpectations", "CSAT_NeedsImprovement", "CSAT_MeetsExpectations", "CSAT_ExceedsExpectations"]
        )
    
    if 'fcr' in df.columns:
        binned['fcr_category'] = df['fcr'].map({
            1: "FirstCallResolved",
            0: "RepeatRequired"
        })
    
    if 'adherence' in df.columns:
        binned['adherence_category'] = pd.cut(
            df['adherence'],
            bins=[0, 0.7, 0.85, 0.95, 1.0],
            labels=["Adherence_Concerning", "Adherence_BelowTarget", "Adherence_OnTarget", "Adherence_Excellent"]
        )
    
    if 'qa_score' in df.columns:
        binned['qa_category'] = pd.cut(
            df['qa_score'],
            bins=[0, 0.6, 0.75, 0.85, 1.0],
            labels=["QA_NeedsImprovement", "QA_BelowStandard", "QA_MeetsStandard", "QA_Exceptional"]
        )
    
    if 'escalations' in df.columns:
        binned['escalation_category'] = pd.cut(
            df['escalations'],
            bins=[-0.1, 0, 1, 3, float('inf')],
            labels=["Escalations_None", "Escalations_Minimal", "Escalations_Moderate", "Escalations_High"]
        )
    
    return binned

def create_market_basket(df_binned):
    """Create market basket format for association rules mining."""
    # Select categorical columns
    cat_cols = df_binned.select_dtypes(include=['object', 'category']).columns
    cat_cols = [c for c in cat_cols if c != 'agent_id']
    
    # Create binary encoding
    basket_parts = []
    for col in cat_cols:
        dummies = pd.get_dummies(df_binned[col], dtype=int)
        basket_parts.append(dummies)
    
    basket_df = pd.concat(basket_parts, axis=1)
    return basket_df

def mine_association_rules(basket_df, algorithm="apriori", min_support=0.05, 
                          metric="lift", min_threshold=1.1):
    """Mine association rules from basket data."""
    try:
        # Generate frequent itemsets
        if algorithm == "fpgrowth":
            frequent_itemsets = fpgrowth(basket_df, min_support=min_support, use_colnames=True)
        else:
            frequent_itemsets = apriori(basket_df, min_support=min_support, use_colnames=True)
        
        if frequent_itemsets.empty:
            return pd.DataFrame()
        
        # Generate association rules
        rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
        
        if rules.empty:
            return pd.DataFrame()
        
        # Sort by lift
        rules = rules.sort_values("lift", ascending=False)
        
        # Convert frozensets to strings
        rules["antecedents_str"] = rules["antecedents"].apply(lambda x: " + ".join(sorted(list(x))))
        rules["consequents_str"] = rules["consequents"].apply(lambda x: " + ".join(sorted(list(x))))
        
        # Add business metrics
        rules["impact_score"] = rules["support"] * rules["confidence"] * rules["lift"]
        rules["business_value"] = rules["confidence"] * rules["lift"] * np.log1p(rules["support"] * 100)
        
        # Select columns
        output_cols = ["antecedents_str", "consequents_str", "support", "confidence", 
                      "lift", "leverage", "conviction", "impact_score", "business_value"]
        rules = rules[output_cols].copy()
        
        # Round numerical columns
        numeric_cols = ["support", "confidence", "lift", "leverage", "conviction", 
                       "impact_score", "business_value"]
        rules[numeric_cols] = rules[numeric_cols].round(4)
        
        return rules.reset_index(drop=True)
        
    except Exception as e:
        st.error(f"Rule mining error: {str(e)}")
        return pd.DataFrame()

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_interactive_network(rules, max_nodes=30):
    """Create an interactive network visualization."""
    if rules.empty:
        return "<div>No rules available for visualization</div>"
    
    display_rules = rules.head(min(max_nodes, len(rules)))
    
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#1f2937")
    
    # Collect unique items
    all_items = set()
    for _, rule in display_rules.iterrows():
        antecedents = rule['antecedents_str'].split(' + ')
        consequents = rule['consequents_str'].split(' + ')
        all_items.update(antecedents)
        all_items.update(consequents)
    
    # Add nodes
    for item in all_items:
        clean_label = item.replace('_', ' ')
        if any(neg in item.lower() for neg in ['below', 'needs', 'concerning']):
            color = "#ef4444"  # Red
        elif any(pos in item.lower() for pos in ['excellent', 'exceeds', 'exceptional']):
            color = "#10b981"  # Green
        else:
            color = "#3b82f6"  # Blue
        
        net.add_node(item, label=clean_label, color=color, size=20)
    
    # Add edges
    for _, rule in display_rules.iterrows():
        antecedents = rule['antecedents_str'].split(' + ')
        consequents = rule['consequents_str'].split(' + ')
        
        for ant in antecedents:
            for con in consequents:
                if ant != con:
                    width = max(1, min(rule['confidence'] * 8, 6))
                    net.add_edge(ant, con, width=width, 
                               title=f"Conf: {rule['confidence']:.1%}, Lift: {rule['lift']:.2f}")
    
    return net.generate_html()

def create_risk_distribution_chart(rules):
    """Create risk distribution chart."""
    fig = go.Figure()
    
    # Simple risk categorization
    high_risk = len(rules[rules['confidence'] > 0.8])
    medium_risk = len(rules[(rules['confidence'] > 0.6) & (rules['confidence'] <= 0.8)])
    low_risk = len(rules[rules['confidence'] <= 0.6])
    
    fig.add_trace(go.Bar(
        x=['High Risk', 'Medium Risk', 'Low Risk'],
        y=[high_risk, medium_risk, low_risk],
        marker=dict(color=['#dc2626', '#f59e0b', '#3b82f6']),
        text=[high_risk, medium_risk, low_risk],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Risk Category Distribution",
        xaxis_title="Risk Category",
        yaxis_title="Number of Patterns",
        template="plotly_white",
        height=400
    )
    
    return fig

def create_business_value_matrix_chart(rules):
    """Create business value matrix chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=rules['impact_score'],
        y=rules['business_value'],
        mode='markers',
        marker=dict(
            size=rules['lift'] * 6,
            color=rules['confidence'],
            colorscale='Viridis',
            colorbar=dict(title="Confidence"),
            sizemode='diameter',
            sizeref=2.*max(rules['lift'])/40**2,
            sizemin=4
        ),
        text=rules['consequents_str'],
        hovertemplate="<b>%{text}</b><br>Impact: %{x:.4f}<br>Value: %{y:.4f}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Business Value vs Impact Matrix",
        xaxis_title="Impact Score",
        yaxis_title="Business Value",
        template="plotly_white",
        height=400
    )
    
    return fig

# =============================================================================
# STREAMLIT UI FUNCTIONS
# =============================================================================

def show_data_overview():
    """Show data overview section."""
    st.subheader("📊 Data Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload data file",
            type=["csv", "parquet", "xlsx"],
            help="Upload CSV, Parquet, or Excel file"
        )
        
        if st.button("🎲 Generate Sample Data", type="primary"):
            st.session_state.data = generate_professional_sample_data()
            st.success("✅ Sample data generated!")
            st.rerun()
    
    with col2:
        st.info("Required columns:\n- agent_id\n- call_type\n- aht\n- csat\n- fcr")
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.data = df
            st.success(f"✅ Data loaded: {len(df)} records")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    if 'data' in st.session_state:
        df = st.session_state.data
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Unique Agents", f"{df['agent_id'].nunique():,}")
        with col3:
            st.metric("Call Types", df['call_type'].nunique())
        with col4:
            completeness = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            st.metric("Data Quality", f"{completeness:.1f}%")
        
        with st.expander("Data Preview"):
            st.dataframe(df.head(20), use_container_width=True)

def show_analysis_setup():
    """Show analysis setup section."""
    if 'data' not in st.session_state:
        st.warning("Please load data first")
        return
    
    st.subheader("⚙️ Analysis Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        algorithm = st.selectbox("Algorithm", ["fpgrowth", "apriori"], index=0)
        min_support = st.slider("Min Support", 0.01, 0.3, 0.05, 0.01)
        min_confidence = st.slider("Min Confidence", 0.1, 0.95, 0.6, 0.05)
    
    with col2:
        st.info(f"Algorithm: {algorithm}\nSupport: {min_support:.1%}\nConfidence: {min_confidence:.1%}")
    
    if st.button("🔍 Run Analysis", type="primary"):
        with st.spinner("Running analysis..."):
            try:
                # Process data
                binned_data = create_professional_bins(st.session_state.data)
                basket_data = create_market_basket(binned_data)
                
                # Mine rules
                rules = mine_association_rules(
                    basket_data, algorithm, min_support, "lift", min_confidence
                )
                
                st.session_state.rules = rules
                st.session_state.binned_data = binned_data
                
                if rules.empty:
                    st.warning("No patterns found. Try adjusting parameters.")
                else:
                    st.success(f"✅ Found {len(rules)} patterns!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

def show_results():
    """Show analysis results."""
    if 'rules' not in st.session_state or st.session_state.rules.empty:
        st.warning("No results available. Run analysis first.")
        return
    
    rules = st.session_state.rules
    
    st.subheader("🔍 Analysis Results")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patterns", len(rules))
    with col2:
        high_conf = len(rules[rules['confidence'] >= 0.8])
        st.metric("High Confidence", high_conf)
    with col3:
        st.metric("Max Lift", f"{rules['lift'].max():.2f}x")
    with col4:
        st.metric("Avg Impact", f"{rules['impact_score'].mean():.4f}")
    
    # Visualizations
    tab1, tab2, tab3 = st.tabs(["Network", "Risk Analysis", "Business Value"])
    
    with tab1:
        network_html = create_interactive_network(rules)
        st.components.v1.html(network_html, height=650)
    
    with tab2:
        fig = create_risk_distribution_chart(rules)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = create_business_value_matrix_chart(rules)
        st.plotly_chart(fig, use_container_width=True)
    
    # Rules table
    st.subheader("Pattern Details")
    st.dataframe(
        rules,
        use_container_width=True,
        column_config={
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.1%"),
            "support": st.column_config.ProgressColumn("Support", min_value=0, max_value=1, format="%.1%"),
            "lift": st.column_config.NumberColumn("Lift", format="%.2fx")
        }
    )

def show_export():
    """Show export section."""
    if 'rules' not in st.session_state:
        st.warning("No results to export")
        return
    
    st.subheader("📥 Export Results")
    
    rules = st.session_state.rules
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = rules.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📋 Download CSV",
            data=csv,
            file_name=f"rules_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        buffer = io.BytesIO()
        rules.to_parquet(buffer, index=False)
        st.download_button(
            "📋 Download Parquet",
            data=buffer.getvalue(),
            file_name=f"rules_{datetime.now().strftime('%Y%m%d')}.parquet",
            mime="application/octet-stream"
        )
    
    with col3:
        # Create a simple report
        report = f"""
        # Analysis Report
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        
        ## Summary
        - Total Patterns: {len(rules)}
        - High Confidence (>80%): {len(rules[rules['confidence'] > 0.8])}
        - Max Lift: {rules['lift'].max():.2f}x
        - Average Impact: {rules['impact_score'].mean():.4f}
        
        ## Top Patterns
        {rules.head(10).to_string()}
        """
        
        st.download_button(
            "📄 Download Report",
            data=report.encode('utf-8'),
            file_name=f"report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application."""
    st.set_page_config(
        page_title="Performance Affinity Analytics",
        page_icon="📊",
        layout="wide"
    )
    
    # CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Performance Affinity Analytics Platform</h1>
        <p>KPI pattern discovery and analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    selected = option_menu(
        menu_title=None,
        options=["📊 Data", "⚙️ Setup", "🔍 Results", "📥 Export"],
        icons=["database", "gear", "search", "download"],
        default_index=0,
        orientation="horizontal"
    )
    
    # Route to pages
    if selected == "📊 Data":
        show_data_overview()
    elif selected == "⚙️ Setup":
        show_analysis_setup()
    elif selected == "🔍 Results":
        show_results()
    elif selected == "📥 Export":
        show_export()

if __name__ == "__main__":
    main()
