"""
Interactive Operations Risk Detection & Mitigation Dashboard for Tata Motors.
Built with Streamlit and Plotly.
Features:
- Top Operational KPIs & Health Scorecard
- Multi-dimensional Filtering (Plant, Supplier, Component, Product, Risk Level, Priority)
- 12 Dynamic Visual Analytics Modules
- Risk Priority Management Table
- Interactive What-If Scenario Stress Testing Simulation
- Action Lifecycle Management & Closed-Loop Feedback Tracker
- AI Explanation & Diagnostic Cards (6 Business Questions)
- Real-Time Autonomous Emergency Alert Center
"""

import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure Page
st.set_page_config(
    page_title="Tata Motors AI Operations Risk Control Tower",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Tata Motors Navy / Professional Steel Blue Theme)
st.markdown("""
<style>
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: #003366;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 14px;
        color: #475467;
        margin-bottom: 16px;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #EAECF0;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(16, 24, 40, 0.05);
    }
    .kpi-title {
        font-size: 12px;
        font-weight: 600;
        color: #667085;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: 700;
        color: #101828;
        margin-top: 4px;
    }
    .badge-critical {
        background-color: #FEE4E2;
        color: #B42318;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-high {
        background-color: #FEF0C7;
        color: #B54708;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-medium {
        background-color: #E0F2FE;
        color: #026AA2;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-low {
        background-color: #D1FADF;
        color: #027A48;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    data_path = "data/processed/ai_risk_mitigated_records.csv"
    if not os.path.exists(data_path):
        st.warning("Processed records not found. Please run 'python run_pipeline.py' first.")
        return None, None, None, None
        
    df = pd.read_csv(data_path)
    
    # Load actions registry
    act_path = "data/processed/action_tracking_registry.csv"
    df_actions = pd.read_csv(act_path) if os.path.exists(act_path) else pd.DataFrame()
    
    # Load alerts
    alert_path = "data/processed/operational_alerts.json"
    alerts = []
    if os.path.exists(alert_path):
        with open(alert_path, "r") as f:
            alerts = json.load(f)
            
    # Load scenario results
    sc_path = "data/processed/scenario_simulation_results.json"
    sc_results = []
    if os.path.exists(sc_path):
        with open(sc_path, "r") as f:
            sc_results = json.load(f)
            
    return df, df_actions, alerts, sc_results


df_main, df_actions, alerts_data, sc_data = load_data()

if df_main is None:
    st.stop()

# ==========================================
# SIDEBAR FILTERS
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/8/8e/Tata_logo.svg", width=90)
st.sidebar.markdown("### Operations Control Filters")

selected_plants = st.sidebar.multiselect(
    "Manufacturing Plant",
    options=sorted(df_main["Plant_ID"].unique()),
    default=sorted(df_main["Plant_ID"].unique())
)

selected_risk_levels = st.sidebar.multiselect(
    "Risk Classification",
    options=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
    default=["CRITICAL", "HIGH", "MEDIUM", "LOW"]
)

selected_priorities = st.sidebar.multiselect(
    "Customer Priority",
    options=["Critical", "High", "Normal"],
    default=["Critical", "High", "Normal"]
)

selected_products = st.sidebar.multiselect(
    "Vehicle Product Line",
    options=sorted(df_main["Product"].unique()),
    default=sorted(df_main["Product"].unique())
)

selected_components = st.sidebar.multiselect(
    "Component System",
    options=sorted(df_main["Component_Name"].unique()),
    default=sorted(df_main["Component_Name"].unique())
)

# Apply Filters
filtered_df = df_main[
    (df_main["Plant_ID"].isin(selected_plants)) &
    (df_main["Risk_Level"].isin(selected_risk_levels)) &
    (df_main["Customer_Priority"].isin(selected_priorities)) &
    (df_main["Product"].isin(selected_products)) &
    (df_main["Component_Name"].isin(selected_components))
].copy()

# ==========================================
# MAIN HEADER
# ==========================================
st.markdown("<div class='main-header'>Tata Motors — Autonomous AI Operations Risk Control Tower</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Predictive multi-echelon disruption detection, root cause diagnosis, and automated mitigation recommendation system</div>", unsafe_allow_html=True)

# ==========================================
# TOP KPIS
# ==========================================
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6, kpi7, kpi8 = st.columns(8)

total_sup = df_main["Supplier_ID"].nunique()
at_risk_sup = filtered_df[filtered_df["Supplier_Delay_Days"] > 2]["Supplier_ID"].nunique()
crit_inv = filtered_df[filtered_df["Days_of_Inventory"] < 2.0]["Component_Name"].nunique()
at_risk_prod = filtered_df[filtered_df["Expected_Prod_Delay_Days"] > 0]["Production_ID"].nunique()
delayed_shp = filtered_df[filtered_df["Transit_Delay_Days"] > 1]["Shipment_ID"].nunique()
at_risk_cust = filtered_df[filtered_df["Expected_Delivery_Delay_Days"] > 0]["Customer_Order_ID"].nunique()
crit_high_risks = (filtered_df["Risk_Level"].isin(["CRITICAL", "HIGH"])).sum()
avg_prod_delay = round(filtered_df["Expected_Prod_Delay_Days"].mean(), 1)

kpi1.metric("Total Suppliers", f"{total_sup}", "Active Tier-1")
kpi2.metric("At-Risk Suppliers", f"{at_risk_sup}", f"Delay > 2d", delta_color="inverse")
kpi3.metric("Critical Inventory", f"{crit_inv}", "< 2 Days Stock", delta_color="inverse")
kpi4.metric("At-Risk Prod Orders", f"{at_risk_prod}", f"{round(at_risk_prod/max(1, len(filtered_df))*100, 1)}%", delta_color="inverse")
kpi5.metric("Delayed Shipments", f"{delayed_shp}", "In-Transit", delta_color="inverse")
kpi6.metric("At-Risk Cust Orders", f"{at_risk_cust}", "SLA Exposure", delta_color="inverse")
kpi7.metric("High/Crit Risks", f"{crit_high_risks}", f"{len(filtered_df)} Monitored", delta_color="inverse")
kpi8.metric("Avg Prod Delay", f"{avg_prod_delay} d", "Predicted", delta_color="inverse")

st.markdown("---")

# ==========================================
# DASHBOARD TABS
# ==========================================
tab_visuals, tab_priority_table, tab_what_if, tab_actions, tab_ai_explain, tab_alerts = st.tabs([
    "📊 Risk Analytics & Visuals",
    "📋 Risk Priority Table",
    "🔮 What-If Scenario Analysis",
    "🔄 Action Management & Tracking",
    "🧠 AI Diagnostic Explanations",
    "🚨 Emergency Alerts Center"
])

# ----------------------------------------------------
# TAB 1: VISUAL ANALYTICS (12 VISUALS)
# ----------------------------------------------------
with tab_visuals:
    st.subheader("Multi-Echelon Operational Risk Telemetry")
    
    # Row 1: Overall Summary & Plant Distribution
    r1_col1, r1_col2 = st.columns([1, 1])
    with r1_col1:
        st.markdown("**1. Overall Risk Severity Distribution**")
        risk_counts = filtered_df["Risk_Level"].value_counts().reset_index()
        risk_counts.columns = ["Risk_Level", "Count"]
        color_map = {"CRITICAL": "#D92D20", "HIGH": "#F79009", "MEDIUM": "#026AA2", "LOW": "#12B76A"}
        fig_pie = px.pie(
            risk_counts, 
            names="Risk_Level", 
            values="Count", 
            color="Risk_Level",
            color_discrete_map=color_map,
            hole=0.45
        )
        fig_pie.update_layout(height=320, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with r1_col2:
        st.markdown("**2. Operational Risk Score by Plant Facility**")
        fig_plant = px.box(
            filtered_df,
            x="Plant_ID",
            y="Risk_Score",
            color="Plant_ID",
            points="outliers",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_plant.update_layout(height=320, margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
        st.plotly_chart(fig_plant, use_container_width=True)

    # Row 2: Supplier Risk & Inventory Risk
    r2_col1, r2_col2 = st.columns([1, 1])
    with r2_col1:
        st.markdown("**3. Supplier Delivery Deviation vs Reliability Score**")
        fig_sup = px.scatter(
            filtered_df,
            x="Supplier_Reliability_Score",
            y="Supplier_Delay_Days",
            color="Risk_Level",
            size="Order_Quantity",
            hover_data=["Supplier_Name", "Component_Name", "Product"],
            color_discrete_map=color_map
        )
        fig_sup.update_layout(height=340, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_sup, use_container_width=True)
        
    with r2_col2:
        st.markdown("**4. Inventory Health: Days of Inventory vs Stockout Probability**")
        fig_inv = px.scatter(
            filtered_df,
            x="Days_of_Inventory",
            y="Stockout_Probability",
            color="Risk_Level",
            size="Safety_Stock_Gap",
            hover_data=["Component_Name", "Plant_ID", "Current_Stock"],
            color_discrete_map=color_map
        )
        fig_inv.add_vline(x=2.0, line_dash="dash", line_color="red", annotation_text="Stockout Danger Line (2.0d)")
        fig_inv.update_layout(height=340, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_inv, use_container_width=True)

    # Row 3: Production Risk & Transportation Risk
    r3_col1, r3_col2 = st.columns([1, 1])
    with r3_col1:
        st.markdown("**5. Production Strain: Capacity Utilization vs Machine Downtime Hours**")
        fig_prod = px.scatter(
            filtered_df,
            x="Capacity_Utilization",
            y="Downtime_Hours",
            color="Risk_Level",
            size="Backlog",
            hover_data=["Production_Line", "Product", "Component_Availability"],
            color_discrete_map=color_map
        )
        fig_prod.update_layout(height=340, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_prod, use_container_width=True)
        
    with r3_col2:
        st.markdown("**6. Transportation Logistics: Transit Delay Days by Carrier & Route**")
        carrier_delay = filtered_df.groupby("Carrier")["Transit_Delay_Days"].mean().reset_index()
        fig_trans = px.bar(
            carrier_delay,
            x="Carrier",
            y="Transit_Delay_Days",
            color="Transit_Delay_Days",
            color_continuous_scale="Reds"
        )
        fig_trans.update_layout(height=340, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_trans, use_container_width=True)

    # Row 4: Customer Delivery Risk & Top 10 Critical Risks
    r4_col1, r4_col2 = st.columns([1, 1])
    with r4_col1:
        st.markdown("**7. Customer Delivery SLA Buffer vs Order Volume**")
        fig_cust = px.scatter(
            filtered_df,
            x="Deadline_Slack_Days",
            y="Order_Quantity",
            color="Customer_Priority",
            size="Risk_Score",
            hover_data=["Customer_Name", "Product", "Expected_Delivery_Delay_Days"],
            color_discrete_map={"Critical": "#D92D20", "High": "#F79009", "Normal": "#026AA2"}
        )
        fig_cust.add_vline(x=0, line_dash="solid", line_color="black", annotation_text="Delivery Breach SLA")
        fig_cust.update_layout(height=340, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_cust, use_container_width=True)
        
    with r4_col2:
        st.markdown("**8. Top 10 Most Critical Operations by Financial Impact**")
        top10 = filtered_df.sort_values(by="Estimated_Business_Impact_Lakhs", ascending=False).head(10)
        fig_top10 = px.bar(
            top10,
            x="Estimated_Business_Impact_Lakhs",
            y="Master_Record_ID",
            orientation="h",
            color="Risk_Score",
            hover_data=["Product", "Component_Name", "Plant_ID", "Supplier_Name"],
            color_continuous_scale="YlOrRd"
        )
        fig_top10.update_layout(height=340, margin=dict(t=20, b=20, l=20, r=20), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_top10, use_container_width=True)

    # Row 5: Risk by Component & Risk Trend
    r5_col1, r5_col2 = st.columns([1, 1])
    with r5_col1:
        st.markdown("**9. Component System Risk Exposure**")
        comp_risk = filtered_df.groupby("Component_Name")["Risk_Score"].agg(["mean", "count"]).reset_index()
        fig_comp = px.bar(
            comp_risk,
            x="Component_Name",
            y="mean",
            color="mean",
            color_continuous_scale="Purples"
        )
        fig_comp.update_layout(height=340, margin=dict(t=20, b=20, l=20, r=20), xaxis_tickangle=-45)
        st.plotly_chart(fig_comp, use_container_width=True)

    with r5_col2:
        st.markdown("**10. Recommended Mitigation Actions by Priority Tier**")
        action_prio = filtered_df["Action_Priority"].value_counts().reset_index()
        action_prio.columns = ["Priority", "Count"]
        prio_names = {1: "Priority 1: Immediate (<4h)", 2: "Priority 2: High (<24h)", 3: "Priority 3: Medium (<48h)", 4: "Priority 4: Routine"}
        action_prio["Priority_Label"] = action_prio["Priority"].map(prio_names)
        fig_act = px.bar(
            action_prio,
            x="Priority_Label",
            y="Count",
            color="Priority_Label",
            color_discrete_sequence=["#D92D20", "#F79009", "#026AA2", "#12B76A"]
        )
        fig_act.update_layout(height=340, margin=dict(t=20, b=20, l=20, r=20), showlegend=False)
        st.plotly_chart(fig_act, use_container_width=True)

# ----------------------------------------------------
# TAB 2: RISK PRIORITY TABLE (Section 13)
# ----------------------------------------------------
with tab_priority_table:
    st.subheader("Operational Risk Priority Master Table")
    st.markdown("Dynamic risk registry showing root cause diagnoses, predicted delays, customer impacts, and mitigation actions.")
    
    # Table columns matching Section 13 exactly:
    # Risk ID, Risk Type, Supplier, Component, Plant, Root Cause, Probability, Impact, Risk Score, Risk Level, Predicted Delay, Customer Impact, Recommended Action, Priority, Status
    display_cols = [
        "Master_Record_ID", "Risk_Type", "Supplier_Name", "Component_Name", "Plant_ID",
        "Primary_Root_Cause", "Probability_Factor", "Impact_Factor", "Risk_Score",
        "Risk_Level", "Expected_Prod_Delay_Days", "Estimated_Business_Impact_Formatted",
        "Recommended_Action", "Action_Priority", "Production_Status"
    ]
    
    table_view = filtered_df[display_cols].copy()
    table_view.columns = [
        "Risk ID", "Risk Type", "Supplier", "Component", "Plant",
        "Root Cause", "Probability", "Impact", "Risk Score",
        "Risk Level", "Predicted Delay (Days)", "Customer Impact",
        "Recommended Action", "Priority", "Status"
    ]
    
    st.dataframe(
        table_view,
        use_container_width=True,
        height=520,
        column_config={
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Score",
                help="Normalized Risk Score (0-100)",
                format="%f",
                min_value=0,
                max_value=100
            ),
            "Probability": st.column_config.NumberColumn(format="%.2f"),
            "Impact": st.column_config.NumberColumn(format="%.2f")
        }
    )
    
    # Download button for CSV export
    csv_data = table_view.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Risk Priority Table (CSV)",
        data=csv_data,
        file_name="Tata_Motors_Risk_Priority_Table.csv",
        mime="text/csv"
    )

# ----------------------------------------------------
# TAB 3: WHAT-IF SCENARIO ANALYSIS (Section 15)
# ----------------------------------------------------
with tab_what_if:
    st.subheader("What-If Operational Scenario & Stress-Test Simulation")
    st.markdown("Simulate supply chain disruptions and recalculate Risk Scores, ML Delay Probabilities, and Financial Exposure in real-time.")
    
    # Standard Scenarios Quick Buttons
    st.markdown("#### Pre-Configured Operational Scenarios:")
    q_col1, q_col2, q_col3, q_col4, q_col5 = st.columns(5)
    
    scenario_choice = None
    if q_col1.button("Scenario 1\nSupplier +5 Days"):
        scenario_choice = 1
    if q_col2.button("Scenario 2\nInventory -30%"):
        scenario_choice = 2
    if q_col3.button("Scenario 3\nCapacity -20%"):
        scenario_choice = 3
    if q_col4.button("Scenario 4\nTransit +3 Days"):
        scenario_choice = 4
    if q_col5.button("Scenario 5\nDemand +15%"):
        scenario_choice = 5
        
    st.markdown("#### Custom Parameter Simulation Sliders:")
    sim_c1, sim_c2, sim_c3, sim_c4, sim_c5 = st.columns(5)
    
    sup_delta = sim_c1.slider("Supplier Delay Shock (Days)", min_value=0, max_value=12, value=5 if scenario_choice==1 else 0)
    inv_change = sim_c2.slider("Inventory Reduction (%)", min_value=-60, max_value=20, value=-30 if scenario_choice==2 else 0)
    cap_change = sim_c3.slider("Capacity Reduction (%)", min_value=-50, max_value=20, value=-20 if scenario_choice==3 else 0)
    trans_delta = sim_c4.slider("Transit Time Shock (Days)", min_value=0, max_value=10, value=3 if scenario_choice==4 else 0)
    dem_change = sim_c5.slider("Demand Spike (%)", min_value=-20, max_value=50, value=15 if scenario_choice==5 else 0)
    
    # Execute simulation dynamically
    from src.scenario_simulator import ScenarioSimulator
    sim_engine = ScenarioSimulator()
    sim_results, sim_df = sim_engine.simulate(
        scenario_name="Interactive User Simulation",
        sup_delay_delta=float(sup_delta),
        inv_pct_change=float(inv_change / 100.0),
        cap_pct_change=float(cap_change / 100.0),
        transit_delta=float(trans_delta),
        demand_pct_change=float(dem_change / 100.0)
    )
    
    comp = sim_results["comparison"]
    st.markdown("---")
    st.markdown("### Simulated Stress-Test Impact Summary")
    
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    m_col1.metric("Average Risk Score", f"{comp['avg_risk_score']['simulated']}", f"{comp['avg_risk_score']['delta']:+0.1f} pts", delta_color="inverse")
    m_col2.metric("Critical Risks Count", f"{comp['critical_risks_count']['simulated']}", f"{comp['critical_risks_count']['delta']:+d}", delta_color="inverse")
    m_col3.metric("Expected Prod Delay", f"{comp['expected_prod_delay_days']['simulated']} days", f"{comp['expected_prod_delay_days']['delta']:+0.2f} d", delta_color="inverse")
    m_col4.metric("Expected Delivery Delay", f"{comp['expected_delivery_delay_days']['simulated']} days", f"{comp['expected_delivery_delay_days']['delta']:+0.2f} d", delta_color="inverse")
    m_col5.metric("Total Financial Exposure", f"Rs {comp['total_financial_exposure_lakhs']['simulated']/100.0:.2f} Cr", f"+Rs {comp['total_financial_exposure_lakhs']['delta']/100.0:.2f} Cr", delta_color="inverse")

    st.markdown("#### Tailored Scenario Mitigations Recommended by AI:")
    for m in sim_results["recommended_mitigation"]:
        st.info(f"👉 **{m}**")

# ----------------------------------------------------
# TAB 4: ACTION TRACKING (Section 14)
# ----------------------------------------------------
with tab_actions:
    st.subheader("Operational Mitigation Action Registry & Closed-Loop Tracker")
    st.markdown("Track execution lifecycles, team accountability, and measure pre-action vs post-action risk reductions.")
    
    if not df_actions.empty:
        # Action Status Summary Cards
        a_col1, a_col2, a_col3, a_col4 = st.columns(4)
        c_comp = (df_actions["Status"] == "Completed").sum()
        c_prog = (df_actions["Status"] == "In Progress").sum()
        c_pend = (df_actions["Status"] == "Pending").sum()
        c_fail = (df_actions["Status"] == "Failed").sum()
        
        a_col1.metric("Completed Actions", f"{c_comp}", "Resolved Risks")
        a_col2.metric("In Progress", f"{c_prog}", "Active Linehaul/PO")
        a_col3.metric("Pending Sign-Off", f"{c_pend}", "Awaiting Approval", delta_color="inverse")
        a_col4.metric("Failed / Escalated", f"{c_fail}", "Needs Escalation", delta_color="inverse")
        
        st.markdown("---")
        
        # Interactive Status Updater
        st.markdown("#### Update Action Status in Real-Time:")
        u_col1, u_col2, u_col3, u_col4 = st.columns([2, 2, 3, 1])
        selected_risk_id = u_col1.selectbox("Select Action (Risk ID)", df_actions["Risk_ID"].tolist()[:50])
        new_stat = u_col2.selectbox("New Lifecycle Status", ["Completed", "In Progress", "Pending", "Failed"])
        custom_notes = u_col3.text_input("Operational Outcome Notes", "Emergency stock delivered; line running on schedule.")
        
        if u_col4.button("Update"):
            from src.action_tracker import ActionTracker
            tracker = ActionTracker()
            updated_df = tracker.update_action_status(selected_risk_id, new_stat, custom_notes)
            st.success(f"Action {selected_risk_id} successfully updated to '{new_stat}'!")
            st.rerun()

        # Action Data Table
        st.dataframe(
            df_actions[[
                "Risk_ID", "Plant_ID", "Product", "Component_Name", "Recommended_Action",
                "Responsible_Team", "Priority", "Status", "Risk_Before_Action",
                "Risk_After_Action", "Risk_Reduction_Pct", "Outcome"
            ]],
            use_container_width=True,
            height=420
        )
    else:
        st.info("No action records loaded. Run pipeline to initialize action tracking registry.")

# ----------------------------------------------------
# TAB 5: AI EXPLANATION CARDS (Section 16)
# ----------------------------------------------------
with tab_ai_explain:
    st.subheader("Explainable AI Operations Diagnostic Cards")
    st.markdown("Auditable 6-question structured reasoning for every HIGH or CRITICAL risk.")
    
    crit_high_list = filtered_df[filtered_df["Risk_Level"].isin(["CRITICAL", "HIGH"])].copy()
    
    if not crit_high_list.empty:
        selected_op_id = st.selectbox(
            "Select Operation to Inspect:",
            crit_high_list["Master_Record_ID"].tolist(),
            format_func=lambda x: f"{x} - {crit_high_list[crit_high_list['Master_Record_ID']==x]['Product'].values[0]} ({crit_high_list[crit_high_list['Master_Record_ID']==x]['Risk_Level'].values[0]})"
        )
        
        op_row = crit_high_list[crit_high_list["Master_Record_ID"] == selected_op_id].iloc[0]
        
        # Display Diagnostic Summary Banner
        st.markdown(f"""
        <div style="background:#F8F9FA; border-left: 5px solid {'#D92D20' if op_row['Risk_Level']=='CRITICAL' else '#F79009'}; padding:16px; border-radius:6px; margin-bottom:16px;">
            <h4 style="margin:0 0 8px 0; color:#101828;">{op_row['Risk_Headline']}</h4>
            <b>Plant:</b> {op_row['Plant_ID']} &nbsp;|&nbsp; <b>Line:</b> {op_row['Production_Line']} &nbsp;|&nbsp; <b>Component:</b> {op_row['Component_Name']} &nbsp;|&nbsp; <b>Risk Score:</b> {op_row['Risk_Score']}/100 [{op_row['Risk_Level']}]<br/>
            <b>Customer:</b> {op_row['Customer_Name']} ({op_row['Customer_Priority']} Priority) &nbsp;|&nbsp; <b>Financial Exposure:</b> {op_row['Estimated_Business_Impact_Formatted']}
        </div>
        """, unsafe_allow_html=True)
        
        # The 6 Required Structured Business Questions
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            st.markdown("##### 1. WHY IS THIS RISK HIGH?")
            st.write(op_row["Why_Is_Risk_High"])
            
            st.markdown("##### 2. WHAT CAUSED IT?")
            st.write(op_row["What_Caused_It"])
            
            st.markdown("##### 3. WHAT WILL HAPPEN IF NO ACTION IS TAKEN?")
            st.error(op_row["What_Will_Happen_If_No_Action"])
            
        with col_q2:
            st.markdown("##### 4. WHAT SHOULD THE COMPANY DO?")
            st.info(op_row["What_Should_Company_Do"])
            
            st.markdown("##### 5. WHY IS THIS ACTION RECOMMENDED?")
            st.write(op_row["Why_Action_Recommended"])
            
            st.markdown("##### 6. EXPECTED IMPACT OF THE ACTION")
            st.success(op_row["Expected_Impact_Of_Action"])
            
        st.markdown("---")
        st.markdown("**Contributing Operational Telemetry Factors:**")
        for factor in str(op_row["Contributing_Factors"]).split(";"):
            st.markdown(f"- {factor.strip()}")
    else:
        st.info("No High or Critical risks match the current filter selection.")

# ----------------------------------------------------
# TAB 6: EMERGENCY ALERTS CENTER (Section 11)
# ----------------------------------------------------
with tab_alerts:
    st.subheader("Autonomous Operations Emergency Alerts")
    st.markdown("Real-time alert dispatch log for shopfloor supervisors, procurement managers, and plant general managers.")
    
    if alerts_data:
        st.caption(f"Displaying {len(alerts_data)} active operational alerts")
        for alt in alerts_data[:15]:
            is_crit = alt["severity"] == "CRITICAL"
            badge_color = "#D92D20" if is_crit else "#F79009"
            bg_color = "#FFF5F5" if is_crit else "#FFFDF5"
            
            with st.expander(f"[{alt['severity']}] {alt['alert_id']} — {alt['headline']} (Score: {alt['risk_score']}/100)"):
                st.markdown(f"""
                **Plant / Line:** `{alt['plant']}` / `{alt['production_line']}` &nbsp;|&nbsp; **Product:** `{alt['product']}` &nbsp;|&nbsp; **Component:** `{alt['component']}`<br/>
                **Supplier:** `{alt['supplier']}` &nbsp;|&nbsp; **Customer:** `{alt['customer']}`<br/>
                **Predicted Delays:** Prod: `{alt['predictions']['expected_production_delay']}` ({alt['predictions']['production_delay_probability']}) &nbsp;|&nbsp; Delivery: `{alt['predictions']['expected_customer_delay']}` ({alt['predictions']['customer_delay_probability']})<br/>
                **Financial Exposure:** `{alt['predictions']['estimated_financial_loss']}`
                """, unsafe_allow_html=True)
                
                st.markdown(f"**Root Cause Diagnosis:**\n> {alt['root_cause']}")
                
                st.markdown(f"**Mitigation Action Plan (Priority {alt['priority']} - Owner: {alt['responsible_team']}):**")
                for act in alt["recommended_actions"]:
                    st.markdown(f"- {act}")
    else:
        st.info("No active alerts generated. Run pipeline to initialize alert dispatch.")
