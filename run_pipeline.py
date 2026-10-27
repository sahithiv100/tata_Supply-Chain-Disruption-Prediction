"""
Master Autonomous Execution Pipeline for Tata Motors AI Operations Risk Detection & Mitigation System.
Orchestrates the entire end-to-end workflow:
1. Data Generation
2. Data Cleaning & Validation
3. Feature Engineering
4. AI Risk Detection & Early Warning
5. Root Cause Analysis
6. ML Delay Prediction & Business Impact
7. Mitigation Recommendation & Action Prioritization
8. Action Management Registry Initialization
9. Autonomous Alert Dispatch
10. What-If Scenario Stress Testing
11. Executive PDF Management Report Generation
"""

import os
import sys
import time
from datetime import datetime

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_generator import generate_all_datasets
from src.data_cleaner import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.ai_risk_engine import AIRiskEngine
from src.root_cause_analyzer import RootCauseAnalyzer
from src.delay_predictor import DelayPredictor
from src.mitigation_engine import MitigationEngine
from src.action_tracker import ActionTracker
from src.alert_system import AlertSystem
from src.scenario_simulator import ScenarioSimulator
from src.pdf_report_generator import PDFReportGenerator


def print_banner():
    banner = """
========================================================================================
     TATA MOTORS AI OPERATIONS RISK DETECTION & MITIGATION SYSTEM
                 Autonomous End-to-End Intelligence Pipeline
========================================================================================
    """
    print(banner)


def run_pipeline():
    start_total = time.time()
    print_banner()
    print(f"Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 88)
    
    # ----------------------------------------------------
    # STEP 1: SYNTHETIC DATA GENERATION
    # ----------------------------------------------------
    print("\n[STEP 1/11] Generating Realistic Synthetic Operational Datasets...")
    t0 = time.time()
    raw_files = generate_all_datasets(output_dir="data/raw")
    print(f"[COMPLETED] Data Generation in {time.time() - t0:.2f}s")
    
    # ----------------------------------------------------
    # STEP 2: DATA CLEANING & VALIDATION
    # ----------------------------------------------------
    print("\n[STEP 2/11] Running Data Cleaning, Validation & Anomaly Correction...")
    t0 = time.time()
    cleaner = DataCleaner(raw_dir="data/raw", cleaned_dir="data/cleaned")
    dq_report, cleaned_data = cleaner.run_cleaning_pipeline()
    print(f"[COMPLETED] Data Cleaning & Quality Reporting in {time.time() - t0:.2f}s")
    
    # ----------------------------------------------------
    # STEP 3: FEATURE ENGINEERING & MASTER DATASET
    # ----------------------------------------------------
    print("\n[STEP 3/11] Performing Feature Engineering Across 5 Operational Domains...")
    t0 = time.time()
    fe = FeatureEngineer(cleaned_dir="data/cleaned", processed_dir="data/processed")
    master_df = fe.run_feature_engineering_pipeline()
    print(f"[COMPLETED] Feature Engineering in {time.time() - t0:.2f}s")
    
    # ----------------------------------------------------
    # STEP 4: AI RISK DETECTION & EARLY WARNING
    # ----------------------------------------------------
    print("\n[STEP 4/11] Executing AI Risk Detection, Anomaly Detection & 0-100 Scoring...")
    t0 = time.time()
    risk_engine = AIRiskEngine(processed_dir="data/processed")
    scored_df = risk_engine.run_risk_detection_pipeline()
    print(f"[COMPLETED] Risk Detection & Scoring in {time.time() - t0:.2f}s")
    
    # ----------------------------------------------------
    # STEP 5: ROOT CAUSE ANALYSIS & 6-QUESTION EXPLANATION
    # ----------------------------------------------------
    print("\n[STEP 5/11] Performing Root Cause Analysis & Diagnostic Attribution...")
    t0 = time.time()
    rca = RootCauseAnalyzer()
    rca_df = rca.analyze_dataset(scored_df)
    rca_df.to_csv("data/processed/ai_risk_rca_records.csv", index=False)
    print(f"[COMPLETED] Root Cause Analysis in {time.time() - t0:.2f}s")
    
    # ----------------------------------------------------
    # STEP 6: ML DELAY PREDICTION & FINANCIAL IMPACT
    # ----------------------------------------------------
    print("\n[STEP 6/11] Training ML Models & Predicting Delays and Business Impact...")
    t0 = time.time()
    predictor = DelayPredictor(models_dir="data/processed/models")
    predictor.train_models(rca_df)
    delay_df = predictor.predict(rca_df)
    delay_df.to_csv("data/processed/ai_risk_delay_predicted_records.csv", index=False)
    print(f"[COMPLETED] ML Delay Prediction in {time.time() - t0:.2f}s")
    
    # ----------------------------------------------------
    # STEP 7: MITIGATION RECOMMENDATION & ACTION PRIORITIZATION
    # ----------------------------------------------------
    print("\n[STEP 7/11] Formulating Tailored Mitigations & Ranking Action Priorities...")
    t0 = time.time()
    mit_engine = MitigationEngine()
    mit_df = mit_engine.process_dataset(delay_df)
    mit_path = "data/processed/ai_risk_mitigated_records.csv"
    mit_df.to_csv(mit_path, index=False)
    print(f"[COMPLETED] Mitigation Recommendations in {time.time() - t0:.2f}s")
    
    # ----------------------------------------------------
    # STEP 8: ACTION MANAGEMENT REGISTRY
    # ----------------------------------------------------
    print("\n[STEP 8/11] Initializing Action Tracking & Closed-Loop Feedback Registry...")
    t0 = time.time()
    tracker = ActionTracker(data_dir="data/processed")
    tracker.initialize_action_registry(mit_df)
    print(f"[COMPLETED] Action Registry Initialization in {time.time() - t0:.2f}s")
    
    # ----------------------------------------------------
    # STEP 9: AUTONOMOUS ALERT DISPATCH
    # ----------------------------------------------------
    print("\n[STEP 9/11] Generating Autonomous Operations Alerts...")
    t0 = time.time()
    alert_sys = AlertSystem(output_dir="data/processed")
    alerts = alert_sys.dispatch_alerts(mit_df, max_alerts=35)
    print(f"[COMPLETED] Dispatched {len(alerts)} alerts in {time.time() - t0:.2f}s")
    
    # ----------------------------------------------------
    # STEP 10: WHAT-IF SCENARIO STRESS TESTING
    # ----------------------------------------------------
    print("\n[STEP 10/11] Running 5 Supply Chain What-If Stress Tests...")
    t0 = time.time()
    simulator = ScenarioSimulator(data_path=mit_path, models_dir="data/processed/models")
    scenarios = simulator.run_all_five_scenarios(output_dir="data/processed")
    print(f"[COMPLETED] Scenario Stress Testing in {time.time() - t0:.2f}s")
    
    # ----------------------------------------------------
    # STEP 11: PROFESSIONAL PDF REPORT GENERATION
    # ----------------------------------------------------
    print("\n[STEP 11/11] Compiling Professional Executive PDF Management Report...")
    t0 = time.time()
    pdf_gen = PDFReportGenerator(
        data_path=mit_path,
        report_path="reports/Tata_Motors_AI_Operations_Risk_Report.pdf"
    )
    pdf_path = pdf_gen.generate_report()
    print(f"[COMPLETED] PDF Report Generated in {time.time() - t0:.2f}s")
    
    # ----------------------------------------------------
    # FINAL SUMMARY REPORT
    # ----------------------------------------------------
    total_time = time.time() - start_total
    crit_count = int((mit_df["Risk_Level"] == "CRITICAL").sum())
    high_count = int((mit_df["Risk_Level"] == "HIGH").sum())
    fin_exposure_cr = round(mit_df["Estimated_Business_Impact_Lakhs"].sum() / 100.0, 2)
    
    print("\n" + "=" * 88)
    print("                    MASTER PIPELINE EXECUTION SUMMARY")
    print("=" * 88)
    print(f"Total Operations Monitored:       {len(mit_df)} records")
    print(f"Critical Risks Identified:        {crit_count} operations")
    print(f"High Risks Identified:            {high_count} operations")
    print(f"Total Potential Financial Loss:   Rs {fin_exposure_cr} Crores")
    print(f"Management Alerts Dispatched:     {len(alerts)} alerts")
    print(f"Stress-Test Scenarios Evaluated:  5 macroeconomic shocks")
    print(f"Executive Report Location:        {os.path.abspath(pdf_path)}")
    print(f"Total Execution Time:             {total_time:.2f} seconds")
    print("=" * 88)
    print("System is fully ready. To launch the interactive dashboard, run:")
    print("  streamlit run dashboard.py")
    print("=" * 88)


if __name__ == "__main__":
    run_pipeline()
