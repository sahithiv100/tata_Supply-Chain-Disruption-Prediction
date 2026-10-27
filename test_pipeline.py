"""
Unit and Integration Test Suite for Tata Motors AI Operations Risk Detection & Mitigation System.
Validates:
1. Synthetic Data Generation & Volume Bounds
2. Data Cleaning & Referential Integrity
3. Feature Engineering & Mathematical Correctness
4. AI Risk Scoring & Classification
5. Root Cause Analysis Coverage
6. ML Delay Prediction & Financial Loss Estimator
7. Mitigation Recommendation & Action Prioritization
8. What-If Scenario Stress Testing
9. PDF Report Generation
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_generator import generate_all_datasets
from src.data_cleaner import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.ai_risk_engine import AIRiskEngine
from src.root_cause_analyzer import RootCauseAnalyzer
from src.delay_predictor import DelayPredictor
from src.mitigation_engine import MitigationEngine
from src.scenario_simulator import ScenarioSimulator
from src.pdf_report_generator import PDFReportGenerator


@pytest.fixture(scope="session")
def setup_environment(tmp_path_factory):
    """Ensure data directories and base artifacts exist."""
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/cleaned", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports", exist_ok=True)


def test_data_generation():
    """Verify synthetic datasets generation satisfies record count requirements (>500 records)."""
    files = generate_all_datasets("data/raw")
    for name, df in files.items():
        assert len(df) >= 500, f"Dataset {name} has fewer than 500 records: {len(df)}"
        assert not df.empty, f"Dataset {name} is empty"


def test_data_cleaning():
    """Verify data cleaner removes duplicates, handles missing values, and builds quality report."""
    cleaner = DataCleaner("data/raw", "data/cleaned")
    report, cleaned_dfs = cleaner.run_cleaning_pipeline()
    
    assert "overall_summary" in report
    assert report["overall_summary"]["final_usable_records"] > 0
    assert report["overall_summary"]["data_health_index"] >= 90.0
    
    for name, df in cleaned_dfs.items():
        assert df.duplicated().sum() == 0, f"Dataset {name} still contains duplicates"
        assert not df.empty


def test_feature_engineering():
    """Verify operational indicators are mathematically sound."""
    fe = FeatureEngineer("data/cleaned", "data/processed")
    master_df = fe.run_feature_engineering_pipeline()
    
    assert len(master_df) >= 500
    assert "Days_of_Inventory" in master_df.columns
    assert "Stockout_Probability" in master_df.columns
    assert "Supplier_Risk_Factor" in master_df.columns
    assert "Deadline_Slack_Days" in master_df.columns
    
    # Mathematical bounds
    assert (master_df["Days_of_Inventory"] >= 0).all()
    assert (master_df["Stockout_Probability"] >= 0).all() and (master_df["Stockout_Probability"] <= 1.0).all()


def test_ai_risk_scoring():
    """Verify AI risk scoring bounds (0-100) and classification integrity."""
    engine = AIRiskEngine("data/processed")
    scored_df = engine.run_risk_detection_pipeline()
    
    assert "Risk_Score" in scored_df.columns
    assert "Risk_Level" in scored_df.columns
    assert (scored_df["Risk_Score"] >= 0).all() and (scored_df["Risk_Score"] <= 100).all()
    
    valid_levels = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
    assert set(scored_df["Risk_Level"].unique()).issubset(valid_levels)
    assert (scored_df["Risk_Level"] == "CRITICAL").sum() > 0


def test_root_cause_analysis():
    """Verify every High/Critical risk receives diagnostic attribution."""
    scored_df = pd.read_csv("data/processed/ai_risk_scored_records.csv")
    rca = RootCauseAnalyzer()
    rca_df = rca.analyze_dataset(scored_df)
    
    crit_high = rca_df[rca_df["Risk_Level"].isin(["CRITICAL", "HIGH"])]
    for _, row in crit_high.iterrows():
        assert len(str(row["Primary_Root_Cause"])) > 10
        assert len(str(row["Why_Is_Risk_High"])) > 10
        assert len(str(row["What_Should_Company_Do"])) > 10


def test_delay_prediction():
    """Verify ML delay prediction models output non-negative probabilities and delay days."""
    rca_df = pd.read_csv("data/processed/ai_risk_rca_records.csv")
    predictor = DelayPredictor("data/processed/models")
    predictor.train_models(rca_df)
    pred_df = predictor.predict(rca_df)
    
    assert "Prod_Delay_Prob_Pct" in pred_df.columns
    assert "Expected_Prod_Delay_Days" in pred_df.columns
    assert "Estimated_Business_Impact_Lakhs" in pred_df.columns
    
    assert (pred_df["Prod_Delay_Prob_Pct"] >= 0).all() and (pred_df["Prod_Delay_Prob_Pct"] <= 100).all()
    assert (pred_df["Expected_Prod_Delay_Days"] >= 0).all()
    assert (pred_df["Estimated_Business_Impact_Lakhs"] >= 0).all()


def test_mitigation_and_prioritization():
    """Verify practical mitigation recommendations are generated and prioritized (1 to 4)."""
    delay_df = pd.read_csv("data/processed/ai_risk_delay_predicted_records.csv")
    engine = MitigationEngine()
    mit_df = engine.process_dataset(delay_df)
    
    assert "Recommended_Action" in mit_df.columns
    assert "Action_Priority" in mit_df.columns
    assert set(mit_df["Action_Priority"].unique()).issubset({1, 2, 3, 4})
    
    # Priority 1 must be present for critical risks
    crit_risks = mit_df[mit_df["Risk_Level"] == "CRITICAL"]
    assert (crit_risks["Action_Priority"] == 1).any()


def test_scenario_simulation():
    """Verify What-If simulations run and output delta metrics."""
    sim = ScenarioSimulator(
        data_path="data/processed/ai_risk_mitigated_records.csv",
        models_dir="data/processed/models"
    )
    results = sim.run_all_five_scenarios("data/processed")
    assert len(results) == 5
    for r in results:
        assert "comparison" in r
        assert "recommended_mitigation" in r
        assert len(r["recommended_mitigation"]) > 0


def test_pdf_report_generation():
    """Verify executive PDF report generates and is non-empty."""
    gen = PDFReportGenerator(
        data_path="data/processed/ai_risk_mitigated_records.csv",
        report_path="reports/Tata_Motors_AI_Operations_Risk_Report.pdf"
    )
    path = gen.generate_report()
    assert os.path.exists(path)
    assert os.path.getsize(path) > 10000, "PDF file size too small"
