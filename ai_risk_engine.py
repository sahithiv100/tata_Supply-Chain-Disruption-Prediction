"""
AI Risk Detection and Scoring Engine for Tata Motors Operations.
Integrates early-warning heuristics, machine learning anomaly detection (Isolation Forest),
multi-echelon cascading risk analysis, and normalized explainable risk scoring (0-100).
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class AIRiskEngine:
    def __init__(self, processed_dir="data/processed"):
        self.processed_dir = processed_dir
        self.anomaly_model = IsolationForest(
            n_estimators=100,
            contamination=0.08,
            random_state=42
        )
        
    def detect_early_warnings(self, row: pd.Series) -> list:
        """Evaluate operational early warning signals across echelons."""
        warnings = []
        
        # 1. Supplier early warning signals
        if row["Supplier_Delay_Days"] >= 3:
            warnings.append(f"Supplier delivery delay of {int(row['Supplier_Delay_Days'])} days detected from {row['Supplier_Name']}")
        if row["Supplier_Delay_Frequency"] > 0.45:
            warnings.append(f"Supplier historical delay frequency high ({round(row['Supplier_Delay_Frequency']*100, 1)}%)")
        if row["Supplier_Defect_Rate"] > 4.5:
            warnings.append(f"Supplier quality warning: defect rate elevated at {row['Supplier_Defect_Rate']}%")
            
        # 2. Inventory early warning signals
        if row["Days_of_Inventory"] < 2.0:
            warnings.append(f"Stockout imminent: Days of inventory ({row['Days_of_Inventory']}d) < 2.0 days")
        elif row["Days_of_Inventory"] < row["Supplier_Lead_Time"] and row["Is_Below_Safety_Stock"]:
            warnings.append(f"Inventory coverage gap: Days of inventory ({row['Days_of_Inventory']}d) < Lead time ({row['Supplier_Lead_Time']}d)")
        if row["Is_Below_Safety_Stock"] and row["Safety_Stock_Gap"] > 100:
            warnings.append(f"Stock below safety threshold (Deficit: {int(row['Safety_Stock_Gap'])} units)")
            
        # 3. Production early warning signals
        if "Unavailable" in str(row["Component_Availability"]):
            warnings.append(f"Line stoppage risk: Required component '{row['Component_Name']}' is unavailable")
        elif "Constrained" in str(row["Component_Availability"]):
            warnings.append(f"Production component availability constrained")
        if row["Capacity_Utilization"] > 95.0:
            warnings.append(f"Extreme capacity strain ({row['Capacity_Utilization']}%) on {row['Production_Line']}")
        if row["Downtime_Hours"] > 8.0:
            warnings.append(f"Excessive unscheduled downtime: {row['Downtime_Hours']} hours")
        if row["Backlog"] > 55:
            warnings.append(f"Heavy production backlog: {int(row['Backlog'])} units")
            
        # 4. Transportation early warning signals
        if row["Transit_Delay_Days"] >= 3:
            warnings.append(f"Logistics bottleneck: {int(row['Transit_Delay_Days'])} days delay via {row['Carrier']}")
        if "Severe" in str(row["Weather_Risk"]):
            warnings.append(f"Severe environmental disruption on route {row['Route']}")
        if "High" in str(row["Traffic_Risk"]):
            warnings.append(f"Highway freight choke point congestion")
            
        # 5. Customer delivery early warning signals
        if row["Deadline_Slack_Days"] < 0:
            warnings.append(f"Customer SLA already breached by {-int(row['Deadline_Slack_Days'])} days")
        elif row["Deadline_Slack_Days"] <= 1 and row["Customer_Priority"] in ["High", "Critical"]:
            warnings.append(f"Critical delivery deadline window: only {int(row['Deadline_Slack_Days'])} days slack remaining")
            
        return warnings

    def fit_anomaly_detector(self, df: pd.DataFrame):
        """Fit Isolation Forest on operational telemetry features."""
        features = [
            "Supplier_Delay_Days", "Supplier_Risk_Factor", "Days_of_Inventory",
            "Stockout_Probability", "Capacity_Utilization", "Downtime_Hours",
            "Transit_Delay_Days", "Route_Risk_Index"
        ]
        X = df[features].fillna(0)
        self.anomaly_model.fit(X)
        scores = self.anomaly_model.decision_function(X)
        predictions = self.anomaly_model.predict(X)
        
        # Invert decision score so higher = more anomalous (0 to 1)
        anomaly_scores = (scores.max() - scores) / (scores.max() - scores.min() + 1e-6)
        return predictions == -1, anomaly_scores.round(3)

    def calculate_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate explainable 0-100 Risk Scores based on:
        Risk Score = Probability x Impact x Urgency (Normalized with calibrated domain thresholds)
        """
        is_anomaly, anomaly_scores = self.fit_anomaly_detector(df)
        df["Is_Operational_Anomaly"] = is_anomaly
        df["Anomaly_Risk_Score"] = anomaly_scores
        
        risk_scores = []
        risk_levels = []
        prob_factors = []
        impact_factors = []
        urgency_factors = []
        reasons = []
        early_warnings_list = []
        risk_types = []
        
        for idx, row in df.iterrows():
            warnings = self.detect_early_warnings(row)
            early_warnings_list.append("; ".join(warnings) if warnings else "Normal operational telemetry")
            
            # --- 1. PROBABILITY FACTOR (0.05 to 0.95) ---
            prob_components = [
                row["Stockout_Probability"] * 0.28,
                row["Supplier_Risk_Factor"] * 0.22,
                min(1.0, row["Downtime_Hours"] / 16.0) * 0.18,
                min(1.0, row["Transit_Delay_Days"] / 6.0) * 0.18,
                row["Anomaly_Risk_Score"] * 0.14
            ]
            prob = sum(prob_components)
            if "Unavailable" in str(row["Component_Availability"]):
                prob += 0.25
            prob = float(np.clip(prob, 0.05, 0.96))
            prob_factors.append(round(prob, 3))
            
            # --- 2. IMPACT FACTOR (0.10 to 0.98) ---
            crit_map = {"Critical": 1.0, "High": 0.70, "Medium": 0.45, "Low": 0.25}
            comp_crit = crit_map.get(row["Component_Criticality"], 0.5)
            
            prio_map = {"Critical": 1.0, "High": 0.75, "Normal": 0.40}
            cust_prio = prio_map.get(row["Customer_Priority"], 0.4)
            
            vol_impact = min(1.0, row["Order_Quantity"] / 200.0)
            
            impact = (
                0.45 * comp_crit +
                0.35 * cust_prio +
                0.20 * vol_impact
            )
            impact = float(np.clip(impact, 0.10, 0.98))
            impact_factors.append(round(impact, 3))
            
            # --- 3. URGENCY FACTOR (0.10 to 0.98) ---
            slack = row["Deadline_Slack_Days"]
            if slack < 0:
                urgency = 0.95
            elif slack <= 2:
                urgency = 0.85
            elif slack <= 4:
                urgency = 0.65
            elif slack <= 7:
                urgency = 0.40
            else:
                urgency = 0.15
                
            if row["Days_of_Inventory"] < 1.8:
                urgency = min(0.98, urgency + 0.20)
            elif row["Days_of_Inventory"] < 3.5:
                urgency = min(0.95, urgency + 0.10)
                
            urgency = float(np.clip(urgency, 0.10, 0.98))
            urgency_factors.append(round(urgency, 3))
            
            # --- 4. SEVERITY CASCADE EVALUATION ---
            has_sup_delay = row["Supplier_Delay_Days"] >= 3
            has_inv_crisis = row["Days_of_Inventory"] < 2.0 or (row["Is_Below_Safety_Stock"] and row["Days_of_Inventory"] < row["Supplier_Lead_Time"])
            has_prod_starvation = "Unavailable" in str(row["Component_Availability"]) or row["Downtime_Hours"] >= 10
            has_transit_disruption = row["Transit_Delay_Days"] >= 3 or "Severe" in str(row["Weather_Risk"])
            has_customer_threat = slack <= 2 and row["Customer_Priority"] in ["High", "Critical"]
            
            disruptions = [has_sup_delay, has_inv_crisis, has_prod_starvation, has_transit_disruption, has_customer_threat]
            disruption_count = sum(disruptions)
            
            base_score = (0.42 * prob + 0.33 * impact + 0.25 * urgency) * 100.0
            
            if disruption_count >= 3:
                cascade_mult = 1.30
                r_type = "Multi-Echelon Cascade Risk"
            elif disruption_count == 2:
                cascade_mult = 1.15
                r_type = "Dual-Echelon Compound Risk"
            else:
                cascade_mult = 1.0
                if has_sup_delay:
                    r_type = "Supplier Disruption Risk"
                elif has_inv_crisis:
                    r_type = "Inventory Stockout Risk"
                elif has_prod_starvation:
                    r_type = "Production Bottleneck Risk"
                elif has_transit_disruption:
                    r_type = "Logistics Transit Risk"
                elif has_customer_threat:
                    r_type = "Customer Delivery Breach Risk"
                else:
                    r_type = "Routine Operational Baseline"
                    
            risk_types.append(r_type)
            final_raw = base_score * cascade_mult
            
            # Calibrate 4-tier risk categories
            if disruption_count >= 3 and row["Component_Criticality"] in ["Critical", "High"]:
                final_score = round(min(98.5, max(76.0, final_raw)), 1)
            elif (disruption_count >= 2) or (has_sup_delay and has_inv_crisis) or (has_prod_starvation and has_customer_threat):
                final_score = round(min(74.0, max(52.0, final_raw)), 1)
            elif disruption_count == 1 and (prob > 0.38 or urgency > 0.50):
                final_score = round(min(49.0, max(26.0, final_raw * 0.72)), 1)
            else:
                final_score = round(min(24.0, max(6.0, final_raw * 0.38)), 1)
                
            risk_scores.append(final_score)
            
            # Classification
            if final_score >= 75.0:
                level = "CRITICAL"
            elif final_score >= 50.0:
                level = "HIGH"
            elif final_score >= 25.0:
                level = "MEDIUM"
            else:
                level = "LOW"
            risk_levels.append(level)
            
            # Explainable Factor Reason
            reason_parts = []
            if disruption_count >= 3:
                reason_parts.append(f"Multi-echelon cascade across {disruption_count} operational domains")
            if row["Supplier_Delay_Days"] > 0:
                reason_parts.append(f"Supplier delay {int(row['Supplier_Delay_Days'])}d ({row['Supplier_Name']})")
            if row["Days_of_Inventory"] < row["Supplier_Lead_Time"]:
                reason_parts.append(f"Stock coverage ({row['Days_of_Inventory']}d) < Supplier lead time ({row['Supplier_Lead_Time']}d)")
            if row["Downtime_Hours"] > 6:
                reason_parts.append(f"Line downtime {row['Downtime_Hours']}h on {row['Production_Line']}")
            if row["Transit_Delay_Days"] > 0:
                reason_parts.append(f"Transit delayed {int(row['Transit_Delay_Days'])}d ({row['Carrier']})")
            if slack <= 3:
                reason_parts.append(f"Customer deadline gap: {int(slack)}d for {row['Customer_Name']} ({row['Customer_Priority']})")
                
            if not reason_parts:
                reason_parts.append("Normal operational tolerance and adequate safety buffers")
                
            reasons.append(" | ".join(reason_parts))
            
        df["Probability_Factor"] = prob_factors
        df["Impact_Factor"] = impact_factors
        df["Urgency_Factor"] = urgency_factors
        df["Risk_Score"] = risk_scores
        df["Risk_Level"] = risk_levels
        df["Risk_Type"] = risk_types
        df["Risk_Reason"] = reasons
        df["Early_Warning_Signals"] = early_warnings_list
        
        return df

    def run_risk_detection_pipeline(self):
        """Execute risk detection engine and generate scored dataset."""
        print("Executing AI Risk Detection & Early Warning Engine...")
        master_path = os.path.join(self.processed_dir, "master_risk_records.csv")
        df_master = pd.read_csv(master_path)
        
        scored_df = self.calculate_risk_scores(df_master)
        
        output_path = os.path.join(self.processed_dir, "ai_risk_scored_records.csv")
        scored_df.to_csv(output_path, index=False)
        
        dist = scored_df["Risk_Level"].value_counts().to_dict()
        print(f"Risk Detection complete. Records analyzed: {len(scored_df)}")
        print(f"Risk Distribution: CRITICAL: {dist.get('CRITICAL', 0)} | HIGH: {dist.get('HIGH', 0)} | MEDIUM: {dist.get('MEDIUM', 0)} | LOW: {dist.get('LOW', 0)}")
        print(f"Saved scored operations to: {output_path}")
        return scored_df


if __name__ == "__main__":
    engine = AIRiskEngine()
    engine.run_risk_detection_pipeline()
