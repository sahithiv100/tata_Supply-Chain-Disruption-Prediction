"""
Scenario Analysis & What-If Simulation Engine for Tata Motors Operations.
Simulates operational stress tests:
- Scenario 1: Supplier delayed by 5 days
- Scenario 2: Inventory decreases by 30%
- Scenario 3: Production capacity decreases by 20%
- Scenario 4: Transportation time increases by 3 days
- Scenario 5: Demand increases by 15%
Calculates New Risk Scores, Production Delay Probabilities, Expected Delays, Customer Impacts, and Mitigations.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd


class ScenarioSimulator:
    def __init__(self, data_path="data/processed/ai_risk_mitigated_records.csv", models_dir="data/processed/models"):
        self.data_path = data_path
        self.models_dir = models_dir
        self.df_base = None
        
        # Load ML models
        self.clf_prod = joblib.load(os.path.join(models_dir, "clf_prod.joblib"))
        self.reg_prod = joblib.load(os.path.join(models_dir, "reg_prod.joblib"))
        self.clf_delv = joblib.load(os.path.join(models_dir, "clf_delv.joblib"))
        self.reg_delv = joblib.load(os.path.join(models_dir, "reg_delv.joblib"))
        
        self.feature_cols = [
            "Supplier_Delay_Days", "Supplier_Risk_Factor", "Supplier_Reliability_Score",
            "Days_of_Inventory", "Stockout_Probability", "Capacity_Utilization",
            "Downtime_Hours", "Backlog", "Transit_Delay_Days", "Route_Risk_Index",
            "Deadline_Slack_Days", "Order_Quantity"
        ]

    def load_base_data(self):
        if self.df_base is None:
            self.df_base = pd.read_csv(self.data_path)
        return self.df_base.copy()

    def simulate(self, 
                 scenario_name: str, 
                 sup_delay_delta: float = 0.0, 
                 inv_pct_change: float = 0.0, 
                 cap_pct_change: float = 0.0, 
                 transit_delta: float = 0.0, 
                 demand_pct_change: float = 0.0) -> dict:
        """Run custom or standard stress-test simulation across all operational records."""
        df = self.load_base_data()
        
        # 1. Apply scenario shocks
        if sup_delay_delta != 0:
            df["Supplier_Delay_Days"] = (df["Supplier_Delay_Days"] + sup_delay_delta).clip(lower=0)
            df["Supplier_Risk_Factor"] = (df["Supplier_Risk_Factor"] + (sup_delay_delta * 0.06)).clip(0.1, 1.0)
            
        if inv_pct_change != 0:
            # e.g. -0.30 reduces inventory by 30%
            factor = 1.0 + inv_pct_change
            df["Current_Stock"] = (df["Current_Stock"] * factor).clip(lower=0)
            df["Days_of_Inventory"] = (df["Days_of_Inventory"] * factor).clip(lower=0.2).round(2)
            df["Safety_Stock_Gap"] = (df["Safety_Stock"] - df["Current_Stock"]).clip(lower=0)
            df["Stockout_Probability"] = np.clip(df["Stockout_Probability"] + (abs(inv_pct_change) * 0.40), 0.05, 0.98).round(3)
            
        if cap_pct_change != 0:
            # e.g. -0.20 decreases available capacity, meaning capacity utilization surges or downtime rises
            factor = 1.0 + cap_pct_change
            if "Machine_Capacity" in df.columns:
                df["Machine_Capacity"] = (df["Machine_Capacity"] * factor).clip(lower=50)
            df["Capacity_Utilization"] = (df["Capacity_Utilization"] / factor).clip(50.0, 99.8).round(1)
            df["Backlog"] = (df["Backlog"] * (1.0 / factor)).clip(0, 150).astype(int)
            df["Downtime_Hours"] = (df["Downtime_Hours"] + (abs(cap_pct_change) * 12.0)).clip(0, 24).round(1)
            
        if transit_delta != 0:
            df["Transit_Delay_Days"] = (df["Transit_Delay_Days"] + transit_delta).clip(lower=0)
            df["Route_Risk_Index"] = (df["Route_Risk_Index"] + (transit_delta * 0.08)).clip(0.1, 1.0).round(2)
            
        if demand_pct_change != 0:
            # e.g. +0.15 increases order quantity and shrinks deadline slack
            factor = 1.0 + demand_pct_change
            df["Order_Quantity"] = (df["Order_Quantity"] * factor).astype(int)
            df["Daily_Usage"] = (df["Daily_Usage"] * factor).astype(int)
            df["Days_of_Inventory"] = (df["Current_Stock"] / df["Daily_Usage"].replace(0, 1)).clip(lower=0.2).round(2)
            df["Deadline_Slack_Days"] = (df["Deadline_Slack_Days"] - (demand_pct_change * 8)).round(1)

        # 2. Re-calculate Machine Learning Predictions
        X = df[self.feature_cols].fillna(0)
        prod_probs = self.clf_prod.predict_proba(X)[:, 1] * 100.0
        prod_days = np.where(prod_probs > 35.0, np.maximum(1.0, self.reg_prod.predict(X)), 0.0)
        delv_probs = self.clf_delv.predict_proba(X)[:, 1] * 100.0
        delv_days = np.where(delv_probs > 30.0, np.maximum(1.0, self.reg_delv.predict(X)), 0.0)
        
        df["Sim_Prod_Delay_Prob_Pct"] = np.round(prod_probs, 1)
        df["Sim_Expected_Prod_Delay_Days"] = np.round(prod_days, 1)
        df["Sim_Delivery_Delay_Prob_Pct"] = np.round(delv_probs, 1)
        df["Sim_Expected_Delivery_Delay_Days"] = np.round(delv_days, 1)
        
        # 3. Re-calculate Financial Business Impact (INR Lakhs)
        financial_impacts = []
        for _, row in df.iterrows():
            p_delay = row["Sim_Expected_Prod_Delay_Days"]
            d_delay = row["Sim_Expected_Delivery_Delay_Days"]
            order_qty = row["Order_Quantity"]
            prio_mult = 1.6 if row["Customer_Priority"] == "Critical" else (1.25 if row["Customer_Priority"] == "High" else 1.0)
            
            if p_delay == 0 and d_delay == 0:
                cost = 0.0
            else:
                line_idle = p_delay * 2.8
                cap_hold = order_qty * 18.0 * 0.0015 * d_delay
                sla_pen = d_delay * 0.75 * prio_mult
                cost = round((line_idle + cap_hold + sla_pen) * prio_mult, 2)
            financial_impacts.append(cost)
            
        df["Sim_Business_Impact_Lakhs"] = financial_impacts
        
        # 4. Re-calculate Simulated Risk Scores
        # Delta score applied to base risk score
        shock_weight = (
            (sup_delay_delta * 3.5) +
            (abs(inv_pct_change) * 35.0) +
            (abs(cap_pct_change) * 32.0) +
            (transit_delta * 4.0) +
            (demand_pct_change * 25.0)
        )
        sim_scores = (df["Risk_Score"] + shock_weight).clip(5.0, 99.5).round(1)
        df["Sim_Risk_Score"] = sim_scores
        
        sim_levels = []
        for s in sim_scores:
            if s >= 75.0: sim_levels.append("CRITICAL")
            elif s >= 50.0: sim_levels.append("HIGH")
            elif s >= 25.0: sim_levels.append("MEDIUM")
            else: sim_levels.append("LOW")
        df["Sim_Risk_Level"] = sim_levels
        
        # 5. Aggregate Summary Metrics
        base_df = self.load_base_data()
        
        base_crit = int((base_df["Risk_Level"] == "CRITICAL").sum())
        sim_crit = int((df["Sim_Risk_Level"] == "CRITICAL").sum())
        base_high = int((base_df["Risk_Level"] == "HIGH").sum())
        sim_high = int((df["Sim_Risk_Level"] == "HIGH").sum())
        
        base_avg_score = round(float(base_df["Risk_Score"].mean()), 1)
        sim_avg_score = round(float(df["Sim_Risk_Score"].mean()), 1)
        
        base_avg_prod_delay = round(float(base_df["Expected_Prod_Delay_Days"].mean()), 2)
        sim_avg_prod_delay = round(float(df["Sim_Expected_Prod_Delay_Days"].mean()), 2)
        
        base_avg_delv_delay = round(float(base_df["Expected_Delivery_Delay_Days"].mean()), 2)
        sim_avg_delv_delay = round(float(df["Sim_Expected_Delivery_Delay_Days"].mean()), 2)
        
        base_total_financial = round(float(base_df["Estimated_Business_Impact_Lakhs"].sum()), 2)
        sim_total_financial = round(float(df["Sim_Business_Impact_Lakhs"].sum()), 2)
        financial_delta = round(sim_total_financial - base_total_financial, 2)
        
        # Scenario Recommended Mitigations
        mitigations = self._generate_scenario_mitigations(scenario_name, sup_delay_delta, inv_pct_change, cap_pct_change, transit_delta, demand_pct_change)
        
        result_payload = {
            "scenario_name": scenario_name,
            "parameters": {
                "supplier_delay_delta_days": sup_delay_delta,
                "inventory_change_pct": f"{int(inv_pct_change*100)}%",
                "capacity_change_pct": f"{int(cap_pct_change*100)}%",
                "transit_delay_delta_days": transit_delta,
                "demand_change_pct": f"{int(demand_pct_change*100)}%"
            },
            "comparison": {
                "avg_risk_score": {"baseline": base_avg_score, "simulated": sim_avg_score, "delta": round(sim_avg_score - base_avg_score, 1)},
                "critical_risks_count": {"baseline": base_crit, "simulated": sim_crit, "delta": sim_crit - base_crit},
                "high_risks_count": {"baseline": base_high, "simulated": sim_high, "delta": sim_high - base_high},
                "expected_prod_delay_days": {"baseline": base_avg_prod_delay, "simulated": sim_avg_prod_delay, "delta": round(sim_avg_prod_delay - base_avg_prod_delay, 2)},
                "expected_delivery_delay_days": {"baseline": base_avg_delv_delay, "simulated": sim_avg_delv_delay, "delta": round(sim_avg_delv_delay - base_avg_delv_delay, 2)},
                "total_financial_exposure_lakhs": {"baseline": base_total_financial, "simulated": sim_total_financial, "delta": financial_delta}
            },
            "recommended_mitigation": mitigations
        }
        
        return result_payload, df

    def _generate_scenario_mitigations(self, name, sup_d, inv_c, cap_c, trn_d, dem_c) -> list:
        """Synthesize specialized mitigations tailored to the simulated shock."""
        mits = []
        if sup_d > 0:
            mits.append("Activate secondary suppliers (Brakes India, Valeo, Tata AutoComp) to absorb 50% replenishment volume.")
            mits.append("Issue emergency air-freight and hot-shot transport authorization for tier-1 suppliers.")
            mits.append("Implement supplier resident engineering audit at delayed vendor facilities.")
        if inv_c < 0:
            mits.append("Execute rapid inter-plant inventory transfers from Sanand, Dharwad, and Jamshedpur hubs.")
            mits.append("Raise ERP dynamic reorder points by 35% and trigger automated advance procurement POs.")
            mits.append("Ration on-hand components exclusively to High and Critical priority customer vehicle batches.")
        if cap_c < 0:
            mits.append("Shift 30% production volume to parallel lines across Pune and Sanand flexible manufacturing cells.")
            mits.append("Schedule continuous 3rd overtime shifts and deploy mobile preventive maintenance strike-teams.")
            mits.append("De-bottleneck robotic welding and battery pack integration cells with auxiliary tooling.")
        if trn_d > 0:
            mits.append("Reroute all critical in-transit shipments to dedicated Golden Quadrilateral bypass expressways.")
            mits.append("Upgrade high-value shipments from road transit to Blue Dart Aviation air-freight cargo.")
            mits.append("Pre-position regional hub buffer inventory at Mumbai and Ahmedabad distribution nodes.")
        if dem_c > 0:
            mits.append("Prioritize production line staging for premium high-margin vehicle lines (Nexon EV, Safari, Prima Heavy Truck).")
            mits.append("Pre-negotiate flexible delivery schedules and partial lot deliveries with enterprise fleet clients.")
            mits.append("Expand supplier production commitments with guaranteed minimum volume contracts.")
        return mits

    def run_all_five_scenarios(self, output_dir="data/processed"):
        """Execute the 5 required prompt scenarios and persist comprehensive results."""
        print("Executing 5 Operational What-If Scenarios...")
        scenarios = [
            {"name": "Scenario 1: Supplier Delayed by 5 Days", "sup_d": 5.0, "inv_c": 0.0, "cap_c": 0.0, "trn_d": 0.0, "dem_c": 0.0},
            {"name": "Scenario 2: Inventory Decreases by 30%", "sup_d": 0.0, "inv_c": -0.30, "cap_c": 0.0, "trn_d": 0.0, "dem_c": 0.0},
            {"name": "Scenario 3: Production Capacity Decreases by 20%", "sup_d": 0.0, "inv_c": 0.0, "cap_c": -0.20, "trn_d": 0.0, "dem_c": 0.0},
            {"name": "Scenario 4: Transportation Time Increases by 3 Days", "sup_d": 0.0, "inv_c": 0.0, "cap_c": 0.0, "trn_d": 3.0, "dem_c": 0.0},
            {"name": "Scenario 5: Demand Increases by 15%", "sup_d": 0.0, "inv_c": 0.0, "cap_c": 0.0, "trn_d": 0.0, "dem_c": 0.15}
        ]
        
        all_results = []
        for sc in scenarios:
            res, _ = self.simulate(
                scenario_name=sc["name"],
                sup_delay_delta=sc["sup_d"],
                inv_pct_change=sc["inv_c"],
                cap_pct_change=sc["cap_c"],
                transit_delta=sc["trn_d"],
                demand_pct_change=sc["dem_c"]
            )
            all_results.append(res)
            print(f" -> {sc['name']}: Avg Score {res['comparison']['avg_risk_score']['baseline']} -> {res['comparison']['avg_risk_score']['simulated']} | Critical Risks: +{res['comparison']['critical_risks_count']['delta']}")
            
        json_path = os.path.join(output_dir, "scenario_simulation_results.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=4)
            
        print(f"Scenario simulations complete. Saved to: {json_path}")
        return all_results


if __name__ == "__main__":
    sim = ScenarioSimulator()
    sim.run_all_five_scenarios()
