"""
Feature Engineering Module for Tata Motors AI Operations Risk Detection System.
Calculates domain-specific operational risk indicators across Supplier, Inventory,
Production, Transportation, and Customer Delivery echelons, and synthesizes a
unified Master Operational Risk Dataset.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm


class FeatureEngineer:
    def __init__(self, cleaned_dir="data/cleaned", processed_dir="data/processed"):
        self.cleaned_dir = cleaned_dir
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def engineer_supplier_features(self, df_sup: pd.DataFrame) -> pd.DataFrame:
        """Calculate Supplier Risk Indicators:
        - Supplier delay frequency
        - Supplier reliability index
        - Quality risk
        - Average lead-time deviation
        """
        df = df_sup.copy()
        
        # Lead time deviation
        df["Promised_DT"] = pd.to_datetime(df["Promised_Delivery_Date"])
        df["Actual_DT"] = pd.to_datetime(df["Actual_Delivery_Date"])
        df["Delivery_Deviation_Days"] = (df["Actual_DT"] - df["Promised_DT"]).dt.days.clip(lower=0)
        
        # Aggregated supplier profile statistics
        sup_agg = df.groupby("Supplier_ID").agg(
            Avg_Lead_Time=("Lead_Time_Days", "mean"),
            Std_Lead_Time=("Lead_Time_Days", "std"),
            Avg_Delay_Days=("Delivery_Deviation_Days", "mean"),
            Delay_Frequency=("Delivery_Deviation_Days", lambda x: (x > 0).mean()),
            Mean_OT_Rate=("On_Time_Delivery_Rate", "mean"),
            Mean_Defect_Rate=("Defect_Rate", "mean"),
            Mean_Reliability=("Supplier_Reliability_Score", "mean")
        ).reset_index()
        
        sup_agg["Std_Lead_Time"] = sup_agg["Std_Lead_Time"].fillna(0.5)
        sup_agg["Quality_Risk_Index"] = (sup_agg["Mean_Defect_Rate"] * 10).clip(0, 100).round(2)
        
        df = df.merge(sup_agg, on="Supplier_ID", how="left")
        
        # Lead time deviation from baseline
        df["Lead_Time_Deviation"] = (df["Lead_Time_Days"] - df["Avg_Lead_Time"]).round(2)
        
        # Supplier composite risk factor (0-1)
        df["Supplier_Risk_Factor"] = (
            (100 - df["Mean_Reliability"]) * 0.45 +
            df["Delay_Frequency"] * 100 * 0.35 +
            df["Quality_Risk_Index"] * 0.20
        ).clip(0, 100) / 100.0
        
        return df

    def engineer_inventory_features(self, df_inv: pd.DataFrame) -> pd.DataFrame:
        """Calculate Inventory Risk Indicators:
        - Days of inventory
        - Stockout probability
        - Safety stock gap
        - Inventory coverage ratio
        """
        df = df_inv.copy()
        
        # Safety stock gap (positive means deficit)
        df["Safety_Stock_Gap"] = (df["Safety_Stock"] - df["Current_Stock"]).clip(lower=0)
        df["Safety_Stock_Ratio"] = (df["Current_Stock"] / df["Safety_Stock"].replace(0, 1)).round(3)
        
        # Inventory coverage ratio against lead time demand
        lead_time_demand = df["Daily_Usage"] * df["Lead_Time"]
        df["Inventory_Coverage_Ratio"] = (df["Current_Stock"] / lead_time_demand.replace(0, 1)).round(3)
        
        # Statistical Stockout Probability:
        # Standard deviation of demand during lead time (assuming 25% CV of daily usage)
        demand_std = np.sqrt(df["Lead_Time"]) * (df["Daily_Usage"] * 0.25)
        demand_std = demand_std.replace(0, 1)
        
        # Z-score = (Current_Stock - Lead_Time_Demand) / demand_std
        z_scores = (df["Current_Stock"] - lead_time_demand) / demand_std
        # Stockout probability = 1 - CDF(z)
        df["Stockout_Probability"] = (1.0 - norm.cdf(z_scores)).round(4)
        
        # Critical stockout indicator: Days of inventory < Lead time
        df["Is_Lead_Time_Exceeded"] = df["Days_of_Inventory"] < df["Lead_Time"]
        
        # Inventory Risk Factor (0-1)
        df["Inventory_Risk_Factor"] = (
            df["Stockout_Probability"] * 0.50 +
            (1.0 - df["Inventory_Coverage_Ratio"].clip(0, 1.5) / 1.5) * 0.30 +
            (df["Is_Below_Safety_Stock"].astype(float)) * 0.20
        ).clip(0, 1.0)
        
        return df

    def engineer_production_features(self, df_prd: pd.DataFrame) -> pd.DataFrame:
        """Calculate Production Risk Indicators:
        - Capacity utilization strain
        - Downtime severity
        - Backlog burden
        - Component availability constraint
        """
        df = df_prd.copy()
        
        # Capacity strain index
        df["Capacity_Strain_Index"] = (
            (df["Capacity_Utilization"] / 100.0) * (1.0 + (df["Downtime_Hours"] / 24.0))
        ).round(3)
        
        # Backlog pressure ratio (backlog vs machine capacity)
        df["Backlog_Pressure_Ratio"] = (df["Backlog"] / df["Machine_Capacity"].replace(0, 1)).round(3)
        
        # Component availability numeric penalty
        comp_penalties = {
            "Available": 0.0,
            "Constrained": 0.4,
            "Unavailable - Supplier Delay": 0.95
        }
        df["Component_Availability_Risk"] = df["Component_Availability"].map(comp_penalties).fillna(0.2)
        
        # Production Risk Factor (0-1)
        df["Production_Risk_Factor"] = (
            (df["Capacity_Strain_Index"].clip(0, 2) / 2.0) * 0.30 +
            (df["Downtime_Hours"].clip(0, 24) / 24.0) * 0.25 +
            df["Component_Availability_Risk"] * 0.30 +
            (df["Backlog_Pressure_Ratio"].clip(0, 1.0)) * 0.15
        ).clip(0, 1.0)
        
        return df

    def engineer_transportation_features(self, df_trp: pd.DataFrame) -> pd.DataFrame:
        """Calculate Transportation Risk Indicators:
        - Transportation delay duration
        - Route & weather risk
        - Carrier reliability factor
        - Expected arrival deviation
        """
        df = df_trp.copy()
        
        # Expected vs Actual deviation
        df["Exp_Arr_DT"] = pd.to_datetime(df["Expected_Arrival"])
        df["Act_Arr_DT"] = pd.to_datetime(df["Actual_Arrival"])
        df["Arrival_Deviation_Days"] = (df["Act_Arr_DT"] - df["Exp_Arr_DT"]).dt.days.clip(lower=0)
        
        # Carrier profile reliability
        carrier_perf = df.groupby("Carrier")["Delay_Days"].agg(["mean", "count"]).reset_index()
        carrier_perf.columns = ["Carrier", "Carrier_Avg_Delay", "Carrier_Trips"]
        df = df.merge(carrier_perf, on="Carrier", how="left")
        
        # Weather & Traffic numeric weights
        weather_weights = {
            "Clear": 0.0,
            "Low": 0.1,
            "Severe (Monsoon Flooding / Landslide)": 0.85
        }
        traffic_weights = {
            "Low": 0.0,
            "Medium": 0.35,
            "High (Choke Point Congestion)": 0.75
        }
        df["Weather_Risk_Score"] = df["Weather_Risk"].map(weather_weights).fillna(0.1)
        df["Traffic_Risk_Score"] = df["Traffic_Risk"].map(traffic_weights).fillna(0.2)
        
        # Combined route environmental risk
        df["Route_Risk_Index"] = (df["Weather_Risk_Score"] * 0.6 + df["Traffic_Risk_Score"] * 0.4).round(3)
        
        # Transportation Risk Factor (0-1)
        df["Transportation_Risk_Factor"] = (
            (df["Delay_Days"].clip(0, 10) / 10.0) * 0.45 +
            df["Route_Risk_Index"] * 0.35 +
            (df["Carrier_Avg_Delay"].clip(0, 5) / 5.0) * 0.20
        ).clip(0, 1.0)
        
        return df

    def engineer_delivery_features(self, df_cst: pd.DataFrame) -> pd.DataFrame:
        """Calculate Customer Delivery Risk Indicators:
        - Days remaining until required delivery
        - Expected production completion & transit gap
        - Delivery deadline gap
        - Customer priority weight
        """
        df = df_cst.copy()
        
        req_dt = pd.to_datetime(df["Required_Delivery_Date"])
        plan_dt = pd.to_datetime(df["Planned_Delivery_Date"])
        exp_dt = pd.to_datetime(df["Expected_Delivery_Date"])
        
        # Delivery deadline gap (positive means expected is late)
        df["Deadline_Slack_Days"] = (req_dt - exp_dt).dt.days
        df["Is_Delivery_Late"] = df["Deadline_Slack_Days"] < 0
        df["Delivery_Delay_Duration"] = (-df["Deadline_Slack_Days"]).clip(lower=0)
        
        # Customer priority factor
        prio_weights = {"Normal": 1.0, "High": 1.4, "Critical": 1.8}
        df["Customer_Priority_Weight"] = df["Customer_Priority"].map(prio_weights).fillna(1.0)
        
        # Delivery Risk Factor (0-1)
        df["Delivery_Risk_Factor"] = (
            (df["Delivery_Delay_Duration"].clip(0, 10) / 10.0) * 0.60 +
            (df["Delivery_Risk"].map({"LOW": 0.1, "MEDIUM": 0.4, "HIGH": 0.75, "CRITICAL": 0.95}).fillna(0.3)) * 0.40
        ).clip(0, 1.0)
        
        return df

    def build_master_risk_dataset(self, df_sup, df_inv, df_prd, df_trp, df_cst) -> pd.DataFrame:
        """Synthesize a unified multi-echelon master operational dataset by harmonizing
        cross-functional operational relationships across Tata Motors plants and products.
        """
        # We synthesize an interconnected master dataset reflecting the end-to-end chain:
        # Supplier -> Inbound Transport -> Plant Inventory -> Production Line -> Customer Delivery Order
        master_rows = []
        n_master = min(len(df_prd), len(df_cst), 850)
        
        # Pre-index component inventory lookup by (Plant_ID, Component_Name)
        inv_lookup = {}
        for _, row in df_inv.iterrows():
            key = (row["Plant_ID"], row["Component_Name"])
            if key not in inv_lookup:
                inv_lookup[key] = row
                
        # Pre-index supplier lookup by Component
        sup_lookup = {}
        for _, row in df_sup.iterrows():
            key = row["Component"]
            if key not in sup_lookup:
                sup_lookup[key] = []
            sup_lookup[key].append(row)
            
        for i in range(n_master):
            prd_row = df_prd.iloc[i]
            cst_row = df_cst.iloc[i]
            trp_row = df_trp.iloc[i]
            
            plant_id = prd_row["Plant_ID"]
            comp_name = prd_row["Required_Component"]
            
            # Match inventory
            inv_row = inv_lookup.get((plant_id, comp_name))
            if inv_row is None:
                inv_row = df_inv.iloc[i % len(df_inv)]
                
            # Match supplier
            sup_candidates = sup_lookup.get(comp_name, [])
            if sup_candidates:
                sup_row = sup_candidates[i % len(sup_candidates)]
            else:
                sup_row = df_sup.iloc[i % len(df_sup)]
                
            master_record = {
                "Master_Record_ID": f"OPS_{10000 + i}",
                # Plant & Product
                "Plant_ID": plant_id,
                "Production_Line": prd_row["Production_Line"],
                "Product": prd_row["Product"],
                "Customer_Order_ID": cst_row["Order_ID"],
                "Customer_Name": cst_row["Customer_Name"],
                "Customer_Priority": cst_row["Customer_Priority"],
                "Customer_Priority_Weight": cst_row["Customer_Priority_Weight"],
                "Destination_City": cst_row["Destination_City"],
                
                # Component & Supplier
                "Component_Name": comp_name,
                "Component_Criticality": inv_row["Criticality"],
                "Supplier_ID": sup_row["Supplier_ID"],
                "Supplier_Name": sup_row["Supplier_Name"],
                "Supplier_Lead_Time": sup_row["Lead_Time_Days"],
                "Supplier_Reliability_Score": sup_row["Supplier_Reliability_Score"],
                "Supplier_Delay_Days": sup_row["Delivery_Deviation_Days"],
                "Supplier_Delay_Frequency": sup_row["Delay_Frequency"],
                "Supplier_Defect_Rate": sup_row["Defect_Rate"],
                "Supplier_Risk_Factor": sup_row["Supplier_Risk_Factor"],
                
                # Inventory
                "Inventory_ID": inv_row["Inventory_ID"],
                "Current_Stock": inv_row["Current_Stock"],
                "Safety_Stock": inv_row["Safety_Stock"],
                "Daily_Usage": inv_row["Daily_Usage"],
                "Days_of_Inventory": inv_row["Days_of_Inventory"],
                "Safety_Stock_Gap": inv_row["Safety_Stock_Gap"],
                "Stockout_Probability": inv_row["Stockout_Probability"],
                "Is_Below_Safety_Stock": inv_row["Is_Below_Safety_Stock"],
                "Inventory_Risk_Factor": inv_row["Inventory_Risk_Factor"],
                
                # Production
                "Production_ID": prd_row["Production_ID"],
                "Machine_Capacity": prd_row["Machine_Capacity"],
                "Planned_Production": prd_row["Planned_Production"],
                "Actual_Production": prd_row["Actual_Production"],
                "Capacity_Utilization": prd_row["Capacity_Utilization"],
                "Downtime_Hours": prd_row["Downtime_Hours"],
                "Backlog": prd_row["Backlog"],
                "Component_Availability": prd_row["Component_Availability"],
                "Production_Status": prd_row["Production_Status"],
                "Production_Risk_Factor": prd_row["Production_Risk_Factor"],
                
                # Transportation
                "Shipment_ID": trp_row["Shipment_ID"],
                "Carrier": trp_row["Carrier"],
                "Transport_Mode": trp_row["Transport_Mode"],
                "Route": trp_row["Route"],
                "Transit_Delay_Days": trp_row["Delay_Days"],
                "Traffic_Risk": trp_row["Traffic_Risk"],
                "Weather_Risk": trp_row["Weather_Risk"],
                "Route_Risk_Index": trp_row["Route_Risk_Index"],
                "Transportation_Risk_Factor": trp_row["Transportation_Risk_Factor"],
                
                # Customer Delivery
                "Required_Delivery_Date": cst_row["Required_Delivery_Date"],
                "Planned_Delivery_Date": cst_row["Planned_Delivery_Date"],
                "Expected_Delivery_Date": cst_row["Expected_Delivery_Date"],
                "Deadline_Slack_Days": cst_row["Deadline_Slack_Days"],
                "Delivery_Delay_Duration": cst_row["Delivery_Delay_Duration"],
                "Delivery_Risk_Factor": cst_row["Delivery_Risk_Factor"],
                "Order_Quantity": cst_row["Order_Quantity"]
            }
            master_rows.append(master_record)
            
        df_master = pd.DataFrame(master_rows)
        return df_master

    def run_feature_engineering_pipeline(self):
        """Execute feature engineering across all datasets."""
        print("Executing Feature Engineering Pipeline...")
        
        sup_df = pd.read_csv(os.path.join(self.cleaned_dir, "cleaned_suppliers.csv"))
        inv_df = pd.read_csv(os.path.join(self.cleaned_dir, "cleaned_inventory.csv"))
        prd_df = pd.read_csv(os.path.join(self.cleaned_dir, "cleaned_production.csv"))
        trp_df = pd.read_csv(os.path.join(self.cleaned_dir, "cleaned_transportation.csv"))
        cst_df = pd.read_csv(os.path.join(self.cleaned_dir, "cleaned_customer_orders.csv"))
        
        fe_sup = self.engineer_supplier_features(sup_df)
        fe_inv = self.engineer_inventory_features(inv_df)
        fe_prd = self.engineer_production_features(prd_df)
        fe_trp = self.engineer_transportation_features(trp_df)
        fe_cst = self.engineer_delivery_features(cst_df)
        
        master_df = self.build_master_risk_dataset(fe_sup, fe_inv, fe_prd, fe_trp, fe_cst)
        
        # Save enriched datasets
        fe_sup.to_csv(os.path.join(self.processed_dir, "fe_suppliers.csv"), index=False)
        fe_inv.to_csv(os.path.join(self.processed_dir, "fe_inventory.csv"), index=False)
        fe_prd.to_csv(os.path.join(self.processed_dir, "fe_production.csv"), index=False)
        fe_trp.to_csv(os.path.join(self.processed_dir, "fe_transportation.csv"), index=False)
        fe_cst.to_csv(os.path.join(self.processed_dir, "fe_customer_orders.csv"), index=False)
        master_df.to_csv(os.path.join(self.processed_dir, "master_risk_records.csv"), index=False)
        
        print(f"Feature Engineering complete.")
        print(f" -> Master Operational Dataset created: {len(master_df)} rows, {len(master_df.columns)} features.")
        return master_df


if __name__ == "__main__":
    fe = FeatureEngineer()
    fe.run_feature_engineering_pipeline()
