"""
Data Cleaning and Validation Pipeline for Tata Motors AI Operations Risk System.
Performs data validation, anomaly correction, entity standardization, and data quality reporting.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

CANONICAL_SUPPLIERS = {
    "SUP_101": "Bosch India Chassis Systems",
    "SUP_102": "Tata AutoComp Systems Ltd",
    "SUP_103": "Exide Energy Solutions",
    "SUP_104": "Endurance Technologies Ltd",
    "SUP_105": "Continental Automotive India",
    "SUP_106": "Uno Minda Electronic Systems",
    "SUP_107": "Bharat Forge Powertrain Div",
    "SUP_108": "Motherson Sumi Wiring Systems",
    "SUP_109": "Lucas TVS Electric Drives",
    "SUP_110": "Valeo India Thermal Systems",
    "SUP_111": "Gabriel India Suspension",
    "SUP_112": "Garrett Motion Turbochargers",
    "SUP_113": "ZF Commercial Vehicle Control",
    "SUP_114": "Denso Kirloskar Electronics",
    "SUP_115": "JTEKT India Steering",
    "SUP_116": "Brakes India Private Ltd",
    "SUP_117": "Sona Comstar E-Drivetrain",
    "SUP_118": "Aptiv Connection Systems",
    "SUP_119": "Subros Thermal Solutions",
    "SUP_120": "Tata Steel Tubes & Axles"
}

VALID_PLANTS = ["PLANT_PUN", "PLANT_SND", "PLANT_JSR", "PLANT_PNT", "PLANT_DHW"]


class DataCleaner:
    def __init__(self, raw_dir="data/raw", cleaned_dir="data/cleaned"):
        self.raw_dir = raw_dir
        self.cleaned_dir = cleaned_dir
        os.makedirs(self.cleaned_dir, exist_ok=True)
        self.quality_stats = {}
        
    def clean_supplier_data(self, df: pd.DataFrame) -> (pd.DataFrame, dict):
        total_records = len(df)
        stats = {
            "dataset": "Suppliers",
            "total_records": total_records,
            "duplicate_records": 0,
            "missing_values": 0,
            "invalid_records": 0,
            "corrected_records": 0,
            "final_usable_records": 0
        }
        
        # 1. Duplicates
        duplicates = df.duplicated().sum()
        stats["duplicate_records"] = int(duplicates)
        df = df.drop_duplicates().copy()
        
        # 2. Entity standardization (trim whitespace, title casing, canonical mapping)
        if "Supplier_Name" in df.columns:
            df["Supplier_Name"] = df["Supplier_Name"].astype(str).str.strip()
            # Map canonical names via Supplier_ID
            df["Supplier_Name"] = df["Supplier_ID"].map(CANONICAL_SUPPLIERS).fillna(df["Supplier_Name"])
            
        if "Component" in df.columns:
            df["Component"] = df["Component"].astype(str).str.strip()
            
        # 3. Missing values
        missing_count = int(df.isnull().sum().sum())
        stats["missing_values"] = missing_count
        
        if "Cost_Per_Unit" in df.columns:
            comp_cost_median = df.groupby("Component")["Cost_Per_Unit"].transform("median")
            df["Cost_Per_Unit"] = df["Cost_Per_Unit"].fillna(comp_cost_median).fillna(25000)
            
        if "On_Time_Delivery_Rate" in df.columns:
            df["On_Time_Delivery_Rate"] = df["On_Time_Delivery_Rate"].fillna(df["On_Time_Delivery_Rate"].median())
            
        # 4. Abnormal lead times & negative numbers
        abnormal_lead_mask = (df["Lead_Time_Days"] < 1) | (df["Lead_Time_Days"] > 60)
        abnormal_count = abnormal_lead_mask.sum()
        stats["invalid_records"] += int(abnormal_count)
        if abnormal_count > 0:
            stats["corrected_records"] += int(abnormal_count)
            median_lead = df.loc[~abnormal_lead_mask, "Lead_Time_Days"].median()
            df.loc[abnormal_lead_mask, "Lead_Time_Days"] = int(median_lead if not np.isnan(median_lead) else 8)
            
        # 5. Date validation and standardization
        date_cols = ["Promised_Delivery_Date", "Actual_Delivery_Date"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
                
        # Fill any unparseable dates
        df["Promised_Delivery_Date"] = df["Promised_Delivery_Date"].fillna("2026-09-15")
        df["Actual_Delivery_Date"] = df["Actual_Delivery_Date"].fillna(df["Promised_Delivery_Date"])
        
        # 6. Recalculate / validate defect rate
        if "Quality_Rate" in df.columns:
            df["Quality_Rate"] = df["Quality_Rate"].clip(50.0, 100.0)
            df["Defect_Rate"] = (100.0 - df["Quality_Rate"]).round(2)
            
        stats["final_usable_records"] = len(df)
        return df, stats

    def clean_inventory_data(self, df: pd.DataFrame) -> (pd.DataFrame, dict):
        total_records = len(df)
        stats = {
            "dataset": "Inventory",
            "total_records": total_records,
            "duplicate_records": 0,
            "missing_values": 0,
            "invalid_records": 0,
            "corrected_records": 0,
            "final_usable_records": 0
        }
        
        # 1. Duplicates
        duplicates = df.duplicated().sum()
        stats["duplicate_records"] = int(duplicates)
        df = df.drop_duplicates().copy()
        
        # 2. Missing values
        missing_count = int(df.isnull().sum().sum())
        stats["missing_values"] = missing_count
        
        if "Incoming_Quantity" in df.columns:
            df["Incoming_Quantity"] = df["Incoming_Quantity"].fillna(df["Daily_Usage"] * 5)
            
        # 3. Negative stock validation
        negative_stock_mask = df["Current_Stock"] < 0
        neg_count = negative_stock_mask.sum()
        stats["invalid_records"] += int(neg_count)
        if neg_count > 0:
            stats["corrected_records"] += int(neg_count)
            # Correct negative stock to non-negative (floor to safety stock * 0.2)
            df.loc[negative_stock_mask, "Current_Stock"] = (df.loc[negative_stock_mask, "Safety_Stock"] * 0.2).astype(int)
            
        # 4. Standardize text & date
        df["Component_Name"] = df["Component_Name"].astype(str).str.strip()
        if "Incoming_Date" in df.columns:
            df["Incoming_Date"] = pd.to_datetime(df["Incoming_Date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("2026-09-20")
            
        # 5. Formula recalculation validation: Days_of_Inventory = Current_Stock / Daily_Usage
        df["Daily_Usage"] = df["Daily_Usage"].clip(lower=1)
        df["Days_of_Inventory"] = (df["Current_Stock"] / df["Daily_Usage"]).round(2)
        
        # 6. Safety stock gap identification
        df["Stock_Safety_Gap"] = df["Safety_Stock"] - df["Current_Stock"]
        df["Is_Below_Safety_Stock"] = df["Current_Stock"] < df["Safety_Stock"]
        
        stats["final_usable_records"] = len(df)
        return df, stats

    def clean_production_data(self, df: pd.DataFrame) -> (pd.DataFrame, dict):
        total_records = len(df)
        stats = {
            "dataset": "Production",
            "total_records": total_records,
            "duplicate_records": 0,
            "missing_values": 0,
            "invalid_records": 0,
            "corrected_records": 0,
            "final_usable_records": 0
        }
        
        # 1. Duplicates
        duplicates = df.duplicated().sum()
        stats["duplicate_records"] = int(duplicates)
        df = df.drop_duplicates().copy()
        
        # 2. Missing values
        missing_count = int(df.isnull().sum().sum())
        stats["missing_values"] = missing_count
        
        if "Actual_Production" in df.columns:
            missing_act = df["Actual_Production"].isnull()
            if missing_act.sum() > 0:
                stats["corrected_records"] += int(missing_act.sum())
                # Impute based on planned production and capacity utilization
                df.loc[missing_act, "Actual_Production"] = (df.loc[missing_act, "Planned_Production"] * 0.9).astype(int)
                
        # 3. Numeric bounds & non-negative
        for col in ["Machine_Capacity", "Capacity_Utilization", "Downtime_Hours", "Backlog"]:
            df[col] = df[col].clip(lower=0)
            
        # 4. Dates
        for col in ["Production_Date", "Expected_Completion_Date"]:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
        df["Production_Date"] = df["Production_Date"].fillna("2026-09-10")
        df["Expected_Completion_Date"] = df["Expected_Completion_Date"].fillna("2026-09-15")
        
        stats["final_usable_records"] = len(df)
        return df, stats

    def clean_transportation_data(self, df: pd.DataFrame) -> (pd.DataFrame, dict):
        total_records = len(df)
        stats = {
            "dataset": "Transportation",
            "total_records": total_records,
            "duplicate_records": 0,
            "missing_values": 0,
            "invalid_records": 0,
            "corrected_records": 0,
            "final_usable_records": 0
        }
        
        # 1. Duplicates
        duplicates = df.duplicated().sum()
        stats["duplicate_records"] = int(duplicates)
        df = df.drop_duplicates().copy()
        
        # 2. Missing values
        missing_count = int(df.isnull().sum().sum())
        stats["missing_values"] = missing_count
        
        if "Carrier" in df.columns:
            missing_carr = df["Carrier"].isnull()
            if missing_carr.sum() > 0:
                stats["corrected_records"] += int(missing_carr.sum())
                df["Carrier"] = df["Carrier"].fillna("TCI Supply Chain Solutions")
                
        # 3. Standardize dates
        date_cols = ["Shipment_Date", "Expected_Arrival", "Actual_Arrival"]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            
        # 4. Detect impossible dates (Actual_Arrival < Shipment_Date)
        impossible_mask = df["Actual_Arrival"] < df["Shipment_Date"]
        impossible_count = impossible_mask.sum()
        stats["invalid_records"] += int(impossible_count)
        if impossible_count > 0:
            stats["corrected_records"] += int(impossible_count)
            df.loc[impossible_mask, "Actual_Arrival"] = df.loc[impossible_mask, "Shipment_Date"] + pd.to_timedelta(df.loc[impossible_mask, "Transit_Time"], unit="D")
            
        # Re-format dates
        for col in date_cols:
            df[col] = df[col].dt.strftime("%Y-%m-%d")
            
        # 5. Non-negative transit and delay
        df["Transit_Time"] = df["Transit_Time"].clip(lower=1)
        df["Delay_Days"] = df["Delay_Days"].clip(lower=0)
        
        stats["final_usable_records"] = len(df)
        return df, stats

    def clean_customer_data(self, df: pd.DataFrame) -> (pd.DataFrame, dict):
        total_records = len(df)
        stats = {
            "dataset": "Customer Orders",
            "total_records": total_records,
            "duplicate_records": 0,
            "missing_values": 0,
            "invalid_records": 0,
            "corrected_records": 0,
            "final_usable_records": 0
        }
        
        # 1. Duplicates
        duplicates = df.duplicated().sum()
        stats["duplicate_records"] = int(duplicates)
        df = df.drop_duplicates().copy()
        
        # 2. Missing values & dates
        missing_count = int(df.isnull().sum().sum())
        stats["missing_values"] = missing_count
        
        for col in ["Order_Date", "Required_Delivery_Date", "Planned_Delivery_Date", "Expected_Delivery_Date"]:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            
        # If Required_Delivery_Date is missing, reconstruct: Order_Date + 20 days
        req_missing = df["Required_Delivery_Date"].isnull()
        if req_missing.sum() > 0:
            stats["corrected_records"] += int(req_missing.sum())
            df.loc[req_missing, "Required_Delivery_Date"] = df.loc[req_missing, "Order_Date"] + pd.Timedelta(days=20)
            
        for col in ["Order_Date", "Required_Delivery_Date", "Planned_Delivery_Date", "Expected_Delivery_Date"]:
            df[col] = df[col].fillna(pd.to_datetime("2026-09-25")).dt.strftime("%Y-%m-%d")
            
        # 3. Quantities
        df["Order_Quantity"] = df["Order_Quantity"].clip(lower=1)
        
        stats["final_usable_records"] = len(df)
        return df, stats

    def run_cleaning_pipeline(self):
        """Execute complete data cleaning and validation pipeline."""
        print("Executing Autonomous Data Cleaning & Validation Pipeline...")
        
        # Load raw files
        raw_suppliers = pd.read_csv(os.path.join(self.raw_dir, "suppliers.csv"))
        raw_inventory = pd.read_csv(os.path.join(self.raw_dir, "inventory.csv"))
        raw_production = pd.read_csv(os.path.join(self.raw_dir, "production.csv"))
        raw_transportation = pd.read_csv(os.path.join(self.raw_dir, "transportation.csv"))
        raw_customer = pd.read_csv(os.path.join(self.raw_dir, "customer_orders.csv"))
        
        # Process each dataset
        clean_sup, stats_sup = self.clean_supplier_data(raw_suppliers)
        clean_inv, stats_inv = self.clean_inventory_data(raw_inventory)
        clean_prd, stats_prd = self.clean_production_data(raw_production)
        clean_trp, stats_trp = self.clean_transportation_data(raw_transportation)
        clean_cst, stats_cst = self.clean_customer_data(raw_customer)
        
        # Save cleaned files
        clean_sup.to_csv(os.path.join(self.cleaned_dir, "cleaned_suppliers.csv"), index=False)
        clean_inv.to_csv(os.path.join(self.cleaned_dir, "cleaned_inventory.csv"), index=False)
        clean_prd.to_csv(os.path.join(self.cleaned_dir, "cleaned_production.csv"), index=False)
        clean_trp.to_csv(os.path.join(self.cleaned_dir, "cleaned_transportation.csv"), index=False)
        clean_cst.to_csv(os.path.join(self.cleaned_dir, "cleaned_customer_orders.csv"), index=False)
        
        summary_stats = [stats_sup, stats_inv, stats_prd, stats_trp, stats_cst]
        totals = {
            "dataset": "TOTAL / SYSTEM SUMMARY",
            "total_records": sum(s["total_records"] for s in summary_stats),
            "duplicate_records": sum(s["duplicate_records"] for s in summary_stats),
            "missing_values": sum(s["missing_values"] for s in summary_stats),
            "invalid_records": sum(s["invalid_records"] for s in summary_stats),
            "corrected_records": sum(s["corrected_records"] for s in summary_stats),
            "final_usable_records": sum(s["final_usable_records"] for s in summary_stats),
            "data_health_index": round(
                (sum(s["final_usable_records"] for s in summary_stats) / 
                 max(1, sum(s["total_records"] for s in summary_stats))) * 100, 2
            )
        }
        
        full_report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "system": "Tata Motors Operations Risk AI Data Cleaning Engine",
            "datasets": summary_stats,
            "overall_summary": totals
        }
        
        # Save JSON report
        report_path = os.path.join(os.path.dirname(self.cleaned_dir), "data_quality_report.json")
        with open(report_path, "w") as f:
            json.dump(full_report, f, indent=4)
            
        print("Data Cleaning Pipeline complete.")
        print(f"Data Quality Report saved to: {report_path}")
        print(f"Cleaned datasets written to: {self.cleaned_dir}")
        for s in summary_stats:
            print(f" - {s['dataset']}: {s['total_records']} raw -> {s['duplicate_records']} dups -> {s['corrected_records']} corrected -> {s['final_usable_records']} clean")
        print(f"Total Usable Records: {totals['final_usable_records']} / {totals['total_records']} (Health Index: {totals['data_health_index']}%)")
        
        return full_report, {
            "suppliers": clean_sup,
            "inventory": clean_inv,
            "production": clean_prd,
            "transportation": clean_trp,
            "customer_orders": clean_cst
        }


if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.run_cleaning_pipeline()
