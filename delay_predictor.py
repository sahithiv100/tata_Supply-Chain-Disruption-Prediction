"""
Machine Learning Delay Prediction & Business Financial Impact Engine.
Predicts:
1. Probability of production delay (%)
2. Expected production delay duration (days)
3. Probability of customer delivery delay (%)
4. Expected customer delivery delay (days)
5. Estimated business impact (in INR Lakhs and Crores)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split


VEHICLE_VALUES_LAKHS = {
    "Nexon EV Empowered": 18.5,
    "Harrier Fearless AT": 22.0,
    "Safari Dark Edition": 24.5,
    "Tiago EV Long Range": 10.5,
    "Curvv EV Coupe": 19.5,
    "Prima 5530.S Heavy Truck": 44.0,
    "Signa 2823.K Tipper": 38.0,
    "Ace Gold EV Mini Truck": 7.5,
    "Intra V50 Smart Pickup": 9.0,
    "Starbus EV Ultra 9m": 68.0
}


class DelayPredictor:
    def __init__(self, models_dir="data/processed/models"):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.clf_prod = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        self.reg_prod = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        self.clf_delv = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        self.reg_delv = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
        
        self.feature_cols = [
            "Supplier_Delay_Days", "Supplier_Risk_Factor", "Supplier_Reliability_Score",
            "Days_of_Inventory", "Stockout_Probability", "Capacity_Utilization",
            "Downtime_Hours", "Backlog", "Transit_Delay_Days", "Route_Risk_Index",
            "Deadline_Slack_Days", "Order_Quantity"
        ]

    def _prepare_training_data(self, df: pd.DataFrame):
        """Synthesize ground truth targets for ML model training based on physical plant telemetry."""
        df_train = df.copy()
        
        # Target 1: Production Delay Binary (1 if actual delay occurred, else 0)
        df_train["Target_Prod_Delayed"] = (
            (df_train["Production_Status"] == "Delayed") |
            (df_train["Component_Availability"].str.contains("Unavailable")) |
            (df_train["Downtime_Hours"] > 7.0) |
            ((df_train["Days_of_Inventory"] < 2.0) & (df_train["Supplier_Delay_Days"] > 2))
        ).astype(int)
        
        # Target 2: Production Delay Duration (days)
        df_train["Target_Prod_Delay_Days"] = np.where(
            df_train["Target_Prod_Delayed"] == 1,
            np.clip(
                (df_train["Supplier_Delay_Days"] * 0.6) +
                (df_train["Downtime_Hours"] / 4.0) +
                np.maximum(0, (3.0 - df_train["Days_of_Inventory"]) * 0.8) +
                np.random.normal(0.5, 0.4, len(df_train)),
                1.0, 12.0
            ),
            0.0
        ).round(1)
        
        # Target 3: Delivery Delay Binary
        df_train["Target_Delivery_Delayed"] = (
            (df_train["Target_Prod_Delayed"] == 1) & (df_train["Deadline_Slack_Days"] < 3) |
            (df_train["Transit_Delay_Days"] > 2) |
            (df_train["Deadline_Slack_Days"] < 0)
        ).astype(int)
        
        # Target 4: Customer Delivery Delay Duration (days)
        df_train["Target_Delivery_Delay_Days"] = np.where(
            df_train["Target_Delivery_Delayed"] == 1,
            np.clip(
                (df_train["Target_Prod_Delay_Days"] * 0.7) +
                (df_train["Transit_Delay_Days"] * 0.5) -
                np.maximum(0, df_train["Deadline_Slack_Days"] * 0.4) +
                np.random.normal(0.4, 0.3, len(df_train)),
                1.0, 10.0
            ),
            0.0
        ).round(1)
        
        return df_train

    def train_models(self, df: pd.DataFrame):
        """Train all 4 predictive machine learning models."""
        print("Training Machine Learning Delay Prediction Models...")
        df_train = self._prepare_training_data(df)
        X = df_train[self.feature_cols].fillna(0)
        
        # Train Production Delay Classifier
        self.clf_prod.fit(X, df_train["Target_Prod_Delayed"])
        # Train Production Delay Regressor
        self.reg_prod.fit(X, df_train["Target_Prod_Delay_Days"])
        # Train Delivery Delay Classifier
        self.clf_delv.fit(X, df_train["Target_Delivery_Delayed"])
        # Train Delivery Delay Regressor
        self.reg_delv.fit(X, df_train["Target_Delivery_Delay_Days"])
        
        # Save models
        joblib.dump(self.clf_prod, os.path.join(self.models_dir, "clf_prod.joblib"))
        joblib.dump(self.reg_prod, os.path.join(self.models_dir, "reg_prod.joblib"))
        joblib.dump(self.clf_delv, os.path.join(self.models_dir, "clf_delv.joblib"))
        joblib.dump(self.reg_delv, os.path.join(self.models_dir, "reg_delv.joblib"))
        print("ML Delay Prediction models trained and persisted successfully.")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run ML inference to predict delays and business impact."""
        df_out = df.copy()
        X = df_out[self.feature_cols].fillna(0)
        
        # 1. Production Delay Probability (%)
        prod_probs = self.clf_prod.predict_proba(X)[:, 1]
        df_out["Prod_Delay_Prob_Pct"] = (prod_probs * 100.0).round(1)
        
        # 2. Expected Production Delay Duration (days)
        pred_prod_days = self.reg_prod.predict(X)
        pred_prod_days = np.where(prod_probs > 0.35, np.maximum(1.0, pred_prod_days), 0.0)
        df_out["Expected_Prod_Delay_Days"] = np.round(pred_prod_days, 1)
        
        # 3. Delivery Delay Probability (%)
        delv_probs = self.clf_delv.predict_proba(X)[:, 1]
        df_out["Delivery_Delay_Prob_Pct"] = (delv_probs * 100.0).round(1)
        
        # 4. Expected Customer Delivery Delay (days)
        pred_delv_days = self.reg_delv.predict(X)
        pred_delv_days = np.where(delv_probs > 0.30, np.maximum(1.0, pred_delv_days), 0.0)
        df_out["Expected_Delivery_Delay_Days"] = np.round(pred_delv_days, 1)
        
        # 5. Estimated Financial Business Impact (INR Lakhs)
        # Components:
        # A. Plant line stoppage idle cost: Rs 2.5 Lakhs per delay day
        # B. Delay SLA penalty and capital holding cost: 0.12% per day of vehicle value x order quantity
        # C. Customer tier severity penalty (1.5x for Critical, 1.25x for High)
        financial_impacts = []
        for idx, row in df_out.iterrows():
            prod_delay = row["Expected_Prod_Delay_Days"]
            delv_delay = row["Expected_Delivery_Delay_Days"]
            veh_price = VEHICLE_VALUES_LAKHS.get(row["Product"], 18.0)
            order_qty = row["Order_Quantity"]
            
            prio_mult = 1.6 if row["Customer_Priority"] == "Critical" else (1.25 if row["Customer_Priority"] == "High" else 1.0)
            
            if prod_delay == 0 and delv_delay == 0:
                cost = 0.0
            else:
                line_idle_cost = prod_delay * 2.8 # Lakhs
                capital_cost = (order_qty * veh_price * 0.0015 * delv_delay) # Lakhs
                sla_penalty = (delv_delay * 0.75 * prio_mult) # Lakhs
                cost = round((line_idle_cost + capital_cost + sla_penalty) * prio_mult, 2)
                
            financial_impacts.append(cost)
            
        df_out["Estimated_Business_Impact_Lakhs"] = financial_impacts
        df_out["Estimated_Business_Impact_Formatted"] = [
            f"Rs {c:.2f} L" if c < 100.0 else f"Rs {c/100.0:.2f} Cr" for c in financial_impacts
        ]
        
        return df_out

    def run_prediction_pipeline(self):
        """Execute model training and prediction pipeline."""
        print("Executing Delay Prediction Pipeline...")
        data_path = "data/processed/ai_risk_rca_records.csv"
        df = pd.read_csv(data_path)
        
        self.train_models(df)
        df_pred = self.predict(df)
        
        out_path = "data/processed/ai_risk_delay_predicted_records.csv"
        df_pred.to_csv(out_path, index=False)
        print(f"Delay Prediction complete. Saved to: {out_path}")
        return df_pred


if __name__ == "__main__":
    predictor = DelayPredictor()
    predictor.run_prediction_pipeline()
