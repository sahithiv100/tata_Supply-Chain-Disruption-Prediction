"""
Autonomous Operations Alert System for Tata Motors.
Monitors operational risks and automatically dispatches urgent management alerts
for Critical and High operational disruptions.
"""

import os
import json
import pandas as pd
from datetime import datetime


class AlertSystem:
    def __init__(self, output_dir="data/processed"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_alert(self, row: pd.Series, alert_idx: int) -> dict:
        """Format an operational alert for an at-risk operation."""
        alert_id = f"ALT-TM-{datetime.now().strftime('%Y%m%d')}-{alert_idx:04d}"
        
        headline = (
            f"CRITICAL OPERATIONAL RISK: {row['Product']} assembly at {row['Plant_ID']}"
            if row["Risk_Level"] == "CRITICAL"
            else f"HIGH OPERATIONAL ALERT: Disruptive threat to {row['Product']}"
        )
        
        # Format recommended actions cleanly as a numbered list
        actions_list = [a.strip() for a in str(row["All_Recommended_Actions"]).split("\n") if a.strip()]
        
        alert_payload = {
            "alert_id": alert_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "severity": row["Risk_Level"],
            "risk_score": float(row["Risk_Score"]),
            "risk_type": row["Risk_Type"],
            "headline": headline,
            "plant": row["Plant_ID"],
            "production_line": row["Production_Line"],
            "product": row["Product"],
            "component": row["Component_Name"],
            "supplier": f"{row['Supplier_Name']} ({row['Supplier_ID']})",
            "customer": f"{row['Customer_Name']} ({row['Customer_Priority']} Priority)",
            "telemetry": {
                "current_inventory_days": float(row["Days_of_Inventory"]),
                "supplier_delay_days": int(row["Supplier_Delay_Days"]),
                "production_downtime_hours": float(row["Downtime_Hours"]),
                "transit_delay_days": int(row["Transit_Delay_Days"]),
                "customer_delivery_deadline_days": int(row["Deadline_Slack_Days"])
            },
            "predictions": {
                "production_delay_probability": f"{row['Prod_Delay_Prob_Pct']}%",
                "expected_production_delay": f"{row['Expected_Prod_Delay_Days']} days",
                "customer_delay_probability": f"{row['Delivery_Delay_Prob_Pct']}%",
                "expected_customer_delay": f"{row['Expected_Delivery_Delay_Days']} days",
                "estimated_financial_loss": row["Estimated_Business_Impact_Formatted"]
            },
            "root_cause": row["Primary_Root_Cause"],
            "recommended_actions": actions_list,
            "responsible_team": row["Responsible_Team"],
            "priority": int(row["Action_Priority"])
        }
        
        formatted_text = (
            f"==========================================================\n"
            f"  {row['Risk_Level']} OPERATIONAL RISK ALERT: {alert_id}\n"
            f"==========================================================\n"
            f"Plant:              {row['Plant_ID']} | Line: {row['Production_Line']}\n"
            f"Product:            {row['Product']}\n"
            f"Component:          {row['Component_Name']} ({row['Component_Criticality']})\n"
            f"Supplier:           {row['Supplier_Name']} ({row['Supplier_ID']})\n"
            f"Current Inventory:  {row['Days_of_Inventory']} days (Safety Stock Gap: {int(row['Safety_Stock_Gap'])} units)\n"
            f"Supplier Delay:     {int(row['Supplier_Delay_Days'])} days\n"
            f"Customer Delivery:  {int(row['Deadline_Slack_Days'])} days slack remaining ({row['Customer_Name']})\n"
            f"----------------------------------------------------------\n"
            f"Risk Score:         {row['Risk_Score']}/100 [{row['Risk_Level']}]\n"
            f"Delay Probability:  {row['Prod_Delay_Prob_Pct']}% (Production) | {row['Delivery_Delay_Prob_Pct']}% (Customer)\n"
            f"Expected Delay:     {row['Expected_Prod_Delay_Days']} days (Production) | {row['Expected_Delivery_Delay_Days']} days (Delivery)\n"
            f"Financial Exposure: {row['Estimated_Business_Impact_Formatted']}\n"
            f"----------------------------------------------------------\n"
            f"ROOT CAUSE:\n{row['Primary_Root_Cause']}\n"
            f"----------------------------------------------------------\n"
            f"RECOMMENDED MITIGATION ACTIONS (Priority {row['Action_Priority']}):\n"
        )
        for act in actions_list:
            formatted_text += f"  * {act}\n"
        formatted_text += (
            f"----------------------------------------------------------\n"
            f"Responsible Team:   {row['Responsible_Team']}\n"
            f"Action Urgency:     {row['Action_Urgency']}\n"
            f"==========================================================\n"
        )
        
        return alert_payload, formatted_text

    def dispatch_alerts(self, df: pd.DataFrame, max_alerts=35) -> list:
        """Scan operational dataset and generate alerts for critical & high risks."""
        print("Executing Autonomous Alert Generation...")
        # Filter for Critical and High risks, sort by Risk Score descending
        alert_candidates = df[df["Risk_Level"].isin(["CRITICAL", "HIGH"])].sort_values(
            by=["Risk_Score", "Estimated_Business_Impact_Lakhs"], ascending=[False, False]
        )
        
        alerts_json = []
        alerts_text = []
        
        for idx, (_, row) in enumerate(alert_candidates.head(max_alerts).iterrows()):
            payload, text_view = self.generate_alert(row, idx + 1)
            alerts_json.append(payload)
            alerts_text.append(text_view)
            
        # Save alerts to disk
        json_path = os.path.join(self.output_dir, "operational_alerts.json")
        with open(json_path, "w") as f:
            json.dump(alerts_json, f, indent=4)
            
        txt_path = os.path.join(self.output_dir, "emergency_alerts.txt")
        with open(txt_path, "w") as f:
            f.write("\n\n".join(alerts_text))
            
        print(f"Generated and dispatched {len(alerts_json)} operational alerts.")
        print(f" -> Alerts JSON saved to: {json_path}")
        print(f" -> Alerts TXT saved to: {txt_path}")
        return alerts_json


if __name__ == "__main__":
    df = pd.read_csv("data/processed/ai_risk_mitigated_records.csv")
    alert_sys = AlertSystem()
    alert_sys.dispatch_alerts(df)
