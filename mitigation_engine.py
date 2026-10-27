"""
Mitigation Recommendation Engine & Action Prioritization for Tata Motors AI Operations.
Generates tailored, root-cause-matched operational mitigations and ranks them
using Urgency, Cost, Expected Risk Reduction, Feasibility, and Customer Impact.
"""

import pandas as pd
import numpy as np


BACKUP_SUPPLIERS = {
    "Brake Assembly System": "Brakes India Private Ltd (Sholinghur)",
    "Li-Ion Battery Pack 40kWh": "Tata AutoComp Systems Ltd (Hinjawadi, Pune)",
    "Electronic Control Unit (ECU)": "Continental Automotive India (Manesar)",
    "PMSM Traction Motor": "Lucas TVS Electric Drives (Chennai)",
    "Automated Manual Transmission": "ZF Commercial Vehicle Control (Chennai)",
    "Variable Geometry Turbocharger": "Garrett Motion Turbochargers (Pune)",
    "Hydraulic Suspension Strut": "Gabriel India Suspension (Chakan)",
    "Radiator Cooling Module": "Valeo India Thermal Systems (Chennai)",
    "ADAS Radar & Front Camera": "Denso Kirloskar Electronics (Dharwad)",
    "High-Voltage Wiring Harness": "Motherson Sumi Wiring Systems (Noida)",
    "Electric Power Steering Rack": "JTEKT India Steering (Gurugram)",
    "Heavy-Duty Drive Axle": "Bharat Forge Powertrain Div (Pune)"
}

SISTER_PLANTS = {
    "PLANT_PUN": "Sanand Plant (PLANT_SND)",
    "PLANT_SND": "Pune Plant (PLANT_PUN)",
    "PLANT_JSR": "Pantnagar Plant (PLANT_PNT)",
    "PLANT_PNT": "Jamshedpur Plant (PLANT_JSR)",
    "PLANT_DHW": "Pune Plant (PLANT_PUN)"
}


class MitigationEngine:
    def __init__(self):
        pass

    def recommend_mitigations(self, row: pd.Series) -> dict:
        """Generate specific prioritized mitigation actions based on root cause and operational telemetry."""
        risk_level = row["Risk_Level"]
        risk_type = row["Risk_Type"]
        comp = row["Component_Name"]
        plant_id = row["Plant_ID"]
        sister_plant = SISTER_PLANTS.get(plant_id, "Sanand Plant (PLANT_SND)")
        backup_sup = BACKUP_SUPPLIERS.get(comp, "Tier-1 Pre-Qualified Secondary Sourcing Partner")
        slack = row["Deadline_Slack_Days"]
        cust_name = row["Customer_Name"]
        cust_prio = row["Customer_Priority"]
        
        actions = []
        action_priorities = []
        responsible_teams = []
        urgencies = []
        costs = []
        risk_reductions = []
        feasibilities = []
        
        # 1. MULTI-ECHELON CASCADE OR CRITICAL RISKS
        if risk_level == "CRITICAL" or "Multi-Echelon" in risk_type:
            actions.append(f"Initiate emergency inter-plant stock transfer of {comp} from {sister_plant} via express courier transit.")
            action_priorities.append(1)
            responsible_teams.append("Supply Chain Operations")
            urgencies.append("Immediate")
            costs.append("Medium")
            risk_reductions.append("High (55-70%)")
            feasibilities.append("High")
            
            actions.append(f"Issue emergency split PO to backup supplier {backup_sup} for 40% buffer allocation.")
            action_priorities.append(1)
            responsible_teams.append("Strategic Sourcing")
            urgencies.append("Immediate")
            costs.append("High")
            risk_reductions.append("High (50-65%)")
            feasibilities.append("High")
            
            actions.append(f"Reschedule {row['Production_Line']} shift sequence: prioritize critical batch for {cust_name} ({cust_prio} Priority).")
            action_priorities.append(2)
            responsible_teams.append("Plant Production Planning")
            urgencies.append("High")
            costs.append("Low")
            risk_reductions.append("Medium (35-45%)")
            feasibilities.append("High")
            
            actions.append(f"Upgrade inbound logistics from Road Freight to Blue Dart Aviation air-freight charter.")
            action_priorities.append(2)
            responsible_teams.append("Logistics & Freight Ops")
            urgencies.append("High")
            costs.append("High")
            risk_reductions.append("High (45-60%)")
            feasibilities.append("Medium")

        # 2. SUPPLIER-CENTRIC DISRUPTIONS
        elif "Supplier" in risk_type or row["Supplier_Delay_Days"] >= 3:
            actions.append(f"Deploy Tata Motors Resident QC Engineer to {row['Supplier_Name']} to expedite dispatch and clear line bottlenecks.")
            action_priorities.append(2)
            responsible_teams.append("Supplier Quality & Expediting")
            urgencies.append("High")
            costs.append("Low")
            risk_reductions.append("High (40-55%)")
            feasibilities.append("High")
            
            actions.append(f"Authorize immediate 50% split lot dispatch via hot-shot express courier from {row['Supplier_Name']}.")
            action_priorities.append(2)
            responsible_teams.append("Procurement Logistics")
            urgencies.append("High")
            costs.append("Medium")
            risk_reductions.append("Medium (35-50%)")
            feasibilities.append("High")
            
            actions.append(f"Place standby replenishment release with backup supplier {backup_sup}.")
            action_priorities.append(3)
            responsible_teams.append("Strategic Sourcing")
            urgencies.append("Medium")
            costs.append("Medium")
            risk_reductions.append("Medium (30-45%)")
            feasibilities.append("Medium")

        # 3. INVENTORY STOCKOUT DISRUPTIONS
        elif "Inventory" in risk_type or row["Days_of_Inventory"] < 2.0:
            actions.append(f"Emergency inventory reallocation: Transfer stock of {comp} from {sister_plant} with priority line haul.")
            action_priorities.append(1)
            responsible_teams.append("Inventory Control")
            urgencies.append("Immediate")
            costs.append("Medium")
            risk_reductions.append("High (50-65%)")
            feasibilities.append("High")
            
            actions.append(f"Recalculate dynamic safety stock buffer upward by +35% to prevent future stockout recurrence.")
            action_priorities.append(3)
            responsible_teams.append("Supply Chain Planning")
            urgencies.append("Medium")
            costs.append("Low")
            risk_reductions.append("Medium (30-40%)")
            feasibilities.append("High")
            
            actions.append(f"Temporarily throttle non-critical assembly batches to preserve {comp} for high-priority fleet orders.")
            action_priorities.append(2)
            responsible_teams.append("Production Scheduling")
            urgencies.append("High")
            costs.append("Low")
            risk_reductions.append("High (45-55%)")
            feasibilities.append("High")

        # 4. PRODUCTION BOTTLENECK / MACHINE BREAKDOWN
        elif "Production" in risk_type or row["Downtime_Hours"] >= 6.0:
            actions.append(f"Activate cross-line balancing: Shift overflow units of {row['Product']} to parallel assembly line.")
            action_priorities.append(2)
            responsible_teams.append("Shopfloor Operations")
            urgencies.append("High")
            costs.append("Medium")
            risk_reductions.append("High (45-60%)")
            feasibilities.append("High")
            
            actions.append(f"Deploy emergency maintenance reliability team to clear {row['Downtime_Hours']}h downtime bottleneck on {row['Production_Line']}.")
            action_priorities.append(1)
            responsible_teams.append("Plant Engineering & Maintenance")
            urgencies.append("Immediate")
            costs.append("Low")
            risk_reductions.append("High (50-65%)")
            feasibilities.append("High")
            
            actions.append(f"Schedule weekend 3rd overtime shift to recover backlog of {int(row['Backlog'])} vehicles.")
            action_priorities.append(3)
            responsible_teams.append("Production Operations")
            urgencies.append("Medium")
            costs.append("Medium")
            risk_reductions.append("Medium (35-45%)")
            feasibilities.append("High")

        # 5. LOGISTICS & TRANSPORTATION DISRUPTIONS
        elif "Logistics" in risk_type or row["Transit_Delay_Days"] >= 2:
            actions.append(f"Reroute shipment {row['Shipment_ID']} via high-speed bypass expressway to circumvent {row['Weather_Risk']} on {row['Route']}.")
            action_priorities.append(2)
            responsible_teams.append("Logistics Control Tower")
            urgencies.append("High")
            costs.append("Low")
            risk_reductions.append("High (45-55%)")
            feasibilities.append("High")
            
            actions.append(f"Switch carrier from {row['Carrier']} to VRL Express dedicated linehaul.")
            action_priorities.append(2)
            responsible_teams.append("Transportation Management")
            urgencies.append("High")
            costs.append("Medium")
            risk_reductions.append("Medium (40-50%)")
            feasibilities.append("High")

        # 6. CUSTOMER DELIVERY DEADLINE AT RISK
        elif "Customer" in risk_type or slack <= 2:
            actions.append(f"Tag Order {row['Customer_Order_ID']} for priority dispatch with direct factory-gate express carrier to {row['Destination_City']}.")
            action_priorities.append(1)
            responsible_teams.append("Outbound Dispatch")
            urgencies.append("Immediate")
            costs.append("Medium")
            risk_reductions.append("High (55-70%)")
            feasibilities.append("High")
            
            actions.append(f"Dispatch partial batch delivery (50% units) to {cust_name} to meet interim contractual commitment.")
            action_priorities.append(2)
            responsible_teams.append("Customer Account Management")
            urgencies.append("High")
            costs.append("Low")
            risk_reductions.append("High (40-50%)")
            feasibilities.append("High")

        # 7. ROUTINE OPERATIONAL VARIANCE
        else:
            actions.append(f"Continue standard monitoring protocol on {row['Production_Line']} and maintain baseline safety stock.")
            action_priorities.append(4)
            responsible_teams.append("Operations Planning")
            urgencies.append("Routine")
            costs.append("None")
            risk_reductions.append("Routine (<15%)")
            feasibilities.append("High")
            
            actions.append("Review weekly supplier delivery adherence metrics at scheduled S&OP meeting.")
            action_priorities.append(4)
            responsible_teams.append("Procurement Review")
            urgencies.append("Routine")
            costs.append("None")
            risk_reductions.append("Routine (<10%)")
            feasibilities.append("High")

        primary_action = actions[0]
        primary_priority = action_priorities[0]
        primary_team = responsible_teams[0]
        primary_urgency = urgencies[0]
        primary_cost = costs[0]
        primary_risk_red = risk_reductions[0]
        primary_feasibility = feasibilities[0]

        all_actions_numbered = "\n".join([f"{i+1}. {act}" for i, act in enumerate(actions)])

        return {
            "Recommended_Action": primary_action,
            "All_Recommended_Actions": all_actions_numbered,
            "Action_Priority": primary_priority,
            "Responsible_Team": primary_team,
            "Action_Urgency": primary_urgency,
            "Action_Cost": primary_cost,
            "Expected_Risk_Reduction": primary_risk_red,
            "Action_Feasibility": primary_feasibility
        }

    def process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process entire dataset and attach prioritized mitigations."""
        print("Executing Mitigation Recommendation & Action Prioritization Engine...")
        results = []
        for _, row in df.iterrows():
            rec = self.recommend_mitigations(row)
            results.append(rec)
            
        rec_df = pd.DataFrame(results)
        combined_df = pd.concat([df.reset_index(drop=True), rec_df.reset_index(drop=True)], axis=1)
        print("Mitigation recommendations and action prioritization completed successfully.")
        return combined_df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/ai_risk_delay_predicted_records.csv")
    engine = MitigationEngine()
    enriched = engine.process_dataset(df)
    enriched.to_csv("data/processed/ai_risk_mitigated_records.csv", index=False)
    print("Saved mitigated records dataset.")
