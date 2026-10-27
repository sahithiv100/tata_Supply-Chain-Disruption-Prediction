"""
Root Cause Analysis Engine for Tata Motors AI Operations Risk System.
Diagnoses operational failures across multi-echelon supply chains, isolates primary root causes,
enumerates contributing telemetry factors, and generates executive business explanations.
"""

import pandas as pd
import numpy as np


class RootCauseAnalyzer:
    def __init__(self):
        pass

    def diagnose_record(self, row: pd.Series) -> dict:
        """Diagnose root cause and build explainable narrative for a single operation record."""
        # Telemetry metrics
        sup_delay = row["Supplier_Delay_Days"]
        lead_time = row["Supplier_Lead_Time"]
        days_inv = row["Days_of_Inventory"]
        safety_gap = row["Safety_Stock_Gap"]
        comp_avail = str(row["Component_Availability"])
        downtime = row["Downtime_Hours"]
        cap_util = row["Capacity_Utilization"]
        transit_delay = row["Transit_Delay_Days"]
        weather = str(row["Weather_Risk"])
        traffic = str(row["Traffic_Risk"])
        slack = row["Deadline_Slack_Days"]
        risk_level = row["Risk_Level"]
        
        contributing_factors = []
        root_cause = ""
        risk_headline = ""
        
        # Build contributing factors list
        contributing_factors.append(f"Current inventory buffer: {days_inv} days (Daily usage: {int(row['Daily_Usage'])} units)")
        contributing_factors.append(f"Supplier replenishment lead time: {int(lead_time)} days ({row['Supplier_Name']})")
        contributing_factors.append(f"Production dependency on component '{row['Component_Name']}': {row['Component_Criticality']}")
        contributing_factors.append(f"Customer delivery deadline buffer: {int(slack)} days for {row['Customer_Name']} ({row['Customer_Priority']} Priority)")
        
        # Decision tree for Primary Root Cause
        if sup_delay >= 3 and days_inv < lead_time and transit_delay >= 2:
            root_cause = (
                f"Compound Multi-Echelon Disruption: Supplier {row['Supplier_Name']} delayed shipment by {int(sup_delay)} days, "
                f"further exacerbated by {int(transit_delay)} days logistics transit delay via {row['Carrier']} on route {row['Route']}."
            )
            risk_headline = f"Severe multi-echelon supply chain blockage starving {row['Production_Line']} of {row['Component_Name']}"
            contributing_factors.append(f"Carrier delay of {int(transit_delay)} days on {row['Route']}")
            
        elif sup_delay >= 3 and days_inv < lead_time:
            root_cause = (
                f"Supplier Replenishment Delay: {row['Supplier_Name']} experienced a {int(sup_delay)}-day delivery slippage "
                f"for {row['Component_Name']}, exceeding plant safety stock threshold."
            )
            risk_headline = f"Supplier delivery slippage causing stockout hazard on {row['Product']}"
            contributing_factors.append(f"Supplier on-time delivery rate is {row['Supplier_Reliability_Score']}%")
            
        elif days_inv < 1.8:
            root_cause = (
                f"Critical Inventory Stockout: On-hand stock of {row['Component_Name']} dropped to {days_inv} days of consumption "
                f"(Safety stock deficit: {int(safety_gap)} units) while replenishment lead time is {int(lead_time)} days."
            )
            risk_headline = f"Imminent stockout of critical assembly {row['Component_Name']} at {row['Plant_ID']}"
            contributing_factors.append(f"Stock safety deficit of {int(safety_gap)} units")
            
        elif downtime >= 8.0:
            root_cause = (
                f"Unscheduled Machine Breakdown: Assembly line {row['Production_Line']} suffered {downtime} hours of unscheduled downtime, "
                f"causing a production backlog of {int(row['Backlog'])} units."
            )
            risk_headline = f"Heavy machine downtime and line bottleneck on {row['Production_Line']}"
            contributing_factors.append(f"Production line capacity utilization at {cap_util}%")
            contributing_factors.append(f"Accumulated line backlog: {int(row['Backlog'])} vehicles")
            
        elif transit_delay >= 3 or "Severe" in weather:
            root_cause = (
                f"Logistics Transportation Disruption: Shipment {row['Shipment_ID']} delayed by {int(transit_delay)} days "
                f"due to {weather} along corridor {row['Route']} operated by {row['Carrier']}."
            )
            risk_headline = f"Inbound transit delay threatening {row['Plant_ID']} assembly schedule"
            contributing_factors.append(f"Route environmental hazard: {weather}")
            
        elif cap_util > 93.0 and row["Backlog"] > 40:
            root_cause = (
                f"Production Line Bottleneck: Extreme capacity utilization ({cap_util}%) with insufficient line buffering "
                f"leading to growing backlog ({int(row['Backlog'])} units) for {row['Product']}."
            )
            risk_headline = f"Capacity saturation on {row['Production_Line']} impacting order fulfilment"
            contributing_factors.append(f"Backlog pressure ratio elevated above normal shift threshold")
            
        elif slack < 0 or (slack <= 2 and row["Customer_Priority"] in ["High", "Critical"]):
            root_cause = (
                f"Customer Delivery SLA Breach Risk: Order {row['Customer_Order_ID']} ({row['Order_Quantity']} units of {row['Product']}) "
                f"has a deadline buffer deficit ({int(slack)} days slack) with production status '{row['Production_Status']}'."
            )
            risk_headline = f"High-priority fleet delivery deadline breach for {row['Customer_Name']}"
            contributing_factors.append(f"Customer priority tier: {row['Customer_Priority']}")
            
        else:
            root_cause = f"Routine operational variability across {row['Production_Line']} and inbound supplier deliveries."
            risk_headline = f"Operational variance within tolerable buffer margins for {row['Product']}"

        # Structured 6-Question Business Explanation for Managers
        why_high = (
            f"The risk score is {row['Risk_Score']}/100 ({risk_level}) because critical path telemetry indicates an impending "
            f"break in operational continuity. The disruption probability is {int(row['Probability_Factor']*100)}%, "
            f"business impact factor is {int(row['Impact_Factor']*100)}%, and urgency factor is {int(row['Urgency_Factor']*100)}%."
        )
        
        what_caused = root_cause
        
        what_will_happen = (
            f"If unmitigated, on-hand inventory of '{row['Component_Name']}' will deplete in {days_inv} days, "
            f"forcing an unscheduled line stoppage on {row['Production_Line']}. This will delay assembly of {row['Product']}, "
            f"resulting in an estimated customer delivery breach of {max(1, int(abs(slack) + 3))} days for {row['Customer_Name']} "
            f"and contractual SLA penalties."
        )
        
        what_to_do = (
            f"Immediately activate the dual-response protocol: 1) Expedite supplier shipment or initiate emergency stock transfer "
            f"from an alternate Tata Motors facility; 2) Reschedule production line sequence to prioritize {row['Customer_Name']} ({row['Customer_Priority']}); "
            f"3) Switch inbound logistics to dedicated express carrier."
        )
        
        why_recommended = (
            f"This combined action directly addresses the primary bottleneck ({root_cause.split(':')[0]}) while securing "
            f"a buffer before the customer deadline expires in {max(0, int(slack))} days."
        )
        
        expected_impact = (
            f"Reduces delay probability from {int(row['Probability_Factor']*100)}% to under 20%, recovers {max(2, int(sup_delay + 1))} days "
            f"of schedule buffer, and protects enterprise customer satisfaction for {row['Customer_Name']}."
        )

        return {
            "Risk_Headline": risk_headline,
            "Primary_Root_Cause": root_cause,
            "Contributing_Factors": "; ".join(contributing_factors),
            "Why_Is_Risk_High": why_high,
            "What_Caused_It": what_caused,
            "What_Will_Happen_If_No_Action": what_will_happen,
            "What_Should_Company_Do": what_to_do,
            "Why_Action_Recommended": why_recommended,
            "Expected_Impact_Of_Action": expected_impact
        }

    def analyze_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute root cause analysis on all records in dataset."""
        print("Executing Root Cause Analysis on operational risk records...")
        results = []
        for _, row in df.iterrows():
            diag = self.diagnose_record(row)
            results.append(diag)
            
        diag_df = pd.DataFrame(results)
        combined_df = pd.concat([df.reset_index(drop=True), diag_df.reset_index(drop=True)], axis=1)
        print(f"Root Cause Analysis complete for {len(combined_df)} records.")
        return combined_df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/ai_risk_scored_records.csv")
    rca = RootCauseAnalyzer()
    enriched = rca.analyze_dataset(df)
    enriched.to_csv("data/processed/ai_risk_rca_records.csv", index=False)
    print("Saved RCA enriched dataset.")
