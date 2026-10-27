"""
Data Generation Module for Tata Motors AI Operations Risk Detection System.
Generates realistic synthetic data for 5 operational domains:
1. Supplier Performance
2. Inventory Levels
3. Production Schedules & Capacity
4. Transportation & Logistics
5. Customer Delivery Schedules
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Fix seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

PLANTS = {
    "PLANT_PUN": {"name": "Pune Plant", "state": "Maharashtra", "focus": "EV & Passenger (Nexon EV, Harrier, Safari)"},
    "PLANT_SND": {"name": "Sanand Plant", "state": "Gujarat", "focus": "Passenger & Compact EV (Tiago EV, Tigor, Curvv)"},
    "PLANT_JSR": {"name": "Jamshedpur Plant", "state": "Jharkhand", "focus": "Commercial Vehicles (Prima Heavy Truck, Signa)"},
    "PLANT_PNT": {"name": "Pantnagar Plant", "state": "Uttarakhand", "focus": "Small Commercial (Ace Mini Truck, Intra)"},
    "PLANT_DHW": {"name": "Dharwad Plant", "state": "Karnataka", "focus": "Light Commercial & Buses (Ultra Bus, Starbus)"}
}

COMPONENTS = [
    {"id": "COMP_BRK", "name": "Brake Assembly System", "category": "Mechanical & Safety", "cost": 18500, "criticality": "Critical", "daily_usage_base": 120},
    {"id": "COMP_BAT", "name": "Li-Ion Battery Pack 40kWh", "category": "EV Powertrain", "cost": 285000, "criticality": "Critical", "daily_usage_base": 85},
    {"id": "COMP_ECU", "name": "Electronic Control Unit (ECU)", "category": "Electronics & Semiconductor", "cost": 32000, "criticality": "Critical", "daily_usage_base": 150},
    {"id": "COMP_MOT", "name": "PMSM Traction Motor", "category": "EV Powertrain", "cost": 95000, "criticality": "Critical", "daily_usage_base": 85},
    {"id": "COMP_TRN", "name": "Automated Manual Transmission", "category": "Drivetrain", "cost": 54000, "criticality": "High", "daily_usage_base": 90},
    {"id": "COMP_TRB", "name": "Variable Geometry Turbocharger", "category": "Engine & Air Management", "cost": 26000, "criticality": "High", "daily_usage_base": 70},
    {"id": "COMP_SUS", "name": "Hydraulic Suspension Strut", "category": "Chassis & Ride", "cost": 12500, "criticality": "Medium", "daily_usage_base": 180},
    {"id": "COMP_RAD", "name": "Radiator Cooling Module", "category": "Thermal Management", "cost": 8900, "criticality": "Medium", "daily_usage_base": 160},
    {"id": "COMP_ADS", "name": "ADAS Radar & Front Camera", "category": "Autonomous & Safety", "cost": 41000, "criticality": "High", "daily_usage_base": 65},
    {"id": "COMP_WRH", "name": "High-Voltage Wiring Harness", "category": "Electrical", "cost": 14200, "criticality": "High", "daily_usage_base": 110},
    {"id": "COMP_STG", "name": "Electric Power Steering Rack", "category": "Steering", "cost": 21000, "criticality": "High", "daily_usage_base": 130},
    {"id": "COMP_AXL", "name": "Heavy-Duty Drive Axle", "category": "Commercial Chassis", "cost": 72000, "criticality": "Critical", "daily_usage_base": 45}
]

PRODUCTS = [
    {"name": "Nexon EV Empowered", "category": "Electric SUV", "plant": "PLANT_PUN", "key_comp": ["COMP_BAT", "COMP_MOT", "COMP_ECU", "COMP_BRK", "COMP_ADS"]},
    {"name": "Harrier Fearless AT", "category": "Premium SUV", "plant": "PLANT_PUN", "key_comp": ["COMP_TRN", "COMP_TRB", "COMP_BRK", "COMP_ADS", "COMP_ECU"]},
    {"name": "Safari Dark Edition", "category": "Flagship SUV", "plant": "PLANT_PUN", "key_comp": ["COMP_TRN", "COMP_TRB", "COMP_BRK", "COMP_ADS", "COMP_SUS"]},
    {"name": "Tiago EV Long Range", "category": "Hatchback EV", "plant": "PLANT_SND", "key_comp": ["COMP_BAT", "COMP_MOT", "COMP_ECU", "COMP_BRK", "COMP_WRH"]},
    {"name": "Curvv EV Coupe", "category": "Coupe EV", "plant": "PLANT_SND", "key_comp": ["COMP_BAT", "COMP_MOT", "COMP_ECU", "COMP_ADS", "COMP_STG"]},
    {"name": "Prima 5530.S Heavy Truck", "category": "Heavy Commercial", "plant": "PLANT_JSR", "key_comp": ["COMP_AXL", "COMP_TRB", "COMP_BRK", "COMP_RAD", "COMP_ECU"]},
    {"name": "Signa 2823.K Tipper", "category": "Construction Tipper", "plant": "PLANT_JSR", "key_comp": ["COMP_AXL", "COMP_TRB", "COMP_BRK", "COMP_SUS"]},
    {"name": "Ace Gold EV Mini Truck", "category": "Small Commercial", "plant": "PLANT_PNT", "key_comp": ["COMP_BAT", "COMP_MOT", "COMP_BRK", "COMP_WRH"]},
    {"name": "Intra V50 Smart Pickup", "category": "Pickup", "plant": "PLANT_PNT", "key_comp": ["COMP_TRN", "COMP_SUS", "COMP_BRK", "COMP_RAD"]},
    {"name": "Starbus EV Ultra 9m", "category": "Public Bus", "plant": "PLANT_DHW", "key_comp": ["COMP_BAT", "COMP_MOT", "COMP_AXL", "COMP_BRK", "COMP_ECU"]}
]

SUPPLIER_PROFILES = [
    {"id": "SUP_101", "name": "Bosch India Chassis Systems", "location": "Chakan, Pune", "comp": "COMP_BRK", "archetype": "highly_reliable", "base_lead": 5},
    {"id": "SUP_102", "name": "Tata AutoComp Systems Ltd", "location": "Hinjawadi, Pune", "comp": "COMP_BAT", "archetype": "highly_reliable", "base_lead": 8},
    {"id": "SUP_103", "name": "Exide Energy Solutions", "location": "Bengaluru, Karnataka", "comp": "COMP_BAT", "archetype": "occasionally_delayed", "base_lead": 10},
    {"id": "SUP_104", "name": "Endurance Technologies Ltd", "location": "Waluj, Aurangabad", "comp": "COMP_BRK", "archetype": "frequently_delayed", "base_lead": 6},
    {"id": "SUP_105", "name": "Continental Automotive India", "location": "Manesar, Haryana", "comp": "COMP_ECU", "archetype": "occasionally_delayed", "base_lead": 12},
    {"id": "SUP_106", "name": "Uno Minda Electronic Systems", "location": "Hosur, Tamil Nadu", "comp": "COMP_ECU", "archetype": "high_defect", "base_lead": 9},
    {"id": "SUP_107", "name": "Bharat Forge Powertrain Div", "location": "Mundhwa, Pune", "comp": "COMP_AXL", "archetype": "highly_reliable", "base_lead": 7},
    {"id": "SUP_108", "name": "Motherson Sumi Wiring Systems", "location": "Noida, Uttar Pradesh", "comp": "COMP_WRH", "archetype": "frequently_delayed", "base_lead": 11},
    {"id": "SUP_109", "name": "Lucas TVS Electric Drives", "location": "Padi, Chennai", "comp": "COMP_MOT", "archetype": "occasionally_delayed", "base_lead": 14},
    {"id": "SUP_110", "name": "Valeo India Thermal Systems", "location": "Vallam, Chennai", "comp": "COMP_RAD", "archetype": "highly_reliable", "base_lead": 6},
    {"id": "SUP_111", "name": "Gabriel India Suspension", "location": "Chakan, Pune", "comp": "COMP_SUS", "archetype": "occasionally_delayed", "base_lead": 5},
    {"id": "SUP_112", "name": "Garrett Motion Turbochargers", "location": "Talegaon, Pune", "comp": "COMP_TRB", "archetype": "high_quality", "base_lead": 8},
    {"id": "SUP_113", "name": "ZF Commercial Vehicle Control", "location": "Ambattur, Chennai", "comp": "COMP_TRN", "archetype": "frequently_delayed", "base_lead": 15},
    {"id": "SUP_114", "name": "Denso Kirloskar Electronics", "location": "Dharwad, Karnataka", "comp": "COMP_ADS", "archetype": "occasionally_delayed", "base_lead": 13},
    {"id": "SUP_115", "name": "JTEKT India Steering", "location": "Gurugram, Haryana", "comp": "COMP_STG", "archetype": "highly_reliable", "base_lead": 7},
    {"id": "SUP_116", "name": "Brakes India Private Ltd", "location": "Sholinghur, Tamil Nadu", "comp": "COMP_BRK", "archetype": "high_quality", "base_lead": 7},
    {"id": "SUP_117", "name": "Sona Comstar E-Drivetrain", "location": "Gurugram, Haryana", "comp": "COMP_MOT", "archetype": "frequently_delayed", "base_lead": 12},
    {"id": "SUP_118", "name": "Aptiv Connection Systems", "location": "Kochi, Kerala", "comp": "COMP_WRH", "archetype": "high_defect", "base_lead": 9},
    {"id": "SUP_119", "name": "Subros Thermal Solutions", "location": "Noida, Uttar Pradesh", "comp": "COMP_RAD", "archetype": "occasionally_delayed", "base_lead": 8},
    {"id": "SUP_120", "name": "Tata Steel Tubes & Axles", "location": "Jamshedpur, Jharkhand", "comp": "COMP_AXL", "archetype": "highly_reliable", "base_lead": 4}
]

CARRIERS = [
    {"carrier": "VRL Logistics Fleet", "mode": "Express Road Freight", "reliability": 0.92, "avg_speed": 45},
    {"carrier": "TCI Supply Chain Solutions", "mode": "Dedicated Container Road", "reliability": 0.88, "avg_speed": 40},
    {"carrier": "Blue Dart Automotive Aviation", "mode": "Air Freight Expedited", "reliability": 0.98, "avg_speed": 450},
    {"carrier": "Container Corp of India (CONCOR)", "mode": "Multimodal Rail Cargo", "reliability": 0.84, "avg_speed": 35},
    {"carrier": "Gati KWE Heavy Haulage", "mode": "Heavy Road Carrier", "reliability": 0.80, "avg_speed": 38},
    {"carrier": "Mahindra Logistics Linehaul", "mode": "Road Freight Carrier", "reliability": 0.89, "avg_speed": 42}
]

CUSTOMERS = [
    {"id": "CUST_DL101", "name": "Tata Motors Dealership West Hub", "city": "Mumbai", "priority": "High"},
    {"id": "CUST_DL102", "name": "National Fleet Logistics Corp", "city": "Delhi NCR", "priority": "Critical"},
    {"id": "CUST_DL103", "name": "Bangalore Urban Transport Corp (BMTC)", "city": "Bengaluru", "priority": "Critical"},
    {"id": "CUST_DL104", "name": "Metro Cabs India EV Fleet", "city": "Hyderabad", "priority": "High"},
    {"id": "CUST_DL105", "name": "Eastern Mining & Logistics Ltd", "city": "Kolkata", "priority": "Critical"},
    {"id": "CUST_DL106", "name": "South Highway Freight Lines", "city": "Chennai", "priority": "Normal"},
    {"id": "CUST_DL107", "name": "Gujarat Express Distribution", "city": "Ahmedabad", "priority": "Normal"},
    {"id": "CUST_DL108", "name": "Apex Commercial Fleet Leasing", "city": "Pune", "priority": "High"},
    {"id": "CUST_DL109", "name": "Punjab Agri Transit Hub", "city": "Chandigarh", "priority": "Normal"},
    {"id": "CUST_DL110", "name": "Kerala Green Mobility Project", "city": "Kochi", "priority": "High"}
]


def generate_supplier_data(n_records=900, base_date=datetime(2026, 9, 1)):
    """Generate realistic synthetic Supplier Data with varied behavioral profiles."""
    records = []
    
    for i in range(n_records):
        sup_prof = random.choice(SUPPLIER_PROFILES)
        comp_meta = next(c for c in COMPONENTS if c["id"] == sup_prof["comp"])
        
        # Archetype behavioral distribution
        archetype = sup_prof["archetype"]
        base_lead = sup_prof["base_lead"]
        
        if archetype == "highly_reliable":
            ot_rate = np.random.uniform(94.0, 99.5)
            quality_rate = np.random.uniform(98.5, 99.9)
            defect_rate = round(100.0 - quality_rate, 2)
            hist_delay = max(0, int(np.random.exponential(0.6)))
            lead_time = max(3, int(np.random.normal(base_lead, 0.8)))
            reliability_score = round(np.random.uniform(88, 98), 1)
        elif archetype == "occasionally_delayed":
            ot_rate = np.random.uniform(82.0, 91.0)
            quality_rate = np.random.uniform(95.0, 98.0)
            defect_rate = round(100.0 - quality_rate, 2)
            hist_delay = int(np.random.exponential(2.2))
            lead_time = max(4, int(np.random.normal(base_lead + 1.5, 1.8)))
            reliability_score = round(np.random.uniform(70, 85), 1)
        elif archetype == "frequently_delayed":
            ot_rate = np.random.uniform(55.0, 75.0)
            quality_rate = np.random.uniform(92.0, 96.5)
            defect_rate = round(100.0 - quality_rate, 2)
            hist_delay = int(np.random.exponential(4.8) + 2)
            lead_time = max(6, int(np.random.normal(base_lead + 4.5, 2.5)))
            reliability_score = round(np.random.uniform(40, 68), 1)
        elif archetype == "high_defect":
            ot_rate = np.random.uniform(78.0, 89.0)
            quality_rate = np.random.uniform(88.0, 93.5)
            defect_rate = round(100.0 - quality_rate, 2)
            hist_delay = int(np.random.exponential(2.5))
            lead_time = max(4, int(np.random.normal(base_lead + 1.0, 2.0)))
            reliability_score = round(np.random.uniform(50, 72), 1)
        else: # high_quality
            ot_rate = np.random.uniform(88.0, 96.0)
            quality_rate = np.random.uniform(99.0, 99.9)
            defect_rate = round(100.0 - quality_rate, 2)
            hist_delay = max(0, int(np.random.exponential(1.2)))
            lead_time = max(4, int(np.random.normal(base_lead, 1.0)))
            reliability_score = round(np.random.uniform(84, 95), 1)
            
        # Risk classification
        if reliability_score >= 85:
            risk_level = "LOW"
        elif reliability_score >= 70:
            risk_level = "MEDIUM"
        elif reliability_score >= 55:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
            
        order_qty = int(np.random.choice([150, 250, 400, 600, 1000, 1500]))
        cost_unit = int(comp_meta["cost"] * np.random.uniform(0.95, 1.08))
        
        # Order timeline
        order_offset = random.randint(0, 60)
        promised_date = base_date + timedelta(days=order_offset)
        
        # Inoculate realistic disruptions (e.g. Endurance S104 experiencing 4-6 days delay)
        is_disrupted = (sup_prof["id"] == "SUP_104" and i % 2 == 0) or (random.random() < 0.12)
        if is_disrupted:
            actual_delay = random.randint(3, 8)
            actual_date = promised_date + timedelta(days=actual_delay)
            reliability_score = max(35.0, round(reliability_score - 15.0, 1))
            if reliability_score < 55:
                risk_level = "CRITICAL"
            elif reliability_score < 70:
                risk_level = "HIGH"
        else:
            actual_delay = max(0, int(np.random.choice([-1, 0, 0, 0, 1, 2])))
            actual_date = promised_date + timedelta(days=actual_delay)
            
        records.append({
            "Supplier_ID": sup_prof["id"],
            "Supplier_Name": sup_prof["name"],
            "Component": comp_meta["name"],
            "Component_Category": comp_meta["category"],
            "Location": sup_prof["location"],
            "Lead_Time_Days": lead_time,
            "Promised_Delivery_Date": promised_date.strftime("%Y-%m-%d"),
            "Actual_Delivery_Date": actual_date.strftime("%Y-%m-%d"),
            "On_Time_Delivery_Rate": round(ot_rate, 2),
            "Quality_Rate": round(quality_rate, 2),
            "Defect_Rate": defect_rate,
            "Historical_Delay_Days": hist_delay,
            "Supplier_Reliability_Score": reliability_score,
            "Supplier_Risk_Level": risk_level,
            "Order_Quantity": order_qty,
            "Cost_Per_Unit": cost_unit
        })
        
    df = pd.DataFrame(records)
    
    # Introduce small intentional data noise for cleaning engine validation
    # Duplicate records (~2%)
    dup_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
    duplicates = df.loc[dup_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Missing values in non-primary columns (~1.5%)
    for col in ["Cost_Per_Unit", "On_Time_Delivery_Rate"]:
        mask = np.random.rand(len(df)) < 0.015
        df.loc[mask, col] = np.nan
        
    # Unstandardized text format in supplier names (whitespace, lowercase)
    noise_idx = np.random.choice(df.index, size=8, replace=False)
    for idx in noise_idx:
        df.loc[idx, "Supplier_Name"] = "  " + str(df.loc[idx, "Supplier_Name"]).upper() + "  "
        
    # Abnormal lead time in 2 rows (e.g. 999 or -5)
    df.loc[random.choice(df.index), "Lead_Time_Days"] = 999
    df.loc[random.choice(df.index), "Lead_Time_Days"] = -4
    
    return df


def generate_inventory_data(n_records=850, base_date=datetime(2026, 9, 1)):
    """Generate realistic synthetic Inventory Data across Tata Motors assembly plants."""
    records = []
    inv_counter = 1000
    
    plant_keys = list(PLANTS.keys())
    
    for i in range(n_records):
        inv_counter += 1
        comp = random.choice(COMPONENTS)
        plant_id = random.choice(plant_keys)
        
        daily_usage = int(max(20, np.random.normal(comp["daily_usage_base"], 15)))
        safety_stock = daily_usage * random.randint(3, 6)
        reorder_level = int(safety_stock * 1.5)
        lead_time = random.randint(4, 14)
        
        # Behavioral distribution: Healthy (65%), Low (15%), Below Safety Stock (12%), Imminent Stockout (5%), Excess (3%)
        dice = random.random()
        if dice < 0.05: # Imminent stockout
            days_of_inv = round(np.random.uniform(0.5, 1.8), 2)
            current_stock = int(daily_usage * days_of_inv)
            stockout_risk = "CRITICAL"
        elif dice < 0.17: # Below safety stock
            days_of_inv = round(np.random.uniform(1.8, 3.2), 2)
            current_stock = int(daily_usage * days_of_inv)
            stockout_risk = "HIGH"
        elif dice < 0.32: # Low inventory
            days_of_inv = round(np.random.uniform(3.2, 5.0), 2)
            current_stock = int(daily_usage * days_of_inv)
            stockout_risk = "MEDIUM"
        elif dice < 0.95: # Healthy inventory
            days_of_inv = round(np.random.uniform(5.5, 12.0), 2)
            current_stock = int(daily_usage * days_of_inv)
            stockout_risk = "LOW"
        else: # Excess inventory
            days_of_inv = round(np.random.uniform(15.0, 24.0), 2)
            current_stock = int(daily_usage * days_of_inv)
            stockout_risk = "LOW"
            
        incoming_qty = int(daily_usage * random.randint(4, 10))
        incoming_date = base_date + timedelta(days=random.randint(1, 15))
        
        records.append({
            "Inventory_ID": f"INV_{inv_counter}",
            "Component_ID": comp["id"],
            "Component_Name": comp["name"],
            "Plant_ID": plant_id,
            "Current_Stock": current_stock,
            "Safety_Stock": safety_stock,
            "Reorder_Level": reorder_level,
            "Daily_Usage": daily_usage,
            "Days_of_Inventory": days_of_inv,
            "Incoming_Quantity": incoming_qty,
            "Incoming_Date": incoming_date.strftime("%Y-%m-%d"),
            "Lead_Time": lead_time,
            "Stockout_Risk": stockout_risk,
            "Criticality": comp["criticality"]
        })
        
    df = pd.DataFrame(records)
    
    # Introduce small intentional data noise
    dup_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
    duplicates = df.loc[dup_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Negative quantity noise in 1 record
    df.loc[random.choice(df.index), "Current_Stock"] = -50
    # Missing incoming quantity in 1% records
    mask = np.random.rand(len(df)) < 0.015
    df.loc[mask, "Incoming_Quantity"] = np.nan
    
    return df


def generate_production_data(n_records=850, base_date=datetime(2026, 9, 1)):
    """Generate realistic synthetic Production Schedules & Capacity Data."""
    records = []
    prod_counter = 2000
    
    lines = {
        "PLANT_PUN": ["LINE_PUN_EV01", "LINE_PUN_SUV02", "LINE_PUN_SUV03"],
        "PLANT_SND": ["LINE_SND_COMP01", "LINE_SND_EV02"],
        "PLANT_JSR": ["LINE_JSR_TRK01", "LINE_JSR_TIP02"],
        "PLANT_PNT": ["LINE_PNT_SCV01", "LINE_PNT_PIK02"],
        "PLANT_DHW": ["LINE_DHW_BUS01"]
    }
    
    for i in range(n_records):
        prod_counter += 1
        prod_model = random.choice(PRODUCTS)
        plant_id = prod_model["plant"]
        prod_line = random.choice(lines[plant_id])
        req_comp_id = random.choice(prod_model["key_comp"])
        req_comp_meta = next(c for c in COMPONENTS if c["id"] == req_comp_id)
        
        machine_cap = int(np.random.choice([120, 150, 180, 220, 250]))
        
        # Scenarios: Normal, High cap utilization, machine downtime, component shortage, bottleneck
        scenario_dice = random.random()
        if scenario_dice < 0.12: # Component shortage disruption
            cap_util = round(np.random.uniform(55, 75), 1)
            downtime_hrs = round(np.random.uniform(8.0, 18.0), 1)
            comp_avail = "Unavailable - Supplier Delay"
            prod_status = "Delayed"
            backlog = int(np.random.uniform(40, 90))
        elif scenario_dice < 0.25: # High capacity utilization bottleneck
            cap_util = round(np.random.uniform(93.0, 99.5), 1)
            downtime_hrs = round(np.random.uniform(4.0, 10.0), 1)
            comp_avail = "Constrained"
            prod_status = "At Risk"
            backlog = int(np.random.uniform(30, 65))
        elif scenario_dice < 0.35: # Machine breakdown
            cap_util = round(np.random.uniform(60, 80), 1)
            downtime_hrs = round(np.random.uniform(12.0, 24.0), 1)
            comp_avail = "Available"
            prod_status = "Delayed"
            backlog = int(np.random.uniform(35, 75))
        else: # Normal or optimal
            cap_util = round(np.random.uniform(78.0, 91.0), 1)
            downtime_hrs = round(np.random.uniform(0.5, 3.5), 1)
            comp_avail = "Available"
            prod_status = "On Schedule"
            backlog = int(np.random.uniform(0, 15))
            
        planned_prod = int(machine_cap * (cap_util / 100.0))
        if prod_status == "Delayed":
            actual_prod = max(10, int(planned_prod * np.random.uniform(0.60, 0.82)))
        elif prod_status == "At Risk":
            actual_prod = int(planned_prod * np.random.uniform(0.85, 0.94))
        else:
            actual_prod = int(planned_prod * np.random.uniform(0.96, 1.02))
            
        prod_date = base_date + timedelta(days=random.randint(0, 45))
        completion_offset = random.randint(2, 6)
        if prod_status == "Delayed":
            completion_offset += random.randint(3, 7)
        exp_completion = prod_date + timedelta(days=completion_offset)
        
        records.append({
            "Production_ID": f"PRD_{prod_counter}",
            "Plant_ID": plant_id,
            "Production_Line": prod_line,
            "Product": prod_model["name"],
            "Planned_Production": planned_prod,
            "Actual_Production": actual_prod,
            "Production_Date": prod_date.strftime("%Y-%m-%d"),
            "Machine_Capacity": machine_cap,
            "Capacity_Utilization": cap_util,
            "Downtime_Hours": downtime_hrs,
            "Backlog": backlog,
            "Required_Component": req_comp_meta["name"],
            "Component_Availability": comp_avail,
            "Production_Status": prod_status,
            "Expected_Completion_Date": exp_completion.strftime("%Y-%m-%d")
        })
        
    df = pd.DataFrame(records)
    
    # Introduce intentional dirty data
    dup_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
    duplicates = df.loc[dup_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Missing values
    mask = np.random.rand(len(df)) < 0.015
    df.loc[mask, "Actual_Production"] = np.nan
    
    return df


def generate_transportation_data(n_records=850, base_date=datetime(2026, 9, 1)):
    """Generate realistic synthetic Transportation & Logistics Data."""
    records = []
    shp_counter = 3000
    
    plant_keys = list(PLANTS.keys())
    
    routes = [
        {"route": "Aurangabad-Pune NH-60 Expressway", "dist": 235},
        {"route": "Bengaluru-Pune Golden Quadrilateral", "dist": 840},
        {"route": "Manesar-Sanand Western Corridor", "dist": 890},
        {"route": "Chennai-Pune NH-48 Freightway", "dist": 1150},
        {"route": "Noida-Pantnagar Industrial Highway", "dist": 260},
        {"route": "Jamshedpur-Kolkata Dedicated Corridor", "dist": 285},
        {"route": "Hosur-Dharwad South Expressway", "dist": 440},
        {"route": "Nhava Sheva Port-Pune Express Link", "dist": 130}
    ]
    
    for i in range(n_records):
        shp_counter += 1
        sup = random.choice(SUPPLIER_PROFILES)
        dest_plant = random.choice(plant_keys)
        carrier_info = random.choice(CARRIERS)
        route_info = random.choice(routes)
        
        dist = route_info["dist"]
        transit_days = max(1, int(round(dist / (carrier_info["avg_speed"] * 10))))
        
        shp_date = base_date + timedelta(days=random.randint(0, 50))
        exp_arr = shp_date + timedelta(days=transit_days)
        
        # Disruption scenarios
        dice = random.random()
        if dice < 0.08: # Weather disruption (monsoon flood)
            traffic_risk = "Medium"
            weather_risk = "Severe (Monsoon Flooding / Landslide)"
            delay_days = random.randint(3, 7)
            status = "Delayed in Transit"
            log_risk = "CRITICAL"
        elif dice < 0.18: # Traffic / highway bottleneck
            traffic_risk = "High (Choke Point Congestion)"
            weather_risk = "Low"
            delay_days = random.randint(2, 4)
            status = "Delayed in Transit"
            log_risk = "HIGH"
        elif dice < 0.25: # Vehicle breakdown / carrier glitch
            traffic_risk = "Low"
            weather_risk = "Low"
            delay_days = random.randint(2, 5)
            status = "Vehicle Breakdown / Hold"
            log_risk = "HIGH"
        elif dice < 0.35: # Port / terminal congestion
            traffic_risk = "Medium"
            weather_risk = "Low"
            delay_days = random.randint(1, 3)
            status = "Customs / Terminal Hold"
            log_risk = "MEDIUM"
        else: # Normal smooth transit
            traffic_risk = "Low"
            weather_risk = "Clear"
            delay_days = 0
            status = "Delivered / On Route"
            log_risk = "LOW"
            
        act_arr = exp_arr + timedelta(days=delay_days)
        
        records.append({
            "Shipment_ID": f"SHP_{shp_counter}",
            "Supplier_ID": sup["id"],
            "Destination_Plant": dest_plant,
            "Shipment_Date": shp_date.strftime("%Y-%m-%d"),
            "Expected_Arrival": exp_arr.strftime("%Y-%m-%d"),
            "Actual_Arrival": act_arr.strftime("%Y-%m-%d"),
            "Transport_Mode": carrier_info["mode"],
            "Carrier": carrier_info["carrier"],
            "Route": route_info["route"],
            "Distance": dist,
            "Transit_Time": transit_days,
            "Delay_Days": delay_days,
            "Shipment_Status": status,
            "Traffic_Risk": traffic_risk,
            "Weather_Risk": weather_risk,
            "Logistics_Risk_Level": log_risk
        })
        
    df = pd.DataFrame(records)
    
    # Introduce intentional dirty data
    dup_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
    duplicates = df.loc[dup_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Missing carrier in 1% records
    mask = np.random.rand(len(df)) < 0.015
    df.loc[mask, "Carrier"] = np.nan
    
    # Impossible date in 1 row (Actual Arrival before Shipment Date)
    bad_idx = random.choice(df.index)
    df.loc[bad_idx, "Actual_Arrival"] = (datetime.strptime(df.loc[bad_idx, "Shipment_Date"], "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d")
    
    return df


def generate_customer_delivery_data(n_records=850, base_date=datetime(2026, 9, 1)):
    """Generate realistic synthetic Customer & Delivery Orders Data."""
    records = []
    ord_counter = 4000
    
    for i in range(n_records):
        ord_counter += 1
        cust = random.choice(CUSTOMERS)
        prod = random.choice(PRODUCTS)
        
        order_qty = int(np.random.choice([15, 30, 50, 100, 150, 250]))
        order_date = base_date + timedelta(days=random.randint(0, 35))
        
        lead_time = random.randint(14, 28)
        req_delivery = order_date + timedelta(days=lead_time)
        planned_delivery = req_delivery - timedelta(days=random.randint(1, 4))
        
        # Link cascading operational states
        dice = random.random()
        if dice < 0.12: # Critical delay chain
            prod_status = "Delayed - Component Shortage"
            inv_status = "Imminent Stockout"
            trans_status = "Severe Delay"
            delivery_delay_days = random.randint(4, 9)
            delivery_risk = "CRITICAL"
        elif dice < 0.26: # High delay chain
            prod_status = "Production Constrained"
            inv_status = "Below Safety Stock"
            trans_status = "Delayed in Transit"
            delivery_delay_days = random.randint(2, 5)
            delivery_risk = "HIGH"
        elif dice < 0.40: # Moderate risk
            prod_status = "On Schedule"
            inv_status = "Low Inventory"
            trans_status = "Congested"
            delivery_delay_days = random.randint(1, 2)
            delivery_risk = "MEDIUM"
        else: # Healthy on-time
            prod_status = "Completed / Assembled"
            inv_status = "Optimal"
            trans_status = "On Time"
            delivery_delay_days = 0
            delivery_risk = "LOW"
            
        exp_delivery = planned_delivery + timedelta(days=delivery_delay_days)
        
        records.append({
            "Order_ID": f"ORD_{ord_counter}",
            "Customer_ID": cust["id"],
            "Customer_Name": cust["name"],
            "Product": prod["name"],
            "Order_Quantity": order_qty,
            "Order_Date": order_date.strftime("%Y-%m-%d"),
            "Required_Delivery_Date": req_delivery.strftime("%Y-%m-%d"),
            "Planned_Delivery_Date": planned_delivery.strftime("%Y-%m-%d"),
            "Current_Production_Status": prod_status,
            "Current_Inventory_Status": inv_status,
            "Transportation_Status": trans_status,
            "Expected_Delivery_Date": exp_delivery.strftime("%Y-%m-%d"),
            "Delivery_Risk": delivery_risk,
            "Customer_Priority": cust["priority"],
            "Destination_City": cust["city"]
        })
        
    df = pd.DataFrame(records)
    
    # Introduce small intentional noise
    dup_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
    duplicates = df.loc[dup_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # Missing delivery date in 1% records
    mask = np.random.rand(len(df)) < 0.015
    df.loc[mask, "Required_Delivery_Date"] = np.nan
    
    return df


def generate_all_datasets(output_dir="data/raw"):
    """Generate and save all 5 operational datasets."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating synthetic operational datasets in '{output_dir}'...")
    
    df_suppliers = generate_supplier_data(n_records=900)
    df_inventory = generate_inventory_data(n_records=850)
    df_production = generate_production_data(n_records=850)
    df_transportation = generate_transportation_data(n_records=850)
    df_customer = generate_customer_delivery_data(n_records=850)
    
    files = {
        "suppliers.csv": df_suppliers,
        "inventory.csv": df_inventory,
        "production.csv": df_production,
        "transportation.csv": df_transportation,
        "customer_orders.csv": df_customer
    }
    
    for filename, df in files.items():
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f" -> Saved {filename}: {len(df)} records ({len(df.columns)} columns)")
        
    print("All synthetic datasets generated successfully.")
    return files


if __name__ == "__main__":
    generate_all_datasets()
