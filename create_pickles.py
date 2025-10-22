
import pandas as pd
import numpy as np
import pickle
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler

# Load data
print("Fetching dataset...")
diabetes = fetch_ucirepo(id=296)
df = pd.concat([diabetes.data.features, diabetes.data.targets], axis=1)
print("Dataset fetched.")

print("Cleaning data...")
df = df.replace("?", np.nan)
df = df.drop(columns=[c for c in ["encounter_id","patient_nbr","weight","payer_code",
                                    "medical_specialty","examide","citoglipton"]
                      if c in df.columns])
df["race"].fillna(df["race"].mode()[0], inplace=True)
df["readmitted"] = (df["readmitted"] == "<30").astype(int)
print("Data cleaned.")

# Feature Engineering
print("Engineering features...")
def charlson(row):
    score = 0
    for c in ["diag_1", "diag_2", "diag_3"]:
        v = str(row[c]) if c in row and pd.notna(row[c]) else ""
        if v.startswith("250"): score += 1
        if v.startswith(("410","411","412","413","414")): score += 1
        if v.startswith("428"): score += 1
        if v.startswith(("582","583","585","586")): score += 2
        if v.startswith(("490","491","492","493","494","495","496")): score += 1
    return min(score, 10)

def lace(row):
    los = row.get("time_in_hospital", 0.0)
    L = 7 if los >= 14 else min(int(los), 7)
    A = 3 if row.get("admission_type_id", 3) in [1, 2] else 0
    C = (5 if row["Charlson_Index"] >= 4 else 3 if row["Charlson_Index"] == 3 else
         2 if row["Charlson_Index"] == 2 else 1 if row["Charlson_Index"] == 1 else 0)
    E = 4 if row.get("number_emergency", 0) >= 4 else int(row.get("number_emergency", 0))
    return L + A + C + E

def hospital_score(row):
    s = 0
    if row.get("num_procedures", 0) > 0: s += 1
    if row.get("admission_type_id", 3) in [1, 2]: s += 1
    n_inp = row.get("number_inpatient", 0)
    if n_inp >= 5: s += 5
    elif n_inp >= 2: s += 2
    if row.get("time_in_hospital", 0) >= 5: s += 2
    return s

df["Charlson_Index"] = df.apply(charlson, axis=1)
df["LACE_Index"] = df.apply(lace, axis=1)
df["HOSPITAL_Score"] = df.apply(hospital_score, axis=1)
df["Days_Since_Last_Discharge"] = df["number_inpatient"].apply(
    lambda n: 365 if n == 0 else int(365 / (n + 1))
)
df["Polypharmacy_Count"] = df["num_medications"]
df["Recent_Hosp_Count"] = df["number_inpatient"]
print("Features engineered.")

# Prepare features
print("Preparing features...")
X = df.drop(columns=["readmitted", "time_in_hospital"])

# One-hot encoding
X_enc = pd.get_dummies(X, drop_first=True, sparse=False)
feature_names = X_enc.columns.tolist()
print(f"Number of features: {len(feature_names)}")

# Scale
print("Scaling features...")
scaler = StandardScaler()
scaler.fit(X_enc)
print("Features scaled.")

# Save the scaler and feature_names
print("Saving scaler and feature names...")
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

print("scaler.pkl and feature_names.pkl created successfully.")
