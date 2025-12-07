import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# LOAD SAVED ARTIFACTS
mlp = load_model("best_deep_learning_model.h5")
scaler = joblib.load("scaler.pkl")
feature_names = [
 'tenure',
 'Contract_Two year',
 'TenureServicesScore',
 'InternetType',
 'Fiber_Monthly',
 'InternetService_Fiber optic',
 'IsNewCustomer',
 'IsLongCustomer',
 'MonthlyCharges',
 'PaymentMethod_Electronic check',
 'InternetService_No',
 'Contract_One year',
 'Partner',
 'OnlineSecurity',
 'PaperlessBilling',
 'OnlineBackup',
 'TotalServicesCount',
 'AutomaticPayment',
 'TechSupport',
 'Dependents',
 'PaymentMethod_Credit card (automatic)',
 'StreamingTV',
 'SeniorCitizen',
 'DeviceProtection',
 'MultipleLines_Yes',
 'StreamingMovies'
]

print("\n===== Churn Prediction Demo (MLP) =====\n")

# ========== Helper to convert YES/NO ==========
def ask_yes_no(text):
    x = input(f"{text} (yes/no): ").strip().lower()
    return 1 if x in ["yes", "y", "1"] else 0


# ========== Collect USER INPUT ==========
user = {}

# NUMERIC FEATURES:
user["tenure"] = float(input("Tenure (months): "))
user["TenureServicesScore"] = float(input("Tenure Services Score: "))
user["MonthlyCharges"] = float(input("Monthly Charges: "))
user["TotalServicesCount"] = float(input("Total Services Count: "))

# INTERNET TYPE (numeric encoded)
print("\nInternet Type:")
print("0) No Internet")
print("1) DSL")
print("2) Fiber Optic")
user["InternetType"] = int(input("Choose option (0/1/2): "))

# CONTRACT (One-hot)
print("\nContract Type:")
print("1) Month-to-month")
print("2) One year")
print("3) Two year")
c = int(input("Choose option (1/2/3): "))

user["Contract_One year"] = 1 if c == 2 else 0
user["Contract_Two year"] = 1 if c == 3 else 0

# INTERNET SERVICE (One-hot)
print("\nInternet Service:")
print("1) DSL")
print("2) Fiber optic")
print("3) No Internet")
s = int(input("Choose option (1/2/3): "))

user["InternetService_Fiber optic"] = 1 if s == 2 else 0
user["InternetService_No"] = 1 if s == 3 else 0

# PAYMENT METHOD (One-hot)
print("\nPayment Method:")
print("1) Bank Transfer")
print("2) Electronic Check")
print("3) Credit Card (automatic)")
p = int(input("Choose option (1/2/3): "))

user["PaymentMethod_Electronic check"] = 1 if p == 2 else 0
user["PaymentMethod_Credit card (automatic)"] = 1 if p == 3 else 0

# YES/NO FEATURES
user["IsNewCustomer"] = ask_yes_no("Is New Customer")
user["IsLongCustomer"] = ask_yes_no("Is Long Customer")
user["Fiber_Monthly"] = ask_yes_no("Fiber Monthly")
user["Partner"] = ask_yes_no("Partner")
user["OnlineSecurity"] = ask_yes_no("Online Security")
user["PaperlessBilling"] = ask_yes_no("Paperless Billing")
user["OnlineBackup"] = ask_yes_no("Online Backup")
user["AutomaticPayment"] = ask_yes_no("Automatic Payment")
user["TechSupport"] = ask_yes_no("Tech Support")
user["Dependents"] = ask_yes_no("Dependents")
user["StreamingTV"] = ask_yes_no("Streaming TV")
user["SeniorCitizen"] = ask_yes_no("Senior Citizen")
user["DeviceProtection"] = ask_yes_no("Device Protection")
user["MultipleLines_Yes"] = ask_yes_no("Multiple Lines")
user["StreamingMovies"] = ask_yes_no("Streaming Movies")

# ========== BUILD FINAL DATAFRAME ==========
row = pd.DataFrame([[user[col] for col in feature_names]], columns=feature_names)

# SCALE
row_scaled = scaler.transform(row)

# PREDICT
proba = mlp.predict(row_scaled)[0][0]
pred = "Churn" if proba >= 0.5 else "Not Churn"

print("\n===============================")
print("Predicted Probability:", round(float(proba), 4))
print("Prediction:", pred)
print("===============================\n")
