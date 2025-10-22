
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# ==============================
# 🎧 APP TITLE
# ==============================
st.set_page_config(page_title="Earbud Price Predictor", page_icon="🎧", layout="centered")
st.title("🎧 Smart Earbud Price Predictor (PriceOye Data)")
st.write("Predicts approximate earbud prices and recommends the closest real model based on your chosen features.")

# ==============================
# 📂 LOAD DATA
# ==============================
file_path = "priceoye_specs.csv"
df = pd.read_csv(file_path)

# Drop unnecessary columns
if "url" in df.columns:
    df.drop(columns=["url"], inplace=True)

# ==============================
# 🧹 CLEANING
# ==============================
def extract_numeric(value):
    """Extract numeric part from strings like '480mAh' or 'v5.3'"""
    if isinstance(value, str):
        digits = ''.join(ch for ch in value if ch.isdigit() or ch == '.')
        return float(digits) if digits else np.nan
    return value

if "Bluetooth Version" in df.columns:
    df["Bluetooth Version"] = df["Bluetooth Version"].astype(str).str.extract(r'([\d.]+)').astype(float)

if "Capacity for buds" in df.columns:
    df["Capacity for buds"] = df["Capacity for buds"].apply(extract_numeric)

if "Capacity for Case" in df.columns:
    df["Capacity for Case"] = df["Capacity for Case"].apply(extract_numeric)

# Drop rows with missing price
df.dropna(subset=["current_price"], inplace=True)

# Replace empty strings and "None"/"nan" with NaN safely
df = df.replace(["", "nan", "None"], np.nan)

# ==============================
# 💾 PREPARE DATA
# ==============================
target = "current_price"
ignore_cols = ["title", target]
X = df.drop(columns=[col for col in ignore_cols if col in df.columns])
y = df[target]

# Split numeric vs categorical columns
categorical_cols = [c for c in X.columns if X[c].dtype == 'object']
numeric_cols = [c for c in X.columns if c not in categorical_cols]

# Fill missing numeric values with median
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values with "Unknown"
for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")

# ==============================
# ⚙️ PIPELINE
# ==============================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', HistGradientBoostingRegressor(random_state=42))
])

# ==============================
# 🧠 TRAINING
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

st.success(f"✅ Model trained successfully with MAE ≈ Rs. {mae:.2f}")

# ==============================
# 🧍 USER INPUT SECTION
# ==============================
st.subheader("🔧 Enter Your Desired Features")

user_input = {}

# Dropdowns for Bluetooth and Capacity
if "Bluetooth Version" in df.columns:
    bluetooth_versions = sorted(df["Bluetooth Version"].dropna().unique().tolist())
    user_input["Bluetooth Version"] = st.selectbox("Bluetooth Version", bluetooth_versions)

if "Capacity for buds" in df.columns:
    buds_caps = sorted(df["Capacity for buds"].dropna().unique().tolist())
    user_input["Capacity for buds"] = st.selectbox("Battery Capacity (Buds)", buds_caps)

if "Capacity for Case" in df.columns:
    case_caps = sorted(df["Capacity for Case"].dropna().unique().tolist())
    user_input["Capacity for Case"] = st.selectbox("Battery Capacity (Case)", case_caps)

# Add all other features dynamically
for col in X.columns:
    if col not in user_input:
        if X[col].dtype == 'object':
            options = df[col].dropna().unique().tolist()
            user_input[col] = st.selectbox(f"{col}", options)
        else:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            default = float(df[col].mean())
            user_input[col] = st.number_input(f"{col}", min_val, max_val, default)

# Convert to DataFrame
user_df = pd.DataFrame([user_input])

# ==============================
# 💰 PREDICTION
# ==============================
if st.button("💸 Predict Price"):
    predicted_price = model.predict(user_df)[0]

    # Find closest match in dataset
    numeric_df = X.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        dist = ((numeric_df - user_df[numeric_df.columns].values) ** 2).sum(axis=1)
        closest_idx = dist.idxmin()
        closest = df.loc[closest_idx]
    else:
        closest = df.iloc[0]

    st.subheader("🎧 Recommended Earbud")
    st.write(f"**Predicted Price:** Rs {predicted_price:.0f}")
    st.write(f"**Closest Match:** {closest['title'] if 'title' in closest else 'N/A'}")
    st.write(f"**Actual Price:** Rs {closest['current_price']:.0f}")
