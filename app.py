# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib # For loading the model and preprocessor
import os

# ==============================
# 🎧 APP TITLE
# ==============================
st.set_page_config(page_title="Earbud Price Predictor", page_icon="🎧", layout="centered")
st.title("🎧 Smart Earbud Price Predictor (PriceOye Data)")
st.write("Predicts approximate earbud prices and recommends the closest real model based on your chosen features.")

# ==============================
# 📂 LOAD DATA & MODEL (Using st.cache_resource)
# ==============================
file_path = "priceoye_specs.csv"
model_filename = 'earbud_price_predictor_model.pkl'
trained_columns_filename = 'trained_columns.pkl'

@st.cache_resource
def load_data_and_model(data_path, model_path, cols_path):
    # Check if all necessary files exist
    if not os.path.exists(data_path):
        st.error(f"Error: Data file '{data_path}' not found. Please ensure the CSV is in the same directory.")
        st.stop()
    if not os.path.exists(model_path):
        st.error(f"Error: Model file '{model_path}' not found. Please run 'train_and_save_model.py' first.")
        st.stop()
    if not os.path.exists(cols_path):
        st.error(f"Error: Trained columns file '{cols_path}' not found. Please run 'train_and_save_model.py' first.")
        st.stop()

    df = pd.read_csv(data_path)
    model_pipeline = joblib.load(model_path)
    trained_columns = joblib.load(cols_path)

    # Re-apply necessary cleaning and imputation to the dataframe for displaying options
    # This must be consistent with the training script
    if "url" in df.columns:
        df.drop(columns=["url"], inplace=True)
    df.dropna(subset=["current_price"], inplace=True) # Drop rows with missing price

    # Cleaning functions from training script
    def extract_numeric(value):
        if isinstance(value, str):
            digits = ''.join(ch for ch in value if ch.isdigit() or ch == '.')
            try:
                return float(digits) if digits else np.nan
            except ValueError:
                return np.nan
        return value

    if "Bluetooth Version" in df.columns:
        df["Bluetooth Version"] = df["Bluetooth Version"].astype(str).str.extract(r'([\d.]+)').astype(float)
    if "Capacity for buds" in df.columns:
        df["Capacity for buds"] = df["Capacity for buds"].apply(extract_numeric)
    if "Capacity for Case" in df.columns:
        df["Capacity for Case"] = df["Capacity for Case"].apply(extract_numeric)
    df = df.replace(["", "nan", "None"], np.nan)

    # Determine numeric and categorical columns *from the trained_columns list*
    # These must be derived consistent with how the model was trained
    # Use the pipeline's preprocessor to get info if needed, but for inputs,
    # it's usually sufficient to just know their dtypes from the original X
    # For now, we'll quickly re-derive types from the loaded df using trained_columns
    temp_df_for_types = df[trained_columns] # Use a subset with only trained columns
    numeric_cols_from_trained = [c for c in trained_columns if temp_df_for_types[c].dtype != 'object']
    categorical_cols_from_trained = [c for c in trained_columns if temp_df_for_types[c].dtype == 'object']

    # Impute NaNs for user input options (consistent with training)
    for col in numeric_cols_from_trained:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols_from_trained:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")


    return df, model_pipeline, trained_columns, numeric_cols_from_trained, categorical_cols_from_trained

df, model, trained_columns, numeric_cols, categorical_cols = load_data_and_model(file_path, model_filename, trained_columns_filename)

st.success("✅ Model and data loaded successfully!")

# ==============================
# 🧍 USER INPUT SECTION (Using st.sidebar for better layout)
# ==============================
st.subheader("🔧 Enter Your Desired Features")

user_input = {}

# Use st.sidebar for inputs
with st.sidebar:
    st.header("Configure Earbud Features")
    # Dynamic input generation based on trained columns and their determined types
    for col in trained_columns:
        if col in numeric_cols:
            # Ensure min_value and max_value are actual floats, not pd.Series/NaN
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            default_val = float(df[col].median())

            user_input[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=default_val, key=f"num_{col}")
        elif col in categorical_cols:
            options = df[col].dropna().unique().tolist()
            user_input[col] = st.selectbox(f"{col}", options, key=f"cat_{col}")
        # else: Handle cases where a trained column might not be numeric/categorical
        # or where its type might be misidentified. For this app, we assume they are.


# Convert to DataFrame and ensure columns match training data
user_df = pd.DataFrame([user_input])
# Reorder columns to match the training data's column order
user_df = user_df[trained_columns]

# ==============================
# 💰 PREDICTION
# ==============================
if st.button("💸 Predict Price"):
    predicted_price = model.predict(user_df)[0]

    st.subheader("Prediction Results")
    st.write(f"**Predicted Price:** Rs {predicted_price:.0f}")

    st.subheader("🎧 Closest Matching Earbud")
    with st.spinner("Finding closest match..."):
        # For closest match, use the loaded df which has already been cleaned and imputed
        df_for_comparison = df.copy()

        # Calculate a combined similarity score
        # Using inverse of MAE for numeric features and binary match for categorical
        similarity_scores = []
        target_ignore_cols = ["title", "current_price"] # Keep consistent
        X_for_comparison = df_for_comparison.drop(columns=[col for col in target_ignore_cols if col in df_for_comparison.columns], errors='ignore')
        X_for_comparison = X_for_comparison[trained_columns] # Ensure same columns and order as user_df

        for index, row in X_for_comparison.iterrows():
            numeric_diff = 0
            categorical_matches = 0
            for col in trained_columns:
                if col in numeric_cols:
                    numeric_diff += abs(row[col] - user_df[col].iloc[0])
                elif col in categorical_cols:
                    if row[col] == user_df[col].iloc[0]:
                        categorical_matches += 1
            # A simple combined score: lower numeric_diff is better, higher categorical_matches is better
            # Adjust weighting as needed.
            # Avoid division by zero if no numeric_cols or categorical_cols
            score = 0
            if len(numeric_cols) > 0:
                score += (numeric_diff / len(numeric_cols)) * 10 # Emphasize numeric difference
            if len(categorical_cols) > 0:
                score -= (categorical_matches / len(categorical_cols)) * 100 # Reward categorical matches more strongly

            similarity_scores.append(score)

        if similarity_scores:
            closest_idx = np.argmin(similarity_scores) # Find the row with the lowest score (most similar)
            closest_match = df_for_comparison.loc[closest_idx]

            st.write(f"**Closest Match:** {closest_match['title'] if 'title' in closest_match else 'N/A'}")
            st.write(f"**Actual Price:** Rs {closest_match['current_price']:.0f}")

            st.markdown("---")
            st.subheader("Detailed Features of Closest Match")
            # Exclude target/ignore columns from the detailed display for clarity
            display_cols = [col for col in closest_match.index if col not in target_ignore_cols]
            st.json(closest_match[display_cols].to_dict())
        else:
            st.info("No matching earbuds found for comparison.")

st.caption("Data source: PriceOye")