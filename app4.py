import streamlit as st
import pandas as pd
import joblib

# Load saved files
model = joblib.load('house_rent_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

st.set_page_config(page_title="House Rent Prediction", page_icon="🏠")

st.title("🏠 House Rent Prediction App")
st.write("Predict house rent using Machine Learning")

st.divider()

# -------- USER INPUT UI --------
area = st.number_input("Area (sq.ft)", min_value=200, max_value=5000, step=50)
beds = st.number_input("Bedrooms", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, step=1)

house_type = st.selectbox("House Type", ["1BHK", "2BHK", "3BHK"])
city = st.selectbox("City", ["Banglore", "Mumbai", "Pune","Nagpur","New Delhi"])
furnishing = st.selectbox(
    "Furnishing",
    ["Unfurnished", "Semi-Furnished", "Furnished"]
)

# -------- PREDICT --------
if st.button("Predict Rent 💰"):

    # Create input row
    input_df = pd.DataFrame(0, columns=model_columns, index=[0])

    # Scale numeric values
    scaled_vals = scaler.transform(
        pd.DataFrame({
            'area': [area],
            'beds': [beds],
            'bathrooms': [bathrooms]
        })
    )

    input_df[['area', 'beds', 'bathrooms']] = scaled_vals

    # Set categorical values
    if f'house_type_{house_type}' in input_df.columns:
        input_df[f'house_type_{house_type}'] = 1

    if f'city_{city}' in input_df.columns:
        input_df[f'city_{city}'] = 1

    if f'furnishing_{furnishing}' in input_df.columns:
        input_df[f'furnishing_{furnishing}'] = 1

    # Set default locality (first one)
    locality_cols = [c for c in input_df.columns if c.startswith('locality_')]
    if locality_cols:
        input_df[locality_cols[0]] = 1

    prediction = model.predict(input_df)[0]
    prediction = max(0, int(prediction))

    st.success(f"🏠 Estimated Rent: ₹ {prediction}")
