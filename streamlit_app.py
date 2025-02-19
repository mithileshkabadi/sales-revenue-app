import streamlit as st
import joblib

# Load trained model
model = joblib.load("xgboost_sales_model.pkl")

# Load scaler if used during training
scaler = joblib.load("scaler.pkl")  # Ensure this file exists in GitHub

# Load Label Encoder
label_encoder = joblib.load("label_encoder.pkl")  # Ensure this exists

# Streamlit UI
st.title("Sales Revenue Prediction App")
st.write("Enter the values to predict revenue.")

st.sidebar.header("Enter Product Details")

unit_price = st.sidebar.number_input("Unit Price ($)", min_value=1.0, format="%.2f")
quantity = st.sidebar.number_input("Quantity", min_value=1)
shipping_fee = st.sidebar.number_input("Shipping Fee ($)", min_value=0.0, format="%.2f")

category = st.sidebar.selectbox("Product Category", ["Electronics", "Clothing", "Home", "Beauty", "Sports"])
region = st.sidebar.selectbox("Region", ["North", "South", "East", "West"])
shipping_status = st.sidebar.selectbox("Shipping Status", ["Pending", "Shipped", "Delivered"])

# Encode categorical variables before creating input_data
category_encoded = label_encoder.fit_transform([category])[0]
region_encoded = label_encoder.fit_transform([region])[0]
shipping_status_encoded = label_encoder.fit_transform([shipping_status])[0]

# Button for prediction
if st.sidebar.button("Predict Revenue"):
    st.write("✅ Button Clicked!")  # Debugging

    # Prepare input data
    input_data = pd.DataFrame([[unit_price, quantity, shipping_fee, category_encoded, region_encoded, shipping_status_encoded]],
                              columns=["Unit Price", "Quantity", "Shipping Fee", "Category", "Region", "Shipping Status"])

    # Apply feature scaling
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = np.expm1(model.predict(input_data_scaled))  # Reverse log transformation if applied

    # Display result
    st.success(f"💰 Predicted Revenue: **${prediction[0]:,.2f}**")
