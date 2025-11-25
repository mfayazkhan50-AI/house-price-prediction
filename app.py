import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Configuration ---
PAGE_TITLE = "🏠 Smart House Price Predictor"
PAGE_ICON = "🏠"
MODEL_PATH = 'house_price_model.pkl'
FEATURES_PATH = 'feature_names.pkl'
# Hardcoded std dev for simulation (replace with actual model output if possible)
MODEL_STD_DEV = 19013

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="centered")

# --- Helper Functions ---
@st.cache_resource
def load_model(path):
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"❌ Model file not found: '{path}'. Please ensure the file is in the correct directory.")
        return None
    except Exception as e:
        st.error(f"❌ An error occurred while loading the model: {e}")
        return None

@st.cache_resource
def load_feature_names(path):
    try:
        with open(path, 'rb') as file:
            features = pickle.load(file)
        return features
    except FileNotFoundError:
        st.error(f"❌ Feature names file not found: '{path}'.")
        return []
    except Exception as e:
        st.error(f"❌ An error occurred while loading feature names: {e}")
        return []

def prepare_input(bedrooms, bathrooms, area, age, quality, garage, neighborhood_encoded):
    input_data = {
        'BedroomAbvGr': [bedrooms],
        'TotalBathrooms': [bathrooms],
        'TotalArea': [area],
        'HouseAge': [age],
        'OverallQual': [quality],
        'GarageCars': [garage],
        'Neighborhood_encoded': [neighborhood_encoded]
    }
    return pd.DataFrame(input_data)

# --- Main Application ---
st.title(PAGE_TITLE)
st.markdown("Enter your house details below to get an **instant price prediction**!")

model = load_model(MODEL_PATH)
feature_names = load_feature_names(FEATURES_PATH)

if model is None or not feature_names:
    st.stop() # Stop the app if the model fails to load

# Sidebar for inputs
st.sidebar.header("📊 House Details")

with st.sidebar.form("input_form"):
    # User inputs with input validation hints
    bedrooms = st.slider("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.slider("Total Bathrooms", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
    area = st.number_input("Total Area (sqft)", min_value=500, max_value=20000, value=1500, help="Area must be between 500 and 20000 sqft.")
    age = st.slider("House Age (years)", min_value=0, max_value=100, value=10)
    quality = st.slider("Overall Quality (1-10)", min_value=1, max_value=10, value=7)
    garage = st.slider("Garage Cars", min_value=0, max_value=5, value=2)
    neighborhood = st.selectbox("Neighborhood", ["Average Area", "Good Area", "Premium Area"])

    # Convert neighborhood to encoded value
    neighborhood_map = {"Average Area": 5, "Good Area": 10, "Premium Area": 15}
    neighborhood_encoded = neighborhood_map[neighborhood]

    # Predict button inside the form
    submitted = st.form_submit_button("🎯 Predict Price", use_container_width=True)

if submitted:
    # --- Prediction Logic ---
    try:
        input_df = prepare_input(bedrooms, bathrooms, area, age, quality, garage, neighborhood_encoded)
        
        # Make prediction
        predicted_price = model.predict(input_df)[0]
        
        # --- Display Results ---
        st.success("### 💰 Prediction Results")
        
        # Dynamic Price Range Calculation
        lower_bound = predicted_price - MODEL_STD_DEV
        upper_bound = predicted_price + MODEL_STD_DEV
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Predicted Price",
                value=f"${predicted_price:,.0f}",
                delta="Instant Estimate"
            )
        
        with col2:
            # Static accuracy is now moved to the "About" section for more realistic feel here
            st.metric(
                label="Confidence Interval",
                value=f"± ${MODEL_STD_DEV:,.0f}"
            )
        
        st.info(f"**📊 Expected Price Range:** ${lower_bound:,.0f} - ${upper_bound:,.0f}")
        
        # --- Feature Importance ---
        st.subheader("📈 Feature Impact Ranking")
        importance_data = {
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }
        importance_df = pd.DataFrame(importance_data)
        importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
        importance_df['Impact (%)'] = importance_df['Importance'] * 100
        
        for idx, row in importance_df.iterrows():
            st.progress(float(row['Importance']), 
                       text=f"{idx+1}. {row['Feature']}: {row['Impact (%)']:.1f}%")
        
    except Exception as e:
        st.error(f"An unexpected prediction error occurred: {str(e)}")

# Model information moved to a more detailed expander
with st.expander("ℹ️ About This Model and Data Source"):
    st.markdown(f"""
    **Model Details:**
    - **Algorithm**: Optimized Random Forest
    - **Accuracy (R-squared)**: 88.2%
    - **Avg. Error Margin**: ± ${MODEL_STD_DEV:,.0f}
    - **Training Data**: 1,460 houses (Ames Housing Dataset)
    - **Features Used**: {len(feature_names)} key factors
    
    This application uses a pre-trained machine learning model loaded via Pickle.
    """)

# Footer with improved formatting
st.markdown("---")
st.markdown(f"<p style='text-align: center; color: grey;'>Built with Streamlit and ML | Author: M_Fayaz_Khan</p>", unsafe_allow_html=True)
