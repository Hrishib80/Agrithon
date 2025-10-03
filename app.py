import os
import pickle
import joblib
import pandas as pd
import streamlit as st


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as pickle_error:
        try:
            return joblib.load(model_path)
        except Exception as joblib_error:
            raise RuntimeError(
                "Unable to load model. Tried pickle and joblib. "
                f"Pickle error: {pickle_error}. Joblib error: {joblib_error}"
            )


def build_input_form():
    with st.form("yield-form"):
        col1, col2 = st.columns(2)
        with col1:
            region = st.selectbox("Region", ["North", "East", "South", "West"])
            soil_type = st.selectbox("Soil Type", ["Clay", "Sandy", "Loam", "Silt", "Peaty", "Chalky"])
            crop = st.selectbox("Crop", ["Wheat", "Rice", "Maize", "Barley", "Soybean", "Cotton"])
            weather_condition = st.selectbox("Weather Condition", ["Sunny", "Rainy", "Cloudy"])
        with col2:
            rainfall_mm = st.number_input("Rainfall (mm)", min_value=0.0, max_value=5000.0, value=800.0, step=1.0)
            temperature_c = st.number_input("Temperature (°C)", min_value=-20.0, max_value=60.0, value=25.0, step=0.1)
            days_to_harvest = st.number_input("Days to Harvest", min_value=30, max_value=400, value=120, step=1)
            fertilizer_used = st.selectbox("Fertilizer Used", ["Yes", "No"], index=0)
            irrigation_used = st.selectbox("Irrigation Used", ["Yes", "No"], index=0)

        submitted = st.form_submit_button("Predict Yield")

        data = {
            "Region": region,
            "Soil_Type": soil_type,
            "Crop": crop,
            "Rainfall_mm": rainfall_mm,
            "Temperature_Celsius": temperature_c,
            "Fertilizer_Used": fertilizer_used == "Yes",
            "Irrigation_Used": irrigation_used == "Yes",
            "Weather_Condition": weather_condition,
            "Days_to_Harvest": int(days_to_harvest),
        }

        return submitted, pd.DataFrame([data])


def encode_if_needed(model, df: pd.DataFrame) -> pd.DataFrame:
    try:
        _ = model.predict(df.copy())
        return df
    except Exception:
        encoded = df.copy()
        for col in encoded.columns:
            if encoded[col].dtype == object:
                encoded[col] = encoded[col].astype("category").cat.codes
        encoded["Fertilizer_Used"] = encoded["Fertilizer_Used"].astype(int)
        encoded["Irrigation_Used"] = encoded["Irrigation_Used"].astype(int)
        return encoded


def main():
    st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾", layout="centered")
    st.title("🌾 Crop Yield Predictor")
    st.write("Provide the field conditions and get a predicted yield (tons/ha).")

    default_model_path = os.path.join(os.path.dirname(__file__), "pipeline3_revised.pkl")
    uploaded = st.file_uploader("Upload a model file (.pkl or .joblib)", type=["pkl", "joblib"], accept_multiple_files=False)

    if uploaded is not None:
        temp_path = os.path.join(os.path.dirname(__file__), "_uploaded_model.tmp")
        with open(temp_path, "wb") as tmpf:
            tmpf.write(uploaded.getbuffer())
        model_path = temp_path
    else:
        model_path = default_model_path

    if not os.path.exists(model_path):
        st.error("No model found. Upload a model file or place 'model.pkl' next to app.py.")
        st.stop()

    model = load_model(model_path)

    submitted, input_df = build_input_form()

    if submitted:
        try:
            df_for_model = encode_if_needed(model, input_df)
            pred = model.predict(df_for_model)
            yield_value = float(pred[0])
            st.success(f"Predicted Yield: {yield_value:.3f} tons per hectare")
            with st.expander("View input data"):
                st.dataframe(input_df)
            with st.expander("View model input (after encoding, if applied)"):
                st.dataframe(df_for_model)
        except Exception as e:
            st.exception(e)
            st.error("Prediction failed. If your model does not include preprocessing, please upload a Pipeline or share the expected preprocessing so we can align inputs.")


if __name__ == "__main__":
    main()

