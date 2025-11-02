import json
import cv2
from io import StringIO, BytesIO
import numpy as np
import os
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from config.config import Config
from streamlit_drawable_canvas import st_canvas
from utils.styling import load_css


st.set_page_config(page_title="Inference", 
                    page_icon=Config.PAGE_ICON, 
                    layout=Config.LAYOUT, 
                    menu_items=Config.MENU_ITEMS)


load_css()


@st.cache_data(ttl=Config.CACHE_TTL)
def get_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


try:
    dataset_attributes_info = get_json(Config.DATASET_ATTRIBUTES_PATH)
    model_attributes_info = get_json(Config.MODEL_ATTRIBUTES_PATH)
    train_records_info = get_json(Config.TRAIN_RECORDS_PATH)
except FileNotFoundError:
    st.error("File not found.")
    st.stop()
except json.JSONDecodeError:
    st.error(f"Error: Invalid JSON file.")
    st.stop()


ADVERTISING_FEATURE_NAMES = dataset_attributes_info.get("advertising", {}).get('feature_names', 'N/A')
WINE_QUALITY_FEATURE_NAMES = dataset_attributes_info.get("wine_quality", {}).get('feature_names', 'N/A')
MNIST_DIGIT_IMAGE_SIZE = dataset_attributes_info.get("mnist_digit", {}).get('image_size', 'N/A')

def download_csv(df, file_name="data.csv"):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    st.download_button(
        label="Download Results as CSV",
        data=csv_data,
        file_name=file_name,
        mime="text/csv"
    )

def predict_api(model_name, inputs=None, files=None):
    base_url = "http://fastapi:8000" if os.environ.get("DOCKER") else "http://localhost:8000"
    try:
        if model_name in ["linear_regression", "random_forest"]:
            response = requests.post(
                url=f"{base_url}/predict_tabular",
                params={"model_name": model_name},
                json={"features": inputs} 
            )
            response.raise_for_status()
        elif model_name == "cnn":
            response = requests.post(
                url=f"{base_url}/predict_image",
                params={"model_name": model_name},
                files=files
            )
            response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to FastAPI.")
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error: {e.response.status_code} - {e.response.text}")  
    except Exception as e:
        st.error(f"An error occurred: {e}")
    return None

def handle_tabular_model(feature_names, model_name):
    if not (feature_names or model_name):
        st.error("Invalid data!")
        st.stop()

    st.header(f"Predict with: {model_name}")
    st.info("Enter values for each feature or upload a CSV file.")

    inputs = []
    cols = st.columns(4)
    for i, feature_name in enumerate(feature_names):
        with cols[i % 4]:
            inputs.append(st.number_input(
                label=feature_name,
                value=0.0,
                step=0.5,
                min_value=-300.0,
                max_value=300.0,
                key=f"{model_name}_{feature_name}"
            ))

    uploaded_file = st.file_uploader("Upload CSV file.", type=["csv"])
    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

    manual_filled = any(x != 0 for x in inputs)
    file_uploaded = uploaded_file is not None

    if st.button("Predict"):
        if not (manual_filled or file_uploaded):
            st.warning("Forms or file cannot be empty.")
            st.stop()

        with st.spinner("Loading..."):
            if file_uploaded:
                result = predict_api(model_name=model_name, inputs=df.values.tolist())
            else:
                result = predict_api(model_name=model_name, inputs=inputs)

        if result:
            st.success("Prediction successful!")
            predictions = result.get("prediction", [])

            if file_uploaded:
                st.subheader("Batch Prediction Results")
                df_pred = pd.DataFrame(predictions, columns=["Prediction"])
                df_result = pd.concat([df.reset_index(drop=True), df_pred.reset_index(drop=True)], axis=1)
                
                st.dataframe(df_result, use_container_width=True)
                download_csv(df_result, file_name=f"{model_name}_predictions.csv")
            
            else:
                st.subheader("Prediction Result")
                prediction_val = predictions[0] if predictions else "N/A"
                try:
                    prediction_val = f"{float(prediction_val):,.2f}"
                except (ValueError, TypeError):
                    pass
                
                with st.container(border=True):
                    st.metric(label=f"Result ({model_name})", value=prediction_val)

def handle_image_model(image_size, model_name):
    if not image_size or not model_name:
        st.error("Invalid data!")
        st.stop()

    st.header(f"Predict with: {model_name}")
    st.info("Draw a digit (0 to 9) on the canvas below.")

    col1, col2 = st.columns([1, 1])
    with col1:
        canvas_result = st_canvas(**Config.CANVAS_PARAMS)

    if st.button("Predict"):
        if np.max(canvas_result.image_data[:, :, :3]) == 0:
            st.warning("Canvas cannot be empty.")
            st.stop()
        
        image = canvas_result.image_data
        image_resized = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
        _, image_encoded = cv2.imencode(".png", image_resized)
        image_bytes = BytesIO(image_encoded.tobytes())
        files = {'image_file': ('drawn_digit.png', image_bytes, 'image/png')}

        with st.spinner("Loading..."):
            result = predict_api(model_name=model_name, files=files)
        if result:
            st.success("Prediction successful!")

            with col2:
                with st.container(border=True):
                    predicted_digit = result.get("predicted_digit", "?")
                    st.metric("Predicted Digit:", value=predicted_digit, delta_color="off")
                    
                    probabilities = result.get("probabilities", [])
                    if probabilities:
                        prob_df = pd.DataFrame({
                            "Digit": list(range(len(probabilities))),
                            "Probability": probabilities
                        })
                        fig = px.bar(prob_df, x="Digit", y="Probability", title="Prediction Probabilities")
                        st.plotly_chart(fig, use_container_width=True)


st.title("Inference Dashboard")
st.write("On this page, you can perform inference with various available models. The models listed are from the Project Dashboard page.")
st.divider()

with st.sidebar:
    st.subheader("Available model(s):")
    selected_model = st.radio("Model:", options=model_attributes_info.keys(), label_visibility="collapsed")

if selected_model == 'linear_regression':
    handle_tabular_model(ADVERTISING_FEATURE_NAMES, selected_model)
elif selected_model == 'random_forest':
    handle_tabular_model(WINE_QUALITY_FEATURE_NAMES, selected_model)
elif selected_model == 'cnn':
    handle_image_model(MNIST_DIGIT_IMAGE_SIZE, selected_model)


st.html("<div class='footer'>©2025 Rifqi Anshari Rasyid.</div>")
