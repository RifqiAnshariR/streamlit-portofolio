import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from config.config import Config
from utils.styling import load_css


st.set_page_config(page_title="Project", 
                   page_icon=Config.PAGE_ICON, 
                   layout=Config.LAYOUT, 
                   menu_items=Config.MENU_ITEMS)


load_css()


@st.cache_data(ttl=Config.CACHE_TTL)
def get_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

@st.cache_data(ttl=Config.CACHE_TTL)
def get_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
    
@st.cache_data(ttl=Config.CACHE_TTL)
def get_image(file_path):
    try:
        return Image.open(file_path)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None


try:
    dataset_attributes = get_json(Config.DATASET_ATTRIBUTES_PATH)
    train_records = get_json(Config.TRAIN_RECORDS_PATH)
except FileNotFoundError:
    st.error("File not found.")
    st.stop()
except json.JSONDecodeError:
    st.error(f"Error: Invalid JSON file.")
    st.stop()

try:
    linear_regression_outlier_img = get_image(Config.OUTLIER_PLOT_ADVERTISING)
    linear_regression_distribution_img = get_image(Config.DISTRIBUTION_PLOT_ADVERTISING)
    random_forest_outlier_img = get_image(Config.OUTLIER_PLOT_WINE_QUALITY)
    random_forest_confusion_matrix_img = get_image(Config.CONFUSION_MATRIX_PLOT_WINE_QUALITY)
    cnn_loss_img = get_image(Config.LOSS_PLOT_MNIST_DIGIT)
    cnn_confusion_matrix_img = get_image(Config.CONFUSION_MATRIX_PLOT_MNIST_DIGIT)
except FileNotFoundError:
    st.error("File not found.")
    st.stop()
except AttributeError as e:
    st.error(f"Error: Ensure image paths exist in config.py. Detail: {e}")
    st.stop()


advertising_dataset_info = dataset_attributes.get("advertising", {})
wine_quality_dataset_info = dataset_attributes.get("wine_quality", {})
mnist_digit_dataset_info = dataset_attributes.get("mnist_digit", {})

linear_regression_model_info = train_records.get("linear_regression", {})
linear_regression_train_info = train_records.get("linear_regression", {})
random_forest_model_info = train_records.get("random_forest", {})
random_forest_train_info = train_records.get("random_forest", {})
cnn_model_info = train_records.get("cnn", {})
cnn_train_info = train_records.get("cnn", {})

ADVERTISING_TOP_5_ROWS = advertising_dataset_info.get("top_5_rows", [])
ADVERTISING_DATA_COUNT = advertising_dataset_info.get('data_count', 'N/A')
ADVERTISING_FEATURE_COUNT = advertising_dataset_info.get('feature_count', 'N/A')
ADVERTISING_FEATURE_NAMES = advertising_dataset_info.get('feature_names', 'N/A')
ADVERTISING_DATA_TYPES = advertising_dataset_info.get("data_types", {})
ADVERTISING_DUPLICATE_COUNT = advertising_dataset_info.get("duplicate_count", "N/A")
ADVERTISING_MISSING_VALUES = advertising_dataset_info.get("missing_values", {})
WINE_QUALITY_TOP_5_ROWS = wine_quality_dataset_info.get("top_5_rows", [])
WINE_QUALITY_DATA_COUNT = wine_quality_dataset_info.get('data_count', 'N/A')
WINE_QUALITY_FEATURE_COUNT = wine_quality_dataset_info.get('feature_count', 'N/A')
WINE_QUALITY_FEATURE_NAMES = wine_quality_dataset_info.get('feature_names', 'N/A')
WINE_QUALITY_DATA_TYPES = wine_quality_dataset_info.get("data_types", {})
WINE_QUALITY_DUPLICATE_COUNT = wine_quality_dataset_info.get("duplicate_count", "N/A")
WINE_QUALITY_MISSING_VALUES = wine_quality_dataset_info.get("missing_values", {})
MNIST_DIGIT_DATA_COUNT = mnist_digit_dataset_info.get("data_count", "N/A")
MNIST_DIGIT_IMAGE_SIZE = mnist_digit_dataset_info.get("image_size", "N/A")

LINEAR_REGRESSION_PARAMS = linear_regression_model_info.get("params", {})
LINEAR_REGRESSION_METRICS = linear_regression_train_info.get("metrics", {})
LINEAR_REGRESSION_FEATURE_IMPORTANCE = linear_regression_train_info.get("feature_importance", {})
RANDOM_FOREST_PARAMS = random_forest_model_info.get("params", {})
RANDOM_FOREST_ACCURACY = random_forest_train_info.get("metrics", {}).get("accuracy", "N/A")
RANDOM_FOREST_CLASSIFICATION_REPORT = random_forest_train_info.get("metrics", {}).get("classification_report", "N/A")
RANDOM_FOREST_FEATURE_IMPORTANCE = random_forest_train_info.get("feature_importance", {})
CNN_PARAMS = cnn_model_info.get("params", {})
CNN_ACCURACY = cnn_train_info.get("metrics", {}).get("accuracy", "N/A")
CNN_CLASSIFICATION_REPORT = cnn_train_info.get("metrics", {}).get("classification_report", "N/A")


def display_not_available():
    st.info("Currently not available.")

def display_dataset_overview(
    top_5_rows,
    data_count,
    feature_count,
    data_types,
    duplicate_count,
    missing_values,
    outlier_img=None,
    outlier_caption=None,
    dist_img=None,
    dist_caption=None
):
    st.subheader("Dataset Overview")
    st.write("Top 5 Rows.")
    st.dataframe(top_5_rows, hide_index=True)
    st.write(f"**Data Count:** {data_count}")
    st.write(f"**Feature Count:** {feature_count}")

    st.write("**Data Types:**")
    data_types_html = "".join(f"<li>{col}: {dtype}</li>" for col, dtype in data_types.items())
    st.html(f"<ul class='items'>{data_types_html}</ul>")

    st.write(f"**Duplicate Data Count:** {duplicate_count}")
    
    st.write("**Missing Values Count:**")
    missing_values_html = "".join(f"<li>{col}: {dtype}</li>" for col, dtype in missing_values.items())
    st.html(f"<ul class='items'>{missing_values_html}</ul>")
    
    st.write("**Outlier Plot:**")
    if outlier_img and outlier_caption:
        st.image(outlier_img, caption=outlier_caption)
    else:
        st.info("Does not handle outliers.")
    st.write("**Distribution Plot:**")
    if dist_img and dist_caption:
        st.image(dist_img, caption=dist_caption)
    else:
        st.info("Does not handle distribution normalization.")

def display_regression_summary(params, metrics, feature_importance):
    st.subheader("Model Overview")
    st.write("**Hyperparameter & Parameter Configuration:**")
    params_html = "".join(f"<li>{col}: {dtype}</li>" for col, dtype in params.items())
    st.html(f"<ul class='items'>{params_html}</ul>")

    st.subheader("Train Summary")
    st.write("**Evaluation Metrics:**")
    metrics_html = "".join(f"<li>{col}: {dtype}</li>" for col, dtype in metrics.items())
    st.html(f"<ul class='items'>{metrics_html}</ul>")

    st.write("**Feature Importance (Coefficients):**")
    feature_importance_html = "".join(f"<li>{col}: {dtype}</li>" for col, dtype in feature_importance.items())
    st.html(f"<ul class='items'>{feature_importance_html}</ul>")

def display_classification_summary(params, accuracy, classification_report, confusion_img, confusion_caption, feature_importance):
    st.subheader("Train Summary")
    st.write("**Hyperparameter & Parameter Configuration:**")
    params_html = "".join(f"<li>{col}: {dtype}</li>" for col, dtype in params.items())
    st.html(f"<ul class='items'>{params_html}</ul>")

    st.write(f"**Accuracy:** {accuracy:.2%}")
    st.write("**Classification Report:**")
    st.dataframe(classification_report)
    
    st.write(f"**Confusion Plot:**")
    st.image(confusion_img, caption=confusion_caption)

    st.write("**Feature Importance:**")
    feature_importance_html = "".join(f"<li>{col}: {dtype}</li>" for col, dtype in feature_importance.items())
    st.html(f"<ul class='items'>{feature_importance_html}</ul>")


st.title("Project Dashboard")
st.write("On this page, you can explore my AI projects. You can perform inference for each project on the Inference Dashboard page.")
st.divider()

tab1, tab2, tab3 = st.tabs(["Machine Learning (classic)", "Deep Learning", "Application Domains"], width="stretch")

with tab1:
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.header("Supervised Learning")

        with st.expander("Linear Regression"):
            st.header("Sales Prediction Based on Advertising Using Linear Regression")
            st.subheader("Description")
            st.write("Predicts total sales from TV, radio, and newspaper advertising using a Linear Regression Model on the Advertising.csv dataset.")
            display_dataset_overview(
                ADVERTISING_TOP_5_ROWS, ADVERTISING_DATA_COUNT, ADVERTISING_FEATURE_COUNT,
                ADVERTISING_DATA_TYPES, ADVERTISING_DUPLICATE_COUNT, ADVERTISING_MISSING_VALUES,
                linear_regression_outlier_img, "Outlier plot", linear_regression_distribution_img, "Distribution plot"
            )
            display_regression_summary(LINEAR_REGRESSION_PARAMS, LINEAR_REGRESSION_METRICS, LINEAR_REGRESSION_FEATURE_IMPORTANCE)
        
        with st.expander("Ridge Regression"): 
            display_not_available()
        
        with st.expander("Random Forest Regressor"): 
            display_not_available()

        with st.expander("Decision Tree Classifier"): 
            display_not_available()
        
        with st.expander("Random Forest Classifier"):
            st.header("Red Wine Quality Classification Using Random Forest")
            st.subheader("Description")
            st.write("Classifies red wine qualities from characteristics: Alcohol, sulphates, volatile_acidity, total_sulfur_dioxide, citric_acid, pH, free_sulfur_dioxide, fixed_acidity, chlorides, residual_sugar, and density using a Random Forest Classifier on the winequality-red.csv dataset.")
            display_dataset_overview(
                WINE_QUALITY_TOP_5_ROWS, WINE_QUALITY_DATA_COUNT, WINE_QUALITY_FEATURE_COUNT,
                WINE_QUALITY_DATA_TYPES, WINE_QUALITY_DUPLICATE_COUNT, WINE_QUALITY_MISSING_VALUES,
                random_forest_outlier_img, "Outlier plot", None, "Distribution plot"
            )
            display_classification_summary(
                RANDOM_FOREST_PARAMS, RANDOM_FOREST_ACCURACY, RANDOM_FOREST_CLASSIFICATION_REPORT,
                random_forest_confusion_matrix_img, "Confusion plot", RANDOM_FOREST_FEATURE_IMPORTANCE
            )
        
        with st.expander("Logistic Regression"): 
            display_not_available()
        
        with st.expander("SVM"): 
            display_not_available()

        with st.expander("XGBoost Classifier"): 
            display_not_available()
        
        with st.expander("CatBoost Classifier"): 
            display_not_available()

    with col2:
        st.header("Unsupervised Learning")

        with st.expander("KMeans Clustering"): 
            display_not_available()
        
        with st.expander("DBSCAN Clustering"):
            display_not_available()

with tab2:
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.header("Computer Vision")

        with st.expander("CNN"): 
            st.write("Digit classification (0 to 9) using CNN on MNIST dataset.")
            
            # display_dataset_overview(
            #     WINE_QUALITY_TOP_5_ROWS, WINE_QUALITY_DATA_COUNT, WINE_QUALITY_FEATURE_COUNT,
            #     WINE_QUALITY_DATA_TYPES, WINE_QUALITY_DUPLICATE_COUNT, WINE_QUALITY_MISSING_VALUES,
            #     random_forest_outlier_img, "Outlier plot",
            #     None, "Distribution plot"
            # )

            # display_classification_summary(
            #     RANDOM_FOREST_PARAMS, RANDOM_FOREST_ACCURACY, RANDOM_FOREST_CLASSIFICATION_REPORT,
            #     random_forest_confusion_matrix_img, "Confusion plot",
            #     RANDOM_FOREST_FEATURE_IMPORTANCE
            # )

        with st.expander("YOLO"): 
            display_not_available()
            
        st.header("Speech / Audio")

        with st.expander("Audio Project"): 
            display_not_available()

    with col2:
        st.header("Natural Language Processing")

        with st.expander("Sentiment Analysis (BERT)"): 
            display_not_available()
        
        with st.expander("AI Chatbot (RAG)"):
            display_not_available()

        with st.expander("NER"): 
            display_not_available()

with tab3:
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.header("Recommendation System")

        with st.expander("Collaborative Filtering"): 
            display_not_available()
        
        with st.expander("Content-Based Filtering"): 
            display_not_available()
            
        st.header("Generative AI")
    
        with st.expander("LLM"): 
            display_not_available()

        with st.expander("Diffusion Model"): 
            display_not_available()

    with col2:
        st.header("Time Series Forecasting")
        
        with st.expander("RNN"): 
            display_not_available()
        
        with st.expander("LSTM (single step)"):
            display_not_available()
        
        with st.expander("LSTM (multi step)"): 
            display_not_available()


st.html("<div class='footer'>©2025 Rifqi Anshari Rasyid.</div>")
