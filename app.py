import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, Query, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Union
from PIL import Image
from config.config import Config
from utils.logger import setup_logger


logger = setup_logger("FASTAPI")


def get_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


try:
    dataset_attributes = get_json(Config.DATASET_ATTRIBUTES_PATH)
    train_records = get_json(Config.TRAIN_RECORDS_PATH)
except FileNotFoundError as e:
    logger.error(f"Failed to load essential asset: {e}")
    raise RuntimeError(f"Failed to load essential asset: {e}")


advertising_dataset_info = dataset_attributes.get("advertising", {})
wine_quality_dataset_info = dataset_attributes.get("wine_quality", {})
mnist_digit_dataset_info = dataset_attributes.get("mnist_digit", {})

ADVERTISING_FEATURE_COUNT = advertising_dataset_info.get('feature_count', 'N/A')
ADVERTISING_FEATURE_NAMES = advertising_dataset_info.get('feature_names', 'N/A')
WINE_QUALITY_FEATURE_COUNT = wine_quality_dataset_info.get('feature_count', 'N/A')
WINE_QUALITY_FEATURE_NAMES = wine_quality_dataset_info.get('feature_names', 'N/A')
MNIST_DIGIT_IMAGE_SIZE = mnist_digit_dataset_info.get('image_size', 'N/A')


class FeaturesInput(BaseModel):
    features: Union[List[float], List[List[float]]]


app = FastAPI(
    title=Config.API_TITLE,
    description=Config.API_DESCRIPTION,
    version=Config.API_VERSION
)


if os.environ.get("DOCKER"):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


models: Dict[str, Any] = {}


@app.get("/")
def home():
    return {"message": "API is ready. Visit /docs for documentation."}

@app.on_event("startup")
async def load_assets():
    logger.info("FastAPI application starting up...")
    try:
        models['linear_regression'] = joblib.load(Config.LR_MODEL_PATH) 
        models['random_forest'] = joblib.load(Config.RF_MODEL_PATH)
        models['cnn'] = tf.keras.models.load_model(Config.CNN_MODEL_PATH)
        logger.info("All models loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Failed to load essential asset: {e}")
        raise RuntimeError(f"Failed to load essential asset: {e}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

@app.post("/predict_tabular")
async def predict_tabular(
    model_name: str = Query(..., enum=["linear_regression", "random_forest"], description="Model name.", example="linear_regression"),
    request_body: FeaturesInput = Body(..., description="List of features for the tabular model.")
):
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found or failed to load.")
    if not request_body.features:
        raise HTTPException(status_code=400, detail="Payload not found or empty.")

    feature_values = request_body.features
    if model_name == "linear_regression":
        expected_count = ADVERTISING_FEATURE_COUNT
        feature_names = ADVERTISING_FEATURE_NAMES
    elif model_name == "random_forest":
        expected_count = WINE_QUALITY_FEATURE_COUNT
        feature_names = WINE_QUALITY_FEATURE_NAMES
    else:
        raise HTTPException(status_code=500, detail=f"Invalid model name: {model_name}")

    if all(isinstance(x, (int, float)) for x in feature_values):
        feature_values = [feature_values]

    for i, row in enumerate(feature_values):
        if len(row) != expected_count:
            raise HTTPException(status_code=400, detail=f"Row {i} must have {expected_count} features.")

    try:
        model = models[model_name]
        input_data = pd.DataFrame(feature_values, columns=feature_names)
        predictions = model.predict(input_data)
        return {
            "model": model_name,
            "prediction": predictions.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

@app.post("/predict_image")
async def predict_image(
    model_name: str = Query(..., enum=["cnn"], description="Model name.", example="cnn"),
    image_file: UploadFile = File(..., description="Image file for 'cnn' model.")
):
    if model_name not in models:
        logger.warning(f"Model '{model_name}' not found or failed to load.")
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found or failed to load.")
    if not image_file.content_type.startswith("image/"):
        logger.warning(f"Uploaded file is not an image. Content-type: {image_file.content_type}")
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        image_data = await image_file.read()
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('L')
        image_array = np.array(image).reshape(1, MNIST_DIGIT_IMAGE_SIZE[0], MNIST_DIGIT_IMAGE_SIZE[1], 1)
        image_array = image_array.astype('float32') / 255.0

        model = models[model_name]
        prediction_probs = model.predict(image_array)
        predicted_class = np.argmax(prediction_probs, axis=1)[0]

        logger.info(f"Successful prediction for model: {model_name}. Predicted digit: {predicted_class}")
        return {
            "model": model_name,
            "predicted_digit": int(predicted_class), 
            "probabilities": prediction_probs.tolist()[0]
        }
    except Exception as e:
        logger.error(f"An error occurred during image prediction: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Uvicorn server on {Config.HOST}:{Config.PORT}")
    uvicorn.run("app:app", 
                host=Config.HOST, 
                port=Config.PORT, 
                reload=True)
