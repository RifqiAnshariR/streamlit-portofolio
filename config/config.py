import os
from pathlib import Path

class Config:
    # DIRECTORY PATHS
    # Folder paths
    BASE_DIR = Path(__file__).parent.parent
    ARTIFACTS_DIR = BASE_DIR / "artifacts"
    CONFIG_DIR = BASE_DIR / "config"
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"
    MODELS_DIR = BASE_DIR / "models"
    STATIC_DIR = BASE_DIR / "static"
    UTILS_DIR = BASE_DIR / "utils"
    # Image paths
    FAVICON_PATH = STATIC_DIR / "img/favicon.ico"
    PROFILE_PHOTO_PATH = STATIC_DIR / "img/Foto_Rifqi Anshari Rasyid.jpg"
    # Data paths
    CV_PATH = DATA_DIR / "profile/CV_Rifqi Anshari Rasyid.pdf"
    PERSONAL_INFO_PATH = CONFIG_DIR / "personal_info.json"
    DATASET_ATTRIBUTES_PATH = CONFIG_DIR / "dataset_attributes.json"
    MODEL_ATTRIBUTES_PATH = CONFIG_DIR / "model_attributes.json"
    TRAIN_RECORDS_PATH = CONFIG_DIR / "train_records.json"
    BUTTERFLY_DATASET_PATH = DATA_DIR / "datasets/butterfly"
    ADVERTISING_DATASET_PATH = DATA_DIR / "datasets/advertising.csv"
    CALIFORNIA_HOUSING_DATASET_PATH = DATA_DIR / "datasets/california_housing.csv"
    IRIS_DATASET_PATH = DATA_DIR / "datasets/iris.csv"
    MNIST_DATASET_PATH = DATA_DIR / "datasets/mnist.npz"
    WINE_QUALITY_DATASET_PATH = DATA_DIR / "datasets/winequality-red.csv"
    # Artifact paths
    OUTLIER_PLOT_ADVERTISING = ARTIFACTS_DIR / "dataset_advertising/outlier.png"
    DISTRIBUTION_PLOT_ADVERTISING = ARTIFACTS_DIR / "dataset_advertising/distribution.png"
    OUTLIER_PLOT_WINEQUALITY = ARTIFACTS_DIR / "dataset_winequality/outlier.png"
    CONFUSION_MATRIX_PLOT_WINEQUALITY = ARTIFACTS_DIR / "dataset_winequality/confusion_matrix.png"
    LOSS_PLOT_MNIST = ARTIFACTS_DIR / "dataset_mnist/loss.png"
    CONFUSION_MATRIX_PLOT_MNIST = ARTIFACTS_DIR / "dataset_mnist/confusion_matrix.png"
    # Model paths
    LR_MODEL_PATH = MODELS_DIR / "lr_model.pkl"
    RF_MODEL_PATH = MODELS_DIR / "rf_model.pkl"
    CNN_MODEL_PATH = MODELS_DIR / "cnn_model.keras"

    # CONFIGURATIONS
    # Logging configuration
    LOG_FILE = LOGS_DIR / "app.log"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_LEVEL = "INFO"
    # FastAPI settings
    API_TITLE = "Streamlit Portofolio API"
    API_DESCRIPTION = "API for Streamlit portofolio"
    API_VERSION = "1.0.0"
    HOST = "0.0.0.0"
    PORT = 8000
    # Streamlit settings
    STREAMLIT_PORT = 8501
    PAGE_TITLE = "Streamlit Portofolio"
    PAGE_ICON = FAVICON_PATH
    LAYOUT = "wide"
    MENU_ITEMS = {
        'Get Help': 'https://docs.streamlit.io/',
        'Report a bug': 'https://github.com/streamlit/streamlit/issues',
        'About': 'Aplikasi ini dibuat dengan Streamlit'
    }
    # Cache settings
    CACHE_TTL = 3600
    # Canvas params
    CANVAS_PARAMS = {
        "fill_color": "#000000",
        "stroke_width": 30,
        "stroke_color": "#FFFFFF",
        "background_color": "#000000",
        "height": 300,
        "width": 300,
        "drawing_mode": "freedraw",
        "key": "canvas_cnn",
    }






    # DATA_PATH = ARTIFACTS_DIR / "boston.csv"
    # MODEL_PATH = ARTIFACTS_DIR / "best_model.pkl"
    # SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
    # METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
    # FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.json"
    
    # # Model parameters
    # RANDOM_STATE = 42
    # TEST_SIZE = 0.3
    # NUM_FEATURES = 8
    # TARGET_COLUMN = "MEDV"
    
    # # Feature columns for model
    # FEATURE_COLUMNS = [
    #     "LSTAT", "RM", "CRIM", "PTRATIO",
    #     "INDUS", "TAX", "NOX", "B"
    # ]
    
    # # Model hyperparameters
    # PARAMS = {
    #     'regressor__max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    #     'regressor__learning_rate': [0.001, 0.01, 0.1],
    #     'regressor__n_estimators': [100, 200, 300],
    #     'regressor__min_child_weight': [1, 3, 5],
    #     'regressor__gamma': [0, 0.1, 0.2],
    #     'regressor__subsample': [0.8, 0.9, 1.0]
    # }
    
    # # Cross validation settings
    # CV_FOLDS = 5
    
    # # Logging configuration
    # LOG_FILE = LOGS_DIR / "app.log"
    # LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # LOG_LEVEL = "INFO"
    
    # # FastAPI settings
    # API_TITLE = "House Price Prediction API"
    # API_DESCRIPTION = "API for predicting house prices using the Boston Housing Dataset"
    # API_VERSION = "1.0.0"
    # HOST = "0.0.0.0"
    # PORT = 8000
    
    # # Streamlit settings
    # STREAMLIT_PORT = 8501
    # PAGE_TITLE = "House Price Prediction"
    # PAGE_ICON = "🏠"
    # LAYOUT = "wide"
    
    # # Cache settings
    # CACHE_TTL = 3600  # 1 hour
    
    # # Feature descriptions for documentation
    # FEATURE_DESCRIPTIONS = {
    #     "CRIM": "Per capita crime rate by town",
    #     "ZN": "Proportion of residential land zoned for large lots",
    #     "INDUS": "Proportion of non-retail business acres",
    #     "CHAS": "Charles River dummy variable",
    #     "NOX": "Nitric oxides concentration",
    #     "RM": "Average number of rooms per dwelling",
    #     "AGE": "Proportion of owner-occupied units built prior to 1940",
    #     "DIS": "Weighted distances to employment centres",
    #     "RAD": "Index of accessibility to radial highways",
    #     "TAX": "Full-value property-tax rate",
    #     "PTRATIO": "Pupil-teacher ratio by town",
    #     "B": "Black population ratio",
    #     "LSTAT": "% lower status of the population",
    #     "MEDV": "Median value of owner-occupied homes"
    # }
    
    # # Model performance thresholds
    # METRIC_THRESHOLDS = {
    #     'r2_score': 0.8,
    #     'mae': 3.0,
    #     'rmse': 4.0
    # }
    
    # # Data validation rules
    # DATA_VALIDATION = {
    #     'CRIM': {'min': 0, 'max': 100},
    #     'RM': {'min': 3, 'max': 9},
    #     'LSTAT': {'min': 0, 'max': 40},
    #     'PTRATIO': {'min': 12, 'max': 22},
    #     'INDUS': {'min': 0, 'max': 30},
    #     'TAX': {'min': 150, 'max': 800},
    #     'NOX': {'min': 0.3, 'max': 0.9},
    #     'B': {'min': 0, 'max': 400}
    # }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [cls.ARTIFACTS_DIR, cls.LOGS_DIR, cls.STATIC_DIR]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_feature_range(cls, feature):
        """Get the valid range for a feature."""
        return cls.DATA_VALIDATION.get(feature, {'min': float('-inf'), 'max': float('inf')})
    
    @classmethod
    def is_valid_feature_value(cls, feature, value):
        """Check if a feature value is within valid range."""
        ranges = cls.get_feature_range(feature)
        return ranges['min'] <= value <= ranges['max']

Config.create_directories()