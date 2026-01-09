import pandas as pd
import numpy as np
from typing import Any
from typing_extensions import Annotated
from zenml import step
import mlflow
import joblib
import mlflow.sklearn
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


@step
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_class: Any = None,
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42
) -> Annotated[Any, "trained_model"]:

    logger.info("Starting model training...")
    logger.info(f"   Training data shape: {X_train.shape}")
    logger.info(f"   Training samples: {len(y_train)}")
    logger.info(f"   Class distribution: {y_train.value_counts().to_dict()}")
    
    # Use default model if not specified
    if model_class is None:
        logger.info("   Using default: RandomForestClassifier")
        model_class = RandomForestClassifier
        model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': random_state,
            'class_weight': 'balanced'
        }
    else:
        model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': random_state,
            'class_weight': 'balanced'
        }
    
    # Create and train the model
    mlflow.log_params(model_params)
    logger.info(f"   Model parameters: {model_params}")
    model = model_class(**model_params)
    
    logger.info("   Training in progress...")

    model.fit(X_train, y_train)
    # Log Model
    mlflow.sklearn.log_model(model, "model")

    # Save Model
    local_path = "model.joblib"
    joblib.dump(model, local_path)

    # Add mlflow tag
    mlflow.set_tag("model_type","RandomForest")
    mlflow.set_tag("version","1.0")
    mlflow.set_tag("author","Mahendiran")
    logger.info("Model training completed!")
    
    return model
