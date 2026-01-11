import pandas as pd
import numpy as np
from typing import Any
from typing_extensions import Annotated
from zenml import step
# from zenml.integrations.mlflow.experiment_trackers import mlflow_experiment_tracker
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

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
        logger.info("   Using default: GradientBoostingClassifier")
        model_class = RandomForestClassifier
        model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': random_state
        }
    else:
        model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': random_state,
            'class_weight': 'balanced'
        }
    
    # Create and train the model
    # tracker = mlflow_experiment_tracker()
    # tracker.log_params(model_params)
    logger.info(f"   Model parameters: {model_params}")
    model = model_class(**model_params)
    
    logger.info("   Training in progress...")

    # Calculate class weights to reduce False Negatives
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    # Create sample weights with higher weight for positive class (reduce FN) - trade-off
    sample_weights = np.array([class_weights[int(y)] * 1.5 if y == 1 else class_weights[int(y)] for y in y_train])
    
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # # Log the full model to MLflow
    # tracker.sklearn.log_model(
    #     model,
    #     artifact_path="model",
    #     registered_model_name="LoanDefaultModel"
    # )

    # # Add mlflow tag
    # tracker.set_tag("model_type","GradientBoosting")
    # tracker.set_tag("version","1.0")
    # tracker.set_tag("author","Mahendiran")
    logger.info("Model training completed!")
    
    return model