from zenml import pipeline
from src.steps.data_loader import ingest_data
from src.steps.data_validator import validate_data
from src.steps.data_preprocessor import data_preprocess
from src.steps.imbalance_handler import handle_imbalance
from src.steps.model_trainer import train_model
from src.steps.model_evaluator import evaluate_model
import logging
from pathlib import Path
from typing import Dict
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


@pipeline
def training_pipeline(strategy: str = "smote"):
  
    logger.info(f"\n{'='*70}")
    logger.info(f"Starting Training Pipeline with strategy: {strategy}")
    logger.info(f"{'='*70}\n")

    # 1. Loading Dataset from source
    Data_Path = Path("data/raw/Loan_default.csv")
    df = ingest_data(filepath=str(Data_Path))

    # 2. Validation of Dataset
    df_validated = validate_data(df)

    # 3. Preprocessing (split into train/test)
    X_train, X_test, y_train, y_test = data_preprocess(df_validated)

    # 4. Handle Imbalance (apply sampling technique)
    X_train_balanced, y_train_balanced = handle_imbalance(
        X_train=X_train,
        y_train=y_train,
        strategy=strategy,
        random_state=42
    )

    # 5. Train model on balanced data
    model = train_model(
        X_train=X_train_balanced,
        y_train=y_train_balanced
    )

    # 6. Evaluate on test set
    metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        strategy_name=strategy
    )
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Pipeline completed successfully!")
    logger.info(f"{'='*70}\n")

    return model, metrics


@pipeline
def experimentation_pipeline(strategies: list = None):

    if strategies is None:
        strategies = [
            'none',           # Baseline (no sampling)
            'random_over',    # Random oversampling
            'random_under',   # Random undersampling
            'smote',          # Synthetic Minority Oversampling
            'adasyn',         # Adaptive Synthetic Sampling
            'smote_tomek',    # SMOTE + Tomek Links
            'smote_enn'       # SMOTE + ENN
        ]
    
    logger.info(f"\n{'='*70}")
    logger.info(f"🔬 EXPERIMENTATION PIPELINE - Testing {len(strategies)} strategies")
    logger.info(f"{'='*70}\n")
    
    results = {}
    
    for strategy in strategies:
        logger.info(f"\n{'─'*70}")
        logger.info(f"Testing strategy: {strategy.upper()}")
        logger.info(f"{'─'*70}\n")
        
        model, metrics = training_pipeline(strategy=strategy)
        results[strategy] = metrics
    
    # Compare results
    logger.info(f"\n\n{'='*70}")
    logger.info("📊 SUMMARY: COMPARING ALL STRATEGIES")
    logger.info(f"{'='*70}\n")
    
    comparison_df = pd.DataFrame(results).T
    print(comparison_df[['accuracy', 'precision', 'recall', 'f1_score', 'false_negative_rate']])
    
    logger.info(f"\n{'='*70}")
    logger.info("✅ Experimentation completed!")
    logger.info(f"{'='*70}\n")
    
    return results
    