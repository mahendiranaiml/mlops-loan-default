import pandas as pd
import numpy as np
from typing import Any, Dict
from typing_extensions import Annotated
from zenml import step
import logging
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    matthews_corrcoef
)

logger = logging.getLogger(__name__)


@step
def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    strategy_name: str = "default"
) -> Annotated[Dict[str, float], "metrics"]:

    logger.info(f"📊 Evaluating model with strategy: {strategy_name}")
    logger.info(f"   Test data shape: {X_test.shape}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = None
    
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    except (AttributeError, IndexError):
        logger.warning("   ⚠️  Model does not support predict_proba()")
    
    # Confusion Matrix Analysis
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary'
    )
    
    # 🎯 KEY METRICS FOR FALSE NEGATIVE REDUCTION:
    
    # False Negative Rate (FNR) - Proportion of actual defaults we miss
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # False Positive Rate (FPR)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Specificity (True Negative Rate) - Correctly identified non-defaults
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Matthews Correlation Coefficient (good for imbalanced data)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # AUC-ROC (Area Under ROC Curve) - Only if probability predictions available
    auc_roc = None
    if y_pred_proba is not None:
        auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    # Compile metrics
    metrics = {
        'strategy': strategy_name,
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': float(precision),
        'recall': float(recall),  # Also: Sensitivity, True Positive Rate
        'f1_score': float(f1),
        'false_negative_rate': float(fnr),  # 🎯 MINIMIZE THIS!
        'false_positive_rate': float(fpr),
        'specificity': float(specificity),
        'mcc': float(mcc),
        'auc_roc': float(auc_roc) if auc_roc is not None else None,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),  # 🎯 MINIMIZE THIS!
    }
    
    # Log detailed results
    logger.info("=" * 70)
    logger.info(f"📈 EVALUATION RESULTS - Strategy: {strategy_name}")
    logger.info("=" * 70)
    logger.info(f"Accuracy:                {metrics['accuracy']:.4f}")
    logger.info(f"Precision:               {metrics['precision']:.4f}")
    logger.info(f"Recall (Sensitivity):    {metrics['recall']:.4f}")
    logger.info(f"F1-Score:                {metrics['f1_score']:.4f}")
    logger.info(f"MCC:                     {metrics['mcc']:.4f}")
    if auc_roc is not None:
        logger.info(f"AUC-ROC:                 {metrics['auc_roc']:.4f}")
    logger.info("")
    logger.info("🎯 FALSE NEGATIVE FOCUS (For Loan Default Prevention):")
    logger.info(f"   False Negative Rate:   {metrics['false_negative_rate']:.4f} ← MINIMIZE THIS!")
    logger.info(f"   False Negatives:       {fn} defaults we MISSED!")
    logger.info("")
    logger.info("Confusion Matrix:")
    logger.info(f"   TP (Correctly predicted defaults):        {tp}")
    logger.info(f"   TN (Correctly predicted non-defaults):    {tn}")
    logger.info(f"   FP (False alarms):                        {fp}")
    logger.info(f"   FN (Missed defaults) 🎯:                  {fn}")
    logger.info("=" * 70)
    
    return metrics
