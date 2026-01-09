import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
import logging

logger = logging.getLogger(__name__)


class ImbalanceHandler(ABC):
    
    @abstractmethod
    def balance(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        pass
    
    def get_class_distribution(self, y: pd.Series) -> Dict[Any, int]:
  
        return y.value_counts().to_dict()
    
    def log_distribution(self, y_before: pd.Series, y_after: pd.Series) -> None:
 
        dist_before = self.get_class_distribution(y_before)
        dist_after = self.get_class_distribution(y_after)
        
        logger.info("📊 Class Distribution:")
        logger.info(f"   Before: {dist_before}")
        logger.info(f"   After:  {dist_after}")
        
        # Calculate imbalance ratio
        majority_before = max(dist_before.values())
        minority_before = min(dist_before.values())
        ratio_before = majority_before / minority_before
        
        majority_after = max(dist_after.values())
        minority_after = min(dist_after.values())
        ratio_after = majority_after / minority_after
        
        logger.info(f"   Imbalance Ratio: {ratio_before:.2f} → {ratio_after:.2f}")


class NoResampling(ImbalanceHandler):

    
    def balance(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Return original data without any modifications."""
        logger.info("🔹 Strategy: No Resampling (Baseline)")
        self.log_distribution(y_train, y_train)
        return X_train, y_train


class RandomOversampling(ImbalanceHandler):
    
    def __init__(self, sampling_strategy: str = 'auto', random_state: int = 42):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
    
    def balance(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply random oversampling to minority class."""
        logger.info("🔹 Strategy: Random Oversampling")
        
        # Combine X and y for resampling
        df = X_train.copy()
        df['target'] = y_train.values
        
        # Separate majority and minority classes
        majority_class = y_train.value_counts().idxmax()
        minority_class = y_train.value_counts().idxmin()
        
        df_majority = df[df['target'] == majority_class]
        df_minority = df[df['target'] == minority_class]
        
        # Oversample minority class
        df_minority_upsampled = resample(
            df_minority,
            replace=True,
            n_samples=len(df_majority),
            random_state=self.random_state
        )
        
        # Combine back
        df_balanced = pd.concat([df_majority, df_minority_upsampled])
        df_balanced = df_balanced.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        X_balanced = df_balanced.drop('target', axis=1)
        y_balanced = df_balanced['target']
        
        self.log_distribution(y_train, y_balanced)
        return X_balanced, y_balanced


class RandomUndersampling(ImbalanceHandler):
    
    def __init__(self, sampling_strategy: str = 'auto', random_state: int = 42):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
    
    def balance(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply random undersampling to majority class."""
        logger.info("🔹 Strategy: Random Undersampling")
        
        rus = RandomUnderSampler(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state
        )
        
        X_balanced, y_balanced = rus.fit_resample(X_train, y_train)
        
        # Convert back to DataFrame/Series
        X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)
        y_balanced = pd.Series(y_balanced, name=y_train.name)
        
        self.log_distribution(y_train, y_balanced)
        return X_balanced, y_balanced


class SMOTEResampling(ImbalanceHandler):
    
    def __init__(
        self, 
        sampling_strategy: str = 'auto', 
        k_neighbors: int = 5,
        random_state: int = 42
    ):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state
    
    def balance(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE to generate synthetic minority samples."""
        logger.info("🔹 Strategy: SMOTE (Synthetic Minority Oversampling)")
        
        smote = SMOTE(
            sampling_strategy=self.sampling_strategy,
            k_neighbors=self.k_neighbors,
            random_state=self.random_state
        )
        
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
        
        # Convert back to DataFrame/Series
        X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)
        y_balanced = pd.Series(y_balanced, name=y_train.name)
        
        self.log_distribution(y_train, y_balanced)
        logger.info(f"   📍 Created {len(y_balanced) - len(y_train)} synthetic samples")
        return X_balanced, y_balanced


class ADAsynResampling(ImbalanceHandler):
    
    def __init__(
        self, 
        sampling_strategy: str = 'auto',
        n_neighbors: int = 5,
        random_state: int = 42
    ):
        self.sampling_strategy = sampling_strategy
        self.n_neighbors = n_neighbors
        self.random_state = random_state
    
    def balance(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply ADASYN for adaptive synthetic sampling."""
        logger.info("🔹 Strategy: ADASYN (Adaptive Synthetic Sampling)")
        
        adasyn = ADASYN(
            sampling_strategy=self.sampling_strategy,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state
        )
        
        X_balanced, y_balanced = adasyn.fit_resample(X_train, y_train)
        
        # Convert back to DataFrame/Series
        X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)
        y_balanced = pd.Series(y_balanced, name=y_train.name)
        
        self.log_distribution(y_train, y_balanced)
        return X_balanced, y_balanced


class SMOTETomekResampling(ImbalanceHandler):

    
    def __init__(self, sampling_strategy: str = 'auto', random_state: int = 42):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
    
    def balance(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE + Tomek Links hybrid approach."""
        logger.info("🔹 Strategy: SMOTE + Tomek Links (Hybrid)")
        
        smote_tomek = SMOTETomek(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state
        )
        
        X_balanced, y_balanced = smote_tomek.fit_resample(X_train, y_train)
        
        # Convert back to DataFrame/Series
        X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)
        y_balanced = pd.Series(y_balanced, name=y_train.name)
        
        self.log_distribution(y_train, y_balanced)
        return X_balanced, y_balanced


class SMOTEENNResampling(ImbalanceHandler):
    
    def __init__(self, sampling_strategy: str = 'auto', random_state: int = 42):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
    
    def balance(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE + ENN hybrid approach."""
        logger.info("🔹 Strategy: SMOTE + ENN (Hybrid)")
        
        smote_enn = SMOTEENN(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state
        )
        
        X_balanced, y_balanced = smote_enn.fit_resample(X_train, y_train)
        
        # Convert back to DataFrame/Series
        X_balanced = pd.DataFrame(X_balanced, columns=X_train.columns)
        y_balanced = pd.Series(y_balanced, name=y_train.name)
        
        self.log_distribution(y_train, y_balanced)
        return X_balanced, y_balanced


class ImbalanceHandlerFactory:
    
    STRATEGIES = {
        'none': NoResampling,
        'random_over': RandomOversampling,
        'random_under': RandomUndersampling,
        'smote': SMOTEResampling,
        'adasyn': ADAsynResampling,
        'smote_tomek': SMOTETomekResampling,
        'smote_enn': SMOTEENNResampling,
    }
    
    @classmethod
    def create_handler(
        cls, 
        strategy: str, 
        **kwargs
    ) -> ImbalanceHandler:
 
        if strategy not in cls.STRATEGIES:
            available = ', '.join(cls.STRATEGIES.keys())
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available strategies: {available}"
            )
        
        handler_class = cls.STRATEGIES[strategy]
        return handler_class(**kwargs)
    
    @classmethod
    def list_strategies(cls) -> list:
      
        return list(cls.STRATEGIES.keys())
    
    from zenml import step
from typing_extensions import Annotated
import pandas as pd
from .imbalance_handler import ImbalanceHandlerFactory
import logging

logger = logging.getLogger(__name__)

from zenml import step


from zenml import step
from typing_extensions import Annotated

@step
def handle_imbalance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str,
    random_state: int = 42
) -> Tuple[Annotated[pd.DataFrame, "X_train_balanced"],Annotated[pd.Series, "y_train_balanced"]]:
    """Handle imbalance using specified strategy"""
    logger.info(f"Applying imbalance strategy: {strategy}")

    handler = ImbalanceHandlerFactory.create_handler(
        strategy=strategy,
        random_state=random_state
    )

    X_balanced, y_balanced = handler.balance(X_train, y_train)
    return X_balanced, y_balanced
