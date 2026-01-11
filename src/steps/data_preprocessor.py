import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from typing_extensions import Annotated
from sklearn.model_selection import train_test_split
import logging
from zenml import step
from typing import Tuple, Any
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)


class BinaryMapper(BaseEstimator, TransformerMixin):
    
    def __init__(self, mapping=None):
        self.mapping = mapping if mapping is not None else {'Yes': 1, 'No': 0}
    
    def fit(self, X, y=None):
       
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
        return self
    
    def transform(self, X):
        
        if isinstance(X, pd.DataFrame):
            X_copy = X.copy()
            for col in X_copy.columns:
                X_copy[col] = X_copy[col].map(self.mapping)
            return X_copy.values
        else:
            # Handle numpy arrays
            X_copy = X.copy()
            for i in range(X_copy.shape[1]):
                X_copy[:, i] = pd.Series(X_copy[:, i]).map(self.mapping).values
            return X_copy
    
    def get_feature_names_out(self, input_features=None):
       
        if input_features is None:
            if hasattr(self, 'feature_names_in_'):
                input_features = self.feature_names_in_
            else:
                raise ValueError("input_features must be provided if transformer was not fitted with feature names")
        
        return np.array([f"categorical_binary__{name}" for name in input_features])


class Preprocessor(ABC):
    
    @abstractmethod
    def splitter(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets."""
        pass

    @abstractmethod
    def build_pipeline(self) -> Pipeline:
        """Build the preprocessing pipeline."""
        pass

    @abstractmethod   
    def fit_transform(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit on training data and transform both train and test sets."""
        pass


class DataPreprocessor(Preprocessor):
    
    
    def __init__(self):
        self.pipeline = None
        self.numerical_continuous = [
            "Age",
            "Income",
            "LoanAmount",
            "CreditScore",
            "InterestRate",
            "DTIRatio",
            "MonthsEmployed"
        ]
        
        self.categorical_nominal = [
            "EmploymentType",
            "MaritalStatus",
            "LoanPurpose",
            "NumCreditLines",
            "LoanTerm"
        ]
        
        self.categorical_binary = [
            "HasMortgage",
            "HasDependents",
            "HasCoSigner"
        ]

        self.ordinal = ["Education"]
        self.order_education = ["High School", "Bachelor's", "Master's", "PhD"]
      
    def splitter(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        
        X = df.drop(columns=["LoanID", "Default"], axis=1)
        y = df["Default"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            stratify=y, 
            random_state=35
        )

        logger.info(f"✅ Data split complete - Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        return X_train, X_test, y_train, y_test
 
    def build_pipeline(self) -> ColumnTransformer:
    
        # Numerical scaling pipeline
        numerical_scale = Pipeline(steps=[
            ("standardscaler", StandardScaler())
        ])

        # Nominal categorical encoding pipeline
        categorical_onehot = Pipeline(steps=[
            ("onehotencoder", OneHotEncoder(handle_unknown='ignore'))
        ])

        # Ordinal categorical encoding pipeline
        categorical_ordinal = Pipeline(steps=[
            ("ordinal", OrdinalEncoder(
                categories=[self.order_education],
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ))
        ])

        # Binary categorical mapping pipeline
        categorical_map = Pipeline(steps=[
            ("mapping", BinaryMapper())
        ])

        # Combine all transformers
        self.pipeline = ColumnTransformer(
            transformers=[
                ("numerical_scale", numerical_scale, self.numerical_continuous),
                ("categorical_onehot", categorical_onehot, self.categorical_nominal),
                ("categorical_ordinal", categorical_ordinal, self.ordinal),
                ("categorical_binary", categorical_map, self.categorical_binary),
            ],
            remainder='passthrough'
        )
        
        logger.info("✅ Preprocessing pipeline built successfully")
        return self.pipeline
    
    def fit_transform(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
        if self.pipeline is None:
            raise RuntimeError("Pipeline has not been built.")

        # Fit on training data and transform
        X_train_transformed = self.pipeline.fit_transform(X_train)
        
        # Only transform test data (NO FITTING to prevent data leakage)
        X_test_transformed = self.pipeline.transform(X_test)

        # Get feature names from the pipeline
        feature_names = self.pipeline.get_feature_names_out()

        # Convert to DataFrames with proper indexing
        X_train_df = pd.DataFrame(
            X_train_transformed, 
            columns=feature_names, 
            index=X_train.index
        )
        X_test_df = pd.DataFrame(
            X_test_transformed, 
            columns=feature_names, 
            index=X_test.index
        )
        
        logger.info(f"✅ Data transformed - Train shape: {X_train_df.shape}, Test shape: {X_test_df.shape}")

        return X_train_df, X_test_df


@step
def data_preprocess(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:

    logger.info("🚀 Starting data preprocessing step...")
    
    # Initialize preprocessor
    dp = DataPreprocessor()

    # Step 1: Train-test split
    X_train, X_test, y_train, y_test = dp.splitter(df)

    # Step 2: Build pipeline
    preprocessor = dp.build_pipeline()

    # Step 3: Apply transformations
    X_train_df, X_test_df = dp.fit_transform(X_train, X_test)

    logger.info("🎉 Data preprocessing completed successfully!")
    
    return X_train_df, X_test_df, y_train, y_test