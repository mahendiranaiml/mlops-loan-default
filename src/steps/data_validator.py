import pandas as pd
import logging
from zenml import step

logger = logging.getLogger(__name__)
from abc import ABC, abstractmethod


class DataValidator(ABC):
    """
    Abstract base class for all data validators.
    """

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate the input dataframe.
        Must raise exception if validation fails.
        Must return df if validation passes.
        """
        pass

class LoanDataValidator(DataValidator):

    REQUIRED_COLUMNS = [
        'LoanID',
        'Age',
        'Income',
        'LoanAmount',
        'CreditScore',
        'MonthsEmployed',
        'NumCreditLines',
        'InterestRate',
        'LoanTerm',
        'DTIRatio',
        'Education',
        'EmploymentType',
        'MaritalStatus',
        'HasMortgage',
        'HasDependents',
        'LoanPurpose',
        'HasCoSigner',
        'Default']

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting loan data validation")

        # 1. Empty dataset check
        if df.empty:
            raise ValueError("Dataset is empty")

        # 2. Required column check
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # 3. Null percentage check
        null_ratio = df.isnull().mean()
        high_null_cols = null_ratio[null_ratio > 0.3]

        if not high_null_cols.empty:
            raise ValueError(
                f"Columns with >30% nulls: {high_null_cols.index.tolist()}"
            )

        logger.info("Loan data validation passed successfully")
        return df


@step
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    ZenML step for data validation.
    """
    validator = LoanDataValidator()
    validated_df = validator.validate(df)

    return validated_df
