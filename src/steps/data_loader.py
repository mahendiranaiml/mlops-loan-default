from zenml import step
import pandas as pd
import logging
from typing_extensions import Annotated
logger = logging.getLogger(__name__)
from abc import ABC, abstractmethod

#Abstract Class for data loading
class DataLoader(ABC):

    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass


#Concrete Class for data loading
class CSVDataLoader(DataLoader):

    def __init__(self, filepath: str):
        self.filepath = filepath

    def load(self) -> pd.DataFrame:

        try:
            df = pd.read_csv(self.filepath)
            logger.info("🎉 Data Loaded Successfully")
            logger.info(f"Rows : {df.shape[0]} : Columns : {df.shape[1]}")
            logger.info(f"Columns : {df.columns.tolist()}")
            logger.info(f"5 Rows of DataSet : {df.head()}")
            return df
        except FileNotFoundError:
            logger.error(f"File Not Found at {self.filepath}")
            raise

        except Exception as e:
            logger.error(f"Error Data Loading {e}")
            raise

@step
def ingest_data(filepath: str) -> Annotated[pd.DataFrame, "raw_data"]:
    """
    Data Ingestor for Zenml step
    """
    loader = CSVDataLoader(filepath)
    df = loader.load()
    return df