import sys
import os
# Make imports robust when running the script directly (so 'src' package is importable).
try:
    from src.expection import CustomException
except ModuleNotFoundError:
    # If running the file directly, the package root may not be on sys.path.
    # Add the project root (two levels up from this file) to sys.path and retry.
    _current_dir = os.path.dirname(__file__)
    _project_root = os.path.abspath(os.path.join(_current_dir, "..", ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from src.expection import CustomException


import pandas as pd

# Import components using package paths so imports work whether the module is run
# directly or as a package (python -m src.components.data_ingestion)
from src.components.data_transformation import DataTransformationConfig, DataTransformation
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer



from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import logging

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts","train.csv")
    test_data_path: str=os.path.join("artifacts","test.csv")
    raw_data_path: str=os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        pass
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv("Notebook/data/LoanApprovalPrediction.csv")
            logging.info("Read the dataset as dataframe ")
            df.drop(['Loan_ID'],axis=1,inplace=True) # drop the id col

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.4,random_state=1)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("ingestion of the data is complted ")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
            

        
if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    # Train on training array and evaluate on test array
    model_path, metrics = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print('model_path ->', model_path)
    print('metrics ->', metrics)