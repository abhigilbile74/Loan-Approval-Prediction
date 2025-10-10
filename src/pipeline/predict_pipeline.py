import sys
import pandas as pd
import numpy as np
import os

from src.logger import logging
from src.utils import load_object

try:
    from src.expection import CustomException
except ModuleNotFoundError:
    _current_dir = os.path.dirname(__file__)
    _project_root = os.path.abspath(os.path.join(_current_dir, ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from src.expection import CustomException


class PredictPipline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            logging.info(f"prprocessor,{preprocessor}")
            data_scaled = preprocessor.transform(features)
            logging.info(f"data_scaled,{data_scaled}")
            preds = model.predict(data_scaled)
            logging.info(f"preds.{preds}")
            return preds
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 Gender: str,
                 Married: str,
                 Dependents: int,
                 Education: str,
                 Self_Employed: str,
                 ApplicantIncome: float,
                 CoapplicantIncome: float,
                 LoanAmount: float,
                 Loan_Amount_Term: float,
                 Credit_History: float,
                 Property_Area: str):
        self.Gender = Gender
        self.Married = Married
        self.Dependents = Dependents
        self.Education = Education
        self.Self_Employed = Self_Employed
        self.ApplicantIncome = ApplicantIncome
        self.CoapplicantIncome = CoapplicantIncome
        self.LoanAmount = LoanAmount
        self.Loan_Amount_Term = Loan_Amount_Term
        self.Credit_History = Credit_History
        self.Property_Area = Property_Area

    def get_data_as_dataframe(self):
        try:
            data = {
                "Gender": [self.Gender],
                "Married": [self.Married],
                "Dependents": [self.Dependents],
                "Education": [self.Education],
                "Self_Employed": [self.Self_Employed],
                "ApplicantIncome": [self.ApplicantIncome],
                "CoapplicantIncome": [self.CoapplicantIncome],
                "LoanAmount": [self.LoanAmount],
                "Loan_Amount_Term": [self.Loan_Amount_Term],
                "Credit_History": [self.Credit_History],
                "Property_Area": [self.Property_Area],
            }
            
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e, sys)
