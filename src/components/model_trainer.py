import os
import sys
from src.logger import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score

from src.utils import save_object, evaluate_models
try:
    from src.expection import CustomException
except ModuleNotFoundError:
    _current_dir = os.path.dirname(__file__)
    _project_root = os.path.abspath(os.path.join(_current_dir, "..", ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from src.expection import CustomException

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            params = {
                "KNeighborsClassifier": {
                    "model": KNeighborsClassifier(),
                    "params": {"n_neighbors": [3]},
                },
                "RandomForestClassifier": {
                    "model": RandomForestClassifier(),
                    "params": {
                        "n_estimators": [7],
                        "criterion": ["entropy"],
                        "random_state": [7],
                    },
                },
                "SVC": {"model": SVC(), "params": {}},
                "LogisticRegression": {"model": LogisticRegression(), "params": {}},
            }

            model_report, best_estimators = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=None,
                param=params,
            )

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = best_estimators[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info("Best found model on both training and testing dataset")
            logging.info(f"Best model: {best_model_name} with score: {best_model_score}")

            # Save the fitted best model
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            return self.model_trainer_config.trained_model_file_path, accuracy

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # smoke test
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, size=(100,))
    arr = np.c_[X, y]

    trainer = ModelTrainer()
    model_path, acc = trainer.initiate_model_trainer(arr, arr)
    print("model_path ->", model_path)
    print("accuracy ->", acc)
