import os
import sys
import dill
import logging
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

try:
    from src.expection import CustomException
except ModuleNotFoundError:
    # If utils is executed or imported when the project root isn't on sys.path,
    # add the project root (one level up from 'src') so imports work.
    _current_dir = os.path.dirname(__file__)
    _project_root = os.path.abspath(os.path.join(_current_dir, ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from src.expection import CustomException

def save_object(file_path, obj):
    """
    Save a Python object to a file using dill.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load a Python object from a file using dill.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def setup_logging(log_file="logs/app.log"):
    """
    Configure logging for the application.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param: dict):
    """
    Train and evaluate multiple models. For each model in `param`:
      - if params dict is non-empty, perform GridSearchCV and pick best estimator
      - otherwise, fit the provided model directly
    Returns a tuple: (scores_dict, best_estimators_dict)
    """
    try:
        report_scores = {}
        best_estimators = {}

        for model_name, config in param.items():
            model = config.get("model")
            parameters = config.get("params", {}) or {}

            if parameters:
                gs = GridSearchCV(model, parameters, cv=3, n_jobs=-1, verbose=0)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                # No hyperparameters to tune; fit model directly
                best_model = model
                best_model.fit(X_train, y_train)

            # Ensure the best model is fitted
            try:
                # some estimators require a second fit call in certain flows
                if not hasattr(best_model, "predict"):
                    best_model.fit(X_train, y_train)
            except Exception:
                # ignore, we'll call predict and let exceptions bubble up
                pass

            y_pred = best_model.predict(X_test)
            score = accuracy_score(y_test, y_pred)

            report_scores[model_name] = score
            best_estimators[model_name] = best_model
            logging.info(f"{model_name} accuracy: {score:.4f}")

        return report_scores, best_estimators
    except Exception as e:
        raise CustomException(e, sys)
