import os
import joblib
import json
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init():
    global model, columns_to_drop
    try:
        # Get the model directory from the environment variable.
        model_dir = os.environ.get("AZUREML_MODEL_DIR")
        if not model_dir:
            raise ValueError("AZUREML_MODEL_DIR environment variable not set")
        model_path = os.path.join(model_dir, "model.pkl")
        logger.info("Loading model from: %s", model_path)
        model = joblib.load(model_path)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error("Error loading model: %s", str(e))
        raise e

    try:
        # Load the list of columns to drop from columns_to_drop.json.
        # Ensure that columns_to_drop.json is packaged alongside your score.py.
        script_dir = os.path.dirname(os.path.realpath(__file__))
        columns_path = os.path.join(script_dir, "columns_to_drop.json")
        logger.info("Loading columns to drop from: %s", columns_path)
        with open(columns_path, "r") as f:
            columns_to_drop = json.load(f)
        logger.info("Columns to drop loaded successfully.")
    except Exception as e:
        logger.error("Error loading columns_to_drop.json: %s", str(e))
        raise e

def run(raw_data):
    try:
        input_json = json.loads(raw_data)
        data_df = pd.DataFrame(input_json["data"])
        data_df = data_df.drop(columns=columns_to_drop, errors="ignore")
        predictions = model.predict(data_df)
        return json.dumps({"result": predictions.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})
