import json
import joblib
import pandas as pd
from azureml.core.model import Model
import os

def init():
    global model
    global columns_to_drop

    # Load the model
    model_path = Model.get_model_path("model_class_2")
    model = joblib.load(model_path)

    # Load the columns to drop
    drop_path = os.path.join(os.path.dirname(__file__), "columns_to_drop.json")
    with open(drop_path, "r") as f:
        columns_to_drop = json.load(f)

def run(raw_data):
    try:
        # Parse incoming JSON
        inputs = json.loads(raw_data)["data"]
        df = pd.DataFrame(inputs)

        # Drop columns listed in columns_to_drop
        df_cleaned = df.drop(columns=columns_to_drop, errors="ignore")

        # Predict
        predictions = model.predict(df_cleaned)

        return predictions.tolist()
    except Exception as e:
        return {"error": str(e)}
