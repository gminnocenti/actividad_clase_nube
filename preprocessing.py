"""Preprocessing script for the Bankruptcy Prediction dataset."""

import json
import kagglehub
import pandas as pd
from imblearn.over_sampling import SMOTE
from statsmodels.stats.outliers_influence import variance_inflation_factor

path = kagglehub.dataset_download("fedesoriano/company-bankruptcy-prediction")
data = pd.read_csv(path + "/data.csv")


class PreprocessData:
    """Demonstration of how to use the class!

    preprocessor = PreprocessData(data)
    x_train, y_train = preprocessor.preprocess_and_split_data()

    >>> Returns the preprocessed data.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def preprocess_and_split_data(self, random_state: int = 42):
        X = data.drop(columns=["Bankrupt?"])
        y = data["Bankrupt?"]
        smote = SMOTE(random_state=random_state)
        X_smote, y_smote = smote.fit_resample(X, y)

        vif_data_smote = pd.DataFrame()
        vif_data_smote["feature"] = X_smote.columns
        vif_data_smote["VIF"] = [
            variance_inflation_factor(X_smote.values, i)
            for i in range(X_smote.shape[1])
        ]

        high_vif_features_smote = vif_data_smote[vif_data_smote["VIF"] > 5]

        X_smote_reduced = X_smote.drop(columns=high_vif_features_smote["feature"])
        return X_smote_reduced, y_smote

    def get_data(self):
        return self.data
