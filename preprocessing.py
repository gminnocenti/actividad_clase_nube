import kagglehub
import pandas as pd
from imblearn.over_sampling import SMOTE
import json
from statsmodels.stats.outliers_influence import variance_inflation_factor

path = kagglehub.dataset_download("fedesoriano/company-bankruptcy-prediction")
data = pd.read_csv(path+"/data.csv")


X = data.drop(columns=['Bankrupt?'])
y = data['Bankrupt?']
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

vif_data_smote = pd.DataFrame()
vif_data_smote["feature"] = X_smote.columns
vif_data_smote["VIF"] = [variance_inflation_factor(X_smote.values, i) for i in range(X_smote.shape[1])]

high_vif_features_smote = vif_data_smote[vif_data_smote["VIF"] > 5]

X_smote_reduced = X_smote.drop(columns=high_vif_features_smote["feature"])

with open('columns_to_drop.json', 'w') as json_file:
    json.dump(list(high_vif_features_smote['feature']), json_file, indent=4)