import pandas as pd
import json
import requests

data = pd.read_csv("prueba.csv").drop(["Unnamed: 0","Bankrupt?"], axis=1)#(input_data)

data_dict = data.to_dict(orient='list')

#Formateado para la API:
data_json = json.dumps({"data": [data_dict]})
# open uri json
suri = open("uri.json", "r")
scoring_uri = json.load(suri)["URI"][0]
suri.close()


headers = {"Content-Type": "application/json"}
response = requests.post(scoring_uri, data=data_json, headers=headers)

if response.status_code == 200:
    result = json.loads(response.json())
    print(result)
    data["Exited"] = result
    print(data)
else:
  print(f"Error: {response.text}")