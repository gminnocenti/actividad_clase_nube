## STEP 1: Set up the workspace.

import json
from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("key")



from azureml.core import Workspace
ws = Workspace.create(name="workspace_class4",
                      subscription_id = key, # replace this value with your subscription id
                      resource_group = "class_resource_group4",
                      location="brazilsouth")

"""
ws = Workspace.get(name="workspace_class",
                      subscription_id = key,
                      resource_group = "class_resource_group")
"""
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig

env = Environment("xgboost-env")
env.python.conda_dependencies.add_pip_package("xgboost")
env.python.conda_dependencies.add_pip_package("numpy")
env.python.conda_dependencies.add_pip_package("joblib")
env.python.conda_dependencies.add_pip_package("pandas")

inference_config = InferenceConfig(entry_script="score.py", environment=env)

from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import Model
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

model = Model.register(workspace=ws,
                       model_path="model.pkl",  
                       model_name="model")
service = Model.deploy(workspace=ws,
                       name="xgboost-service",
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=deployment_config,
                       overwrite=True,)

service.wait_for_deployment(show_output=True)

print(service.get_logs())

print("Scoring URI:", service.scoring_uri)

scoring_uri = service.scoring_uri

scoreuri = json.dumps({"URI": [scoring_uri]})
file = open("uri.json", "w")
file.write(scoreuri)
file.close()