## STEP 1: Set up the workspace.

import json
from dotenv import load_dotenv
import os

# Load variables from .env file into the environment
load_dotenv()

# Now you can access them using os.getenv
key = os.getenv("key")



from azureml.core import Workspace

ws = Workspace.get(name="workspace_class",
                      subscription_id = key,
                      resource_group = "class_resource_group")


from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig

# Create a new environment for deployment
env = Environment("xgboost-env")
env.python.conda_dependencies.add_pip_package("xgboost")
env.python.conda_dependencies.add_pip_package("numpy")
env.python.conda_dependencies.add_pip_package("joblib")
env.python.conda_dependencies.add_pip_package("pandas")

# Set up the inference configuration
inference_config = InferenceConfig(entry_script="score.py", environment=env)

from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import Model
# Define the deployment configuration (adjust resources as needed)
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

# Deploy the model
model = Model.register(workspace=ws,
                       model_path="model.pkl",  # Path to your .pkl file
                       model_name="xgboost_model")
service = Model.deploy(workspace=ws,
                       name="xgboost-service",
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=deployment_config)

service.wait_for_deployment(show_output=True)

print(service.get_logs())

# Print the scoring URI
print("Scoring URI:", service.scoring_uri)

scoring_uri = service.scoring_uri

scoreuri = json.dumps({"URI": [scoring_uri]})
file = open("uri.json", "w")
file.write(scoreuri)
file.close()