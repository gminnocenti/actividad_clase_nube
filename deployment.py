## STEP 1: Set up the workspace.

import json
from dotenv import load_dotenv
import os

# Load variables from .env file into the environment
load_dotenv()

# Now you can access them using os.getenv
key = os.getenv("key")



from azureml.core import Workspace

ws = Workspace.create(name="Polytopia-workspace",
                      subscription_id = key, #CAMBIALO BEIGE
                      resource_group = "resource_group-polytopia",
                      location = "centralindia")


# register model 
from azureml.core.model import Model

mname = "model_class_2"
registered_model = Model.register(model_path="model.pkl",
                                  model_name=mname,
                                  workspace=ws)

from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies

env = Environment(name="env-bankruptcy-tf")
env.python.conda_dependencies = CondaDependencies.create(conda_packages=['pandas','scikit-learn','tensorflow','numpy'])



from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

inference_config = InferenceConfig(
    environment=env,
    entry_script="score.py"  # This is the script that will be run for inference
)

aci_config = AciWebservice.deploy_configuration(
    cpu_cores=0.5,
    memory_gb=1
)

service = Model.deploy(
    workspace=ws,
    name="actividad1",
    models=[registered_model],
    inference_config=inference_config,
    deployment_config=aci_config,
    overwrite=True
)

service.wait_for_deployment(show_output=True)
print(service.get_logs())


scoring_uri = service.scoring_uri

scoreuri = json.dumps({"URI": [scoring_uri]})
file = open("uri.json", "w")
file.write(scoreuri)
file.close()