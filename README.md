# Cloud Computing – Actividad Semana 2

Este repositorio contiene el desarrollo de una actividad práctica enfocada en la creación, entrenamiento y despliegue de un modelo de clasificación utilizando **XGBoost** para predecir si un usuario caerá en bancarrota.

## Objetivo

Construir un flujo completo de machine learning, desde el preprocesamiento de datos hasta el despliegue del modelo en la nube utilizando **Azure Machine Learning**.

## Estructura del Equipo

El proyecto fue dividido entre tres equipos principales:

### Departamento de Datos

- Responsable del archivo `preprocessing.py`.
- Seleccionó y procesó las características más relevantes para predecir la variable objetivo: **`Bankrupt ?`**.

### Departamento de Modelos

- Encargado del archivo `Model.py`.
- Entrenó un modelo de clasificación utilizando **XGBoost**.
- Exportó el modelo entrenado como archivo `.pkl` (pickle) para su posterior despliegue.

### Departamento de Cómputo en la Nube

- Encargado del archivo `deployment.py`.
- Creó un **Azure Workspace**, registró el modelo y desplegó el servicio.
- El endpoint del modelo está documentado en el archivo `uri.json`.
- También desarrolló el archivo `API.py`, el cual demuestra que el URI funciona correctamente al conectarse mediante una API y generar predicciones utilizando el archivo `prueba.csv`.

## Tecnologías Utilizadas

- Python
- XGBoost
- Azure Machine Learning
- Pickle
- JSON

## Pasos para Desplegar el Proyecto

1. **Instala las dependencias.**  
   Asegúrate de tener Python instalado. Luego, en la terminal, ejecuta:

   ```bash
   pip install -r requirements.txt

2. **Haz un fork del repositorio.**  
   Clona el repositorio en tu máquina local usando Git o descarga el ZIP.

3. **Entrena el modelo.**  
   Ejecuta el archivo `Model.py`, el cual llama a la clase `PreprocessData` del archivo `preprocessing.py` para limpiar la base de datos, entrenar un modelo de clasificación con **XGBoost** y guardarlo como un archivo `.pkl`.

4. **Configura el archivo `deployment.py`.**  
   Abre el archivo `deployment.py` y dirígete a la línea 15. Reemplaza la palabra `key` con tu ID de suscripción de Azure en el siguiente fragmento de código:

   ```python
   ws = Workspace.create(name="workspace_class4",
                         subscription_id=key,  # reemplaza este valor con tu ID de suscripción
                         resource_group="class_resource_group4",
                         location="brazilsouth")
    ```

5. **Despliega el modelo en Azure.**  
   Ejecuta el archivo `deployment.py`. Este proceso creará un servicio en Azure y generará un archivo `uri.json` que contiene el endpoint del modelo desplegado.  
   Al correr este archivo se llama también al archivo `score.py`, que es utilizado para definir cómo se reciben los datos de entrada y cómo se retorna la predicción del modelo. 


6. **Prueba el endpoint con una API.**
Ejecuta el archivo API.py, el cual se conecta al URI del servicio y realiza predicciones utilizando los datos de prueba contenidos en prueba.csv.
