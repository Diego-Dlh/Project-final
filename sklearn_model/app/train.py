"""
Entrenamiento modelo regresión lineal, logging MLflow
Autor: [Tu Nombre]
Fecha: [Fecha]
"""
import mlflow
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.start_run(run_name="linear_regression")

data = pd.read_csv("data/train.csv")
X = data.drop("target", axis=1)
y = data["target"]

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("reg", LinearRegression())
])
pipeline.fit(X, y)

mlflow.sklearn.log_model(pipeline, "model")
mlflow.log_param("model_type", "LinearRegression")
mlflow.log_metric("score", pipeline.score(X, y))

dump(pipeline, "models/reg_model.joblib")
mlflow.end_run()
