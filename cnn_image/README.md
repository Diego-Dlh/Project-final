# MNIST CNN Classification

Este módulo entrena un simple modelo CNN para clasificar dígitos escritos a mano (0-9).

## Entrenamiento
Ejecutar:
python app/train.py

## API de predicción
POST /predict
Envía una imagen jpg/png en el cuerpo multipart/form-data
Respuesta:
{
  "prediction": int  # dígito 0-9 classificado
}

## Docker
docker build -t mnist-cnn .
docker run -p 8000:8000 mnist-cnn
