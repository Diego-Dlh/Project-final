# SMS Spam Classifier

Clasifica mensajes SMS como spam o no spam.

## Entrenamiento
Ejecutar:
python app/train.py

## Consumo API
POST /predict
JSON: {"message": "Win a free car!"}
Respuesta: {"prediction": "spam"}

## Docker
docker build -t spam-classifier .
docker run -p 8000:8000 spam-classifier
