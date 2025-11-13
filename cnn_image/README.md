# Cat Image Recognition (CNN)

Reconoce si una imagen contiene un gato usando una CNN sencilla.

## Entrenamiento
Estructura tu dataset en data/train/cat/ y data/train/not_cat/
Ejecuta:
python app/train.py

## Predicción API
POST /predict
Content-Type: multipart/form-data (imagen .jpg)
Respuesta: {"prediction": "cat"}

## Docker
docker build -t cat-cnn .
docker run -p 8000:8000 cat-cnn
