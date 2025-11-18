import pytest
from fastapi.testclient import TestClient
from app.predict import app
from PIL import Image
import io

client = TestClient(app)

@pytest.fixture
def sample_image():
    # Crear imagen blanca con un dígito dibujado puede ser complejo,
    # mejor cargar un archivo png de prueba ya existente.
    # Aquí se crea una imagen en blanco 28x28 para prueba rápida:
    img = Image.new("L", (28, 28), color=0)  # fondo negro
    return img

def test_predict_endpoint(sample_image):
    buf = io.BytesIO()
    sample_image.save(buf, format="PNG")
    buf.seek(0)
    response = client.post("/predict", files={"file": ("digit.png", buf, "image/png")})
    assert response.status_code == 200
    json_response = response.json()
    assert "prediction" in json_response
    assert isinstance(json_response["prediction"], int)
    assert 0 <= json_response["prediction"] <= 9
