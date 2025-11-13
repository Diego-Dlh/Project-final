import torch
from PIL import Image
import torchvision.transforms as transforms

class SimpleCNN(torch.nn.Module):  # igual que el de train.py
    # ... def __init__ y forward ...
    pass

model = SimpleCNN()
model.load_state_dict(torch.load("cnn_cat.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    outputs = model(img)
    _, predicted = torch.max(outputs.data, 1)
    return "cat" if predicted.item() == 0 else "not cat"

# API REST (usando FastAPI)
from fastapi import FastAPI, UploadFile, File

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_path = f"temp.jpg"
    with open(img_path, "wb") as buffer:
        buffer.write(await file.read())
    label = predict_image(img_path)
    return {"prediction": label}
