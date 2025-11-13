import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

model = joblib.load('spam_model.pkl')

app = FastAPI()

class TextRequest(BaseModel):
    message: str

@app.post("/predict")
def predict(req: TextRequest):
    try:
        pred = model.predict([req.message])
        label = "spam" if pred[0] else "ham"
        return {"prediction": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
