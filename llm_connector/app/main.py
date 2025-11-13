from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import ollama

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(req: PromptRequest):
    try:
        result = ollama.chat(model="phi3", prompt=req.prompt)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
