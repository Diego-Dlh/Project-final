"""
Propósito: API REST para LLM Ollama Phi3
Autor: [Tu Nombre]
Fecha: [Fecha]
"""
from fastapi import FastAPI, Request
import ollama
import logging
import os

app = FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    mensaje = data.get("mensaje")
    try:
        respuesta = ollama.chat(model="phi3", prompt=mensaje)
        logging.info({"request": mensaje, "response": respuesta})
        return {"respuesta": respuesta}
    except Exception as e:
        logging.error({"error": str(e)})
        return {"error": str(e)}

