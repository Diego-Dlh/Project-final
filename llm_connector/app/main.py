from dotenv import load_dotenv
from google import genai
from google.genai import types
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

class Message(BaseModel):
    role: str
    content: str

app = FastAPI()

# Carga variables de entorno (asegúrate de tener Gemini_API_KEY definida)
load_dotenv()
genai_api_key = os.getenv("Gemini_API_KEY")

if not genai_api_key:
    raise ValueError("No se encontró la API KEY: Gemini_API_KEY")

client = genai.Client(api_key=genai_api_key)

@app.post("/chat")
def generateResponse(messages: List[Message]):
    gemini_messages = []

    # Mapeo roles según especificación Gemini
    for m in messages:
        role_map = 'model' if m.role.lower() == 'assistant' else 'user'
        content_object = types.Content(
            role=role_map,
            parts=[types.Part(text=m.content)]
        )
        gemini_messages.append(content_object)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=gemini_messages
        )
        answer = response.text
        return {"response": answer.strip()}
    except Exception as e:
        return {"response": f"error del LLM: {e}"}
