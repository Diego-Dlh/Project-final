import gradio as gr
import requests

LLM_URL = "http://localhost:8000/chat"
SPAM_URL = "http://localhost:8000/predict"
CNN_URL = "http://localhost:8000/predict"

def llm_chat(prompt):
    response = requests.post(LLM_URL, json={"prompt": prompt})
    return response.json().get("response", "Error en LLM")

def spam_predict(text):
    response = requests.post(SPAM_URL, json={"message": text})
    return response.json().get("prediction", "Error en spam model")

def cnn_predict(image):
    image.save("temp.jpg")
    files = {"file": open("temp.jpg", "rb")}
    response = requests.post(CNN_URL, files=files)
    return response.json().get("prediction", "Error en CNN")

with gr.Blocks() as demo:
    gr.Markdown("# Proyecto MLOps - Interfaz Unificada")
    with gr.Tab("Chat LLM"):
        inp = gr.Textbox(label="Prompt")
        out = gr.Textbox(label="Respuesta")
        inp.submit(llm_chat, inp, out)
    with gr.Tab("Clasificación Spam SMS"):
        inp2 = gr.Textbox(label="Mensaje SMS")
        out2 = gr.Textbox(label="Predicción")
        inp2.submit(spam_predict, inp2, out2)
    with gr.Tab("Clasificación MNIST"):
        inp3 = gr.Image(type="pil", label="Imagen")
        out3 = gr.Textbox(label="Predicción")
        inp3.change(cnn_predict, inp3, out3)
    
    gr.Markdown("**Asegúrate que los servicios backend estén corriendo y en las URLs correctas.**")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
