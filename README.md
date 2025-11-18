# Proyecto MLOps: Plataforma de Modelos Integrados con FastAPI, Gradio y Gemini

Este proyecto integra distintos microservicios de Machine Learning orquestados con Docker Compose y Docker Swarm. Incluye:
- Un conector LLM (Google Gemini)
- Clasificador Spam (Scikit-learn)
- Clasificador de Imágenes (CNN/MNIST)
- Frontend unificado con Gradio
- MLflow para Tracking/registro de experimentos

## Estructura del Proyecto
<pre>
├── llm_connector/
├── gradio_frontend/
├── sklearn_model/
├── cnn_image/
├── mlflow/
├── docker-compose.yml
├── docker-stack.yml
├── .github/
│ └── workflows/
│ └── ci.yml
└── README.md
</pre>

## Requerimientos

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/)
- [Docker Swarm](https://docs.docker.com/engine/swarm/)
- Cuenta en [Google AI Studio](https://aistudio.google.com/) (para la API Key de Gemini)
- Cuenta en [Docker Hub](https://hub.docker.com/) (para CI/CD)

## Instalación y Uso: Desarrollo local

1. Clona el repositorio:
    ```
    git clone https://github.com/tuusuario/tu_repo.git
    cd tu_repo
    ```

2. Crea un archivo `.env` en la raíz con tu clave API y variables necesarias:
    ```
    GEMINI_API_KEY=tu_api_key_de_gemini
    ```

3. Levanta la plataforma localmente:
    ```
    docker compose up --build
    ```
    Accede al panel Gradio en [http://localhost:7860](http://localhost:7860).

## CI/CD con GitHub Actions

- Cada push a `main` dispara el workflow en `.github/workflows/ci.yml`.
- Se construyen y publican todas las imágenes a Docker Hub.
- Configura los siguientes secretos en tu repo:
    - `DOCKER_USERNAME`
    - `DOCKER_PASSWORD`
    - OTRAS CLAVES que automatices (ejemplo: claves HuggingFace, etc).

## Despliegue en producción (Docker Swarm)

1. Prepara las variables de entorno necesarias en el servidor (`GEMINI_API_KEY`, etc).
2. Inicializa Swarm (solo la primera vez):
    ```
    docker swarm init
    ```
3. Despliega la stack:
    ```
    docker stack deploy -c docker-stack.yml mlops_stack
    ```
4. Monitorea y escala servicios con:
    ```
    docker service ls
    ```

## Endpoints principales

- **Gradio Frontend:** [http://localhost:7860](http://localhost:7860)
- **LLM Connector API:** [http://localhost:8001/chat](http://localhost:8001/chat)  
    (entrada: lista JSON de mensajes, ver ejemplos)
- **Clasificador Spam:** [http://localhost:8002/predict](http://localhost:8002/predict)
- **Clasificador Imagen:** [http://localhost:8003/predict](http://localhost:8003/predict)
- **MLflow:** [http://localhost:5000](http://localhost:5000)

## Ejemplo de uso del LLM Connector

POST a `/chat` con:
[
{"role": "user", "content": "¿Cuál es la capital de Francia?"}
]

## Licencia
MIT

## Autores
- Diego De la hoz
- Camilo Ramos
