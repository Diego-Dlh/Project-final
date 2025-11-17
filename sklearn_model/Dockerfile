FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY SMSSpamCollection ./
COPY app/ ./app/
COPY pipeline/ ./pipeline/
COPY tests/ ./tests/
COPY spam_model.pkl ./
EXPOSE 8000

CMD ["uvicorn", "app.predict:app", "--host", "0.0.0.0", "--port", "8000"]
