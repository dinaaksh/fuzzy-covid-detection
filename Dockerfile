FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

COPY app_utils/ /app/app_utils

COPY src/ /app/src/

COPY artifacts/model_trainer/final_model.pkl /app/artifacts/model_trainer/
COPY artifacts/model_trainer/model_features.pkl /app/artifacts/model_trainer/

ENV PYTHONPATH="/app/src"

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]