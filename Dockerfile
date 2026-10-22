FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py .

ARG MODEL_ARTIFACT_PATH=mlruns/1/models/m-f6dd84f3b42a417d90606eaefa5b9463/artifacts
COPY ${MODEL_ARTIFACT_PATH} /app/model

ENV MODEL_URI=/app/model

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]