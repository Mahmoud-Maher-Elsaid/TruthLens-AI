FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY services/ai-api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY services/ai-api ./services/ai-api
COPY data ./data
ENV PYTHONPATH=/app/services/ai-api
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
