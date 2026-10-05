FROM truthlens-ai-api:test
COPY services/ai-api/requirements-dev.txt /app/services/ai-api/requirements-dev.txt
RUN pip install --no-cache-dir -r /app/services/ai-api/requirements-dev.txt
WORKDIR /app/services/ai-api
CMD ["sh", "-c", "ruff check . && pytest -q"]
