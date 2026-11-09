FROM python:3.12-slim

WORKDIR /srv

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .
COPY models/ ./models/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:aplicacao --host 0.0.0.0 --port ${PORT:-8000}"]
