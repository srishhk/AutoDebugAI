FROM python:3.11-slim

WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy code and model artifacts
COPY src/ src/
COPY models/ models/

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]