FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY grpc_embedder.py .
COPY tei_pb2.py tei_pb2_grpc.py .

EXPOSE 50051

CMD ["python", "grpc_embedder.py"]
