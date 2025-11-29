```
python3.13 -m venv venv
source venv/bin/activate # On Windows, use venv\Scripts\activate
pip install grpcio grpcio-tools sentence-transformers asyncio open_clip_torch
python3.13 embedding_api_grpc.py
```

```
python3.13 -m grpc_tools.protoc \
    -I../proto \
    --python_out=. \
    --grpc_python_out=. \
    tei.proto
```
