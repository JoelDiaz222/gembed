## gRPC Server Setup (Remote Inference)

If you are using the [gRPC embedder](../src/embedders/grpc.rs) for remote inference, you need to run a separate gRPC server to handle the model requests.

### 1. Prerequisites and Running the API

```bash
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3.13 grpc_embedder.py
```

### 2. Protobuf Compilation

Use the following command if you need to manually re-generate the Python gRPC client/server files from the [protocol definition](../proto/tei.proto):

```bash
python3.13 -m grpc_tools.protoc \
  -I../proto \
  --python_out=. \
  --grpc_python_out=. \
  tei.proto
```
