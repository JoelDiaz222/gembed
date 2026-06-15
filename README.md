# Gembed: A High-Performance, Portable, Extensible, Rust-Powered Embedding Library

The portable Rust core of the **Gembed** architecture for **in-database embedding generation**.

This library is compiled as a `staticlib` and exposes a stable **C Foreign Function Interface (FFI)**, allowing it to
be linked into any DBMS adapter regardless of the host language or runtime.

## C FFI

The library exposes four functions:

| Function | Description |
|---|---|
| `validate_backend(name)` | Resolves a backend name to an integer ID, `-1` if non-existent |
| `validate_model(backend_id, model, input_type)` | Resolves a model name to an integer ID, `-1` if non-existent |
| `generate_embeddings(backend_id, model_id, input, out_batch)` | Runs inference and writes a flat `float32` buffer to `out_batch` |
| `free_embedding_batch(batch)` | Frees memory allocated by `generate_embeddings` |

Callers resolve backend and model names to IDs once and cache them, which avoids additional FFI round-trips.

## Embedding Backends

Backends are registered at link time using the `linkme` distributed-slice pattern and implement a common `Backend`
trait.

| Feature flag | Backend | Notes                        |
|---|---|------------------------------|
| `embed_anything` *(default)* | embed_anything | Candle-based local inference |
| `fastembed` | FastEmbed | ONNX-based local inference   |
| `ort` | ONNX Runtime | Direct ORT integration       |
| `grpc` | gRPC | Remote inference via gRPC    |
| `http` | HTTP | Remote inference via HTTP    |

### Adding a Backend

To add a new backend:

1. Add its dependencies to `Cargo.toml`, optionally behind a new feature flag.
2. Create its source code under `src/backends/`, implementing the common `Backend` trait.
3. Register it at link time using the `linkme` distributed-slice pattern, so it is automatically discovered by the Core.
4. Recompile, enabling the corresponding feature flag if one was added (e.g., `cargo build --release --features "new_backend"`).

## Build

```bash
# Default (embed_anything backend)
cargo build --release

# With multiple backends
cargo build --release --features "fastembed,grpc"
```

The compiled `libgembed.a` (or `libgembed_jni.so` for JNI) is then linked into the DBMS adapter.

## Embedding Servers

If using the `grpc` or the `http` backends, a compatible server must be running. See
[`server/README.md`](server/README.md) for setup instructions of a Python gRPC server example using `asyncio`.
The [gembed-sandbox](https://github.com/JoelDiaz222/gembed-sandbox) repository contains a similar compatible gRPC
server and a FastAPI HTTP server.

## License

This project is dual-licensed under the Apache License, Version 2.0 or the GNU General Public License, Version 2.0.

- [Apache License, Version 2.0](LICENSE-APACHE)
- [GNU General Public License, Version 2.0](LICENSE-GPL)

You may choose which license you wish to use when adopting this software.
