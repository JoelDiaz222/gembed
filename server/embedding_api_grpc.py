import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import grpc
from sentence_transformers import SentenceTransformer

import tei_pb2 as pb2
import tei_pb2_grpc as pb2_grpc

model_cache = {}
executor = ThreadPoolExecutor(max_workers=4)

def get_model(model_name: str):
    if model_name not in model_cache:
        print(f"Loading model: {model_name}")
        model_cache[model_name] = SentenceTransformer(model_name)
    return model_cache[model_name]

async def embed_texts_async(texts, model_name):
    loop = asyncio.get_running_loop()
    model = get_model(model_name)
    encode_func = partial(model.encode, texts, show_progress_bar=False, device="cpu")
    return await loop.run_in_executor(executor, encode_func)

class EmbedService(pb2_grpc.EmbedServicer):
    async def Embed(self, request, context):
        text = request.inputs
        model_name = request.model or "all-MiniLM-L6-v2"
        embeddings_array = await embed_texts_async([text], model_name)
        response = pb2.EmbedResponse()
        response.embeddings.extend(embeddings_array[0].tolist())
        return response

    async def EmbedBatch(self, request, context):
        model_name = request.model or "all-MiniLM-L6-v2"
        embeddings_array = await embed_texts_async(request.inputs, model_name)
        response = pb2.EmbedBatchResponse()

        for vector in embeddings_array:
            embedding_msg = pb2.Embedding()
            embedding_msg.values.extend(vector.tolist())
            response.embeddings.append(embedding_msg)

        return response

async def serve():
    server = grpc.aio.server()
    pb2_grpc.add_EmbedServicer_to_server(EmbedService(), server)
    port = 50051
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    print(f"gRPC server running on port {port}")
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())
