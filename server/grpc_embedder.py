import asyncio
import contextlib
import io
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import grpc
import open_clip
import tei_pb2 as pb2
import tei_pb2_grpc as pb2_grpc
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

model_cache = {}
clip_model_cache = {}
executor = ThreadPoolExecutor(max_workers=4)


def get_model(model_name: str):
    if model_name not in model_cache:
        print(f"Loading model: {model_name}")
        model_cache[model_name] = SentenceTransformer(model_name)
    return model_cache[model_name]


def get_clip_model(model_name: str, pretrained: str):
    """Load and cache CLIP models with their preprocessor and tokenizer."""
    cache_key = f"{model_name}_{pretrained}"
    if cache_key not in clip_model_cache:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        clip_model_cache[cache_key] = {
            "model": model,
            "preprocess": preprocess,
            "tokenizer": tokenizer,
        }
    return clip_model_cache[cache_key]


async def embed_texts_async(texts, model_name):
    loop = asyncio.get_running_loop()
    model = get_model(model_name)
    encode_func = partial(model.encode, texts, show_progress_bar=False, device="cpu")
    return await loop.run_in_executor(executor, encode_func)


def encode_multimodal(images_pil, text_inputs, model_name, pretrained):
    """
    Encode images and/or multiple texts using CLIP.
    Returns list of normalized embeddings (images first, then texts).
    """
    clip_components = get_clip_model(model_name, pretrained)
    model = clip_components["model"]
    preprocess = clip_components["preprocess"]
    tokenizer = clip_components["tokenizer"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    embeddings = []

    # Use autocast only for CUDA to avoid BFloat16 issues on CPU
    context_manager = (
        torch.autocast(device, dtype=torch.float16)
        if device == "cuda"
        else contextlib.nullcontext()
    )

    with torch.no_grad(), context_manager:
        if images_pil:
            # Process images in batch
            image_tensors = [preprocess(img).unsqueeze(0) for img in images_pil]
            image_batch = torch.cat(image_tensors).to(device)
            image_features = model.encode_image(image_batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            for feature in image_features:
                embeddings.append(feature.cpu().numpy())

        if text_inputs:
            text_tensor = tokenizer(list(text_inputs)).to(device)
            text_features = model.encode_text(text_tensor)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            for feature in text_features:
                embeddings.append(feature.cpu().numpy())

    return embeddings


async def embed_multimodal_async(images_pil, text_inputs, model_name, pretrained):
    loop = asyncio.get_running_loop()
    encode_func = partial(
        encode_multimodal, images_pil, text_inputs, model_name, pretrained
    )
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

    async def EmbedMultimodal(self, request, context):
        """
        Handle multimodal embedding requests with optional images and/or multiple text inputs.
        Returns one embedding per input (image embeddings first if present, then text embeddings).
        Uses open_clip for encoding.
        """
        model_name = request.model or "ViT-B-32"

        if "@" in model_name:
            model_name, pretrained = model_name.split("@", 1)
        else:
            pretrained = "laion2b_s34b_b79k"

        if not request.images and not request.text_inputs:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(
                "At least one of images or text_inputs must be provided"
            )
            return pb2.EmbedBatchResponse()

        try:
            images_pil = []
            text_inputs = list(request.text_inputs) if request.text_inputs else []

            if request.images:
                try:
                    for img_bytes in request.images:
                        img = Image.open(io.BytesIO(img_bytes))
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        images_pil.append(img)
                except Exception as e:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details(f"Failed to parse image: {str(e)}")
                    return pb2.EmbedBatchResponse()

            embeddings_list = await embed_multimodal_async(
                images_pil, text_inputs, model_name, pretrained
            )

            response = pb2.EmbedBatchResponse()
            for embedding in embeddings_list:
                embedding_msg = pb2.Embedding()
                embedding_msg.values.extend(embedding.tolist())
                response.embeddings.append(embedding_msg)

            return response

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error generating multimodal embedding: {str(e)}")
            return pb2.EmbedBatchResponse()


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
