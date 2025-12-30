#![cfg(feature = "grpc")]
use crate::embedders::grpc::tei::v1::embed_client::EmbedClient;
use crate::embedders::grpc::tei::v1::EmbedBatchRequest;
use crate::embedders::grpc::tei::v1::EmbedMultimodalRequest;
use crate::embedders::{Embedder, Input, InputType, ModelInfo, EMBEDDERS};
use anyhow::{anyhow, Result};
use std::sync::LazyLock;
use std::time::Duration;
use tokio::runtime::Runtime;
use tonic::transport::{Channel, Endpoint};

pub mod tei {
    pub mod v1 {
        tonic::include_proto!("tei.v1");
    }
}

pub static EMBED_METHOD_GRPC_ID: i32 = 1;
pub static EMBED_METHOD_GRPC_NAME: &str = "grpc";

static RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("Failed to build Tokio runtime")
});

static ENDPOINT: LazyLock<Endpoint> = LazyLock::new(|| {
    let url = std::env::var("GRPC_EMBEDDER_ENDPOINT")
        .unwrap_or_else(|_| "http://127.0.0.1:50051".to_string());

    Channel::from_shared(url)
        .expect("Invalid gRPC URL")
        .http2_keep_alive_interval(Duration::from_secs(75))
        .keep_alive_timeout(Duration::from_secs(20))
        .connect_timeout(Duration::from_secs(5))
        .tcp_nodelay(true)
        .http2_adaptive_window(true)
});

thread_local! {
    static CLIENT: std::cell::RefCell<Option<EmbedClient<Channel>>> = std::cell::RefCell::new(None);
}

struct GrpcEmbedder;

impl GrpcEmbedder {
    const MODELS: &'static [ModelInfo] = &[
        ModelInfo::new(
            0,
            "sentence-transformers/all-MiniLM-L6-v2",
            &[InputType::Text],
        ),
        ModelInfo::new(
            1,
            "sentence-transformers/bge-large-en-v1.5",
            &[InputType::Text],
        ),
        ModelInfo::new(
            2,
            "ViT-B-32",
            &[InputType::Text, InputType::Image, InputType::Multimodal],
        ),
    ];

    fn get_grpc_client() -> Result<EmbedClient<Channel>> {
        CLIENT.with(|cell| {
            let mut client_opt = cell.borrow_mut();
            if client_opt.is_none() {
                let channel = RUNTIME.block_on(ENDPOINT.connect())?;
                *client_opt = Some(EmbedClient::new(channel));
            }
            Ok(client_opt.as_ref().unwrap().clone())
        })
    }
}

impl Embedder for GrpcEmbedder {
    fn id(&self) -> i32 {
        EMBED_METHOD_GRPC_ID
    }

    fn name(&self) -> &'static str {
        EMBED_METHOD_GRPC_NAME
    }

    fn embed(&self, model_id: i32, input: Input) -> Result<(Vec<f32>, usize, usize)> {
        let model = Self::MODELS
            .iter()
            .find(|m| m.id() == model_id)
            .ok_or_else(|| anyhow!("Unknown model ID: {}", model_id))?;

        match input {
            Input::Texts(texts) => embed_texts(texts, model),
            Input::Image(image) => embed_image(image, model),
            Input::Multimodal { image, texts } => embed_multimodal(image, texts, model),
        }
    }

    fn model_info(&self, model_name: &str) -> Option<&ModelInfo> {
        Self::MODELS.iter().find(|m| m.name() == model_name)
    }

    fn supports_input_for_model(&self, model_id: i32, input_type: InputType) -> bool {
        Self::MODELS
            .iter()
            .find(|m| m.id() == model_id)
            .map(|m| m.supports_input_type(input_type))
            .unwrap_or(false)
    }
}

fn embed_texts(texts: Vec<&str>, model_info: &ModelInfo) -> Result<(Vec<f32>, usize, usize)> {
    let mut client = GrpcEmbedder::get_grpc_client()?;

    let response = RUNTIME.block_on(async {
        let request = EmbedBatchRequest {
            inputs: texts.iter().map(|&s| s.to_string()).collect(),
            truncate: true,
            normalize: true,
            truncation_direction: 0,
            prompt_name: None,
            dimensions: None,
            model: model_info.name().to_string(),
        };
        client.embed_batch(tonic::Request::new(request)).await
    })?;

    let embeddings: Vec<Vec<f32>> = response
        .into_inner()
        .embeddings
        .into_iter()
        .map(|e| e.values)
        .collect();

    let n_vectors = embeddings.len();
    let dim = embeddings.first().map(|e| e.len()).unwrap_or(0);
    let total = n_vectors * dim;
    let mut flat: Vec<f32> = Vec::with_capacity(total);
    for e in embeddings {
        flat.extend_from_slice(&e);
    }

    Ok((flat, n_vectors, dim))
}

fn embed_image(image: &[u8], model_info: &ModelInfo) -> Result<(Vec<f32>, usize, usize)> {
    let mut client = GrpcEmbedder::get_grpc_client()?;

    let response = RUNTIME.block_on(async {
        let request = EmbedMultimodalRequest {
            model: Some(model_info.name().to_string()),
            image_bytes: Some(image.to_vec()),
            text_inputs: vec![],
        };
        client.embed_multimodal(tonic::Request::new(request)).await
    })?;

    let embeddings = response.into_inner().embeddings;
    let first = embeddings
        .first()
        .ok_or_else(|| anyhow!("No embedding returned"))?;

    let dim = first.values.len();
    let embedding = first.values.clone();

    Ok((embedding, 1, dim))
}

fn embed_multimodal(
    image: Option<&[u8]>,
    texts: Vec<&str>,
    model_info: &ModelInfo,
) -> Result<(Vec<f32>, usize, usize)> {
    let mut client = GrpcEmbedder::get_grpc_client()?;

    let response = RUNTIME.block_on(async {
        let request = EmbedMultimodalRequest {
            model: Some(model_info.name().to_string()),
            image_bytes: image.map(|b| b.to_vec()),
            text_inputs: texts.iter().map(|&s| s.to_string()).collect(),
        };
        client.embed_multimodal(tonic::Request::new(request)).await
    })?;

    let embeddings: Vec<Vec<f32>> = response
        .into_inner()
        .embeddings
        .into_iter()
        .map(|e| e.values)
        .collect();

    let n_vectors = embeddings.len();
    let dim = embeddings.first().map(|e| e.len()).unwrap_or(0);
    let total = n_vectors * dim;
    let mut flat: Vec<f32> = Vec::with_capacity(total);
    for e in embeddings {
        flat.extend_from_slice(&e);
    }

    Ok((flat, n_vectors, dim))
}

#[linkme::distributed_slice(EMBEDDERS)]
static GRPC: &dyn Embedder = &GrpcEmbedder;
