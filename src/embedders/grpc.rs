#![cfg(feature = "grpc")]
use crate::embedders::grpc::tei::v1::embed_client::EmbedClient;
use crate::embedders::grpc::tei::v1::EmbedBatchRequest;
use crate::embedders::grpc::tei::v1::EmbedMultimodalRequest;
use crate::embedders::{Embedder, Input, InputType, ModelInfo, EMBEDDERS};
use crate::utils::flatten_vectors;
use anyhow::{anyhow, bail, Result};
use linkme::distributed_slice;
use std::sync::LazyLock;
use std::time::Duration;
use tokio::runtime::Runtime;
use tonic::transport::{Channel, Endpoint};

pub mod tei {
    pub mod v1 {
        tonic::include_proto!("tei.v1");
    }
}

pub static GRPC_EMBEDDER_ID: i32 = 1;
pub static GRPC_EMBEDDER_NAME: &str = "grpc";

#[distributed_slice]
pub static GRPC_REGISTERED_MODELS: [ModelInfo] = [..];

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
    fn lookup_model_registration(model_id: i32) -> Option<&'static ModelInfo> {
        GRPC_REGISTERED_MODELS.iter().find(|m| m.id() == model_id)
    }

    fn grpc_client() -> Result<EmbedClient<Channel>> {
        CLIENT.with(|cell| {
            let mut client_opt = cell.borrow_mut();
            if client_opt.is_none() {
                let channel = RUNTIME.block_on(ENDPOINT.connect())?;
                *client_opt = Some(
                    EmbedClient::new(channel)
                        .max_decoding_message_size(usize::MAX)
                        .max_encoding_message_size(usize::MAX),
                );
            }
            Ok(client_opt.as_ref().unwrap().clone())
        })
    }
}

impl Embedder for GrpcEmbedder {
    fn id(&self) -> i32 {
        GRPC_EMBEDDER_ID
    }

    fn name(&self) -> &'static str {
        GRPC_EMBEDDER_NAME
    }

    fn embed(&self, model_id: i32, input: Input) -> Result<(Vec<f32>, usize, usize)> {
        let model_info = Self::lookup_model_registration(model_id)
            .ok_or_else(|| anyhow!("Unknown model ID: {}", model_id))?;

        match input {
            Input::Texts(texts) => embed_texts(texts, model_info),
            Input::Images(images) => embed_images(images, model_info),
            Input::Multimodal { images, texts } => embed_multimodal(images, texts, model_info),
            _ => bail!("Unsupported input type"),
        }
    }

    fn model_info(&self, model_name: &str) -> Option<&ModelInfo> {
        GRPC_REGISTERED_MODELS
            .iter()
            .find(|m| m.name() == model_name)
    }

    fn supports_input_for_model(&self, model_id: i32, input_type: InputType) -> bool {
        Self::lookup_model_registration(model_id)
            .map(|m| m.supports_input_type(input_type))
            .unwrap_or(false)
    }
}

fn embed_texts(texts: Vec<&str>, model_info: &ModelInfo) -> Result<(Vec<f32>, usize, usize)> {
    let mut client = GrpcEmbedder::grpc_client()?;

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

    flatten_vectors(embeddings)
}

fn embed_images(images: Vec<&[u8]>, model_info: &ModelInfo) -> Result<(Vec<f32>, usize, usize)> {
    let mut client = GrpcEmbedder::grpc_client()?;

    let response = RUNTIME.block_on(async {
        let request = EmbedMultimodalRequest {
            model: Some(model_info.name().to_string()),
            images: images.iter().map(|&img| img.to_vec()).collect(),
            text_inputs: vec![],
        };
        client.embed_multimodal(tonic::Request::new(request)).await
    })?;

    let embeddings: Vec<Vec<f32>> = response
        .into_inner()
        .embeddings
        .into_iter()
        .map(|e| e.values)
        .collect();

    flatten_vectors(embeddings)
}

fn embed_multimodal(
    images: Vec<&[u8]>,
    texts: Vec<&str>,
    model_info: &ModelInfo,
) -> Result<(Vec<f32>, usize, usize)> {
    let mut client = GrpcEmbedder::grpc_client()?;

    let response = RUNTIME.block_on(async {
        let request = EmbedMultimodalRequest {
            model: Some(model_info.name().to_string()),
            images: images.iter().map(|&b| b.to_vec()).collect(),
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

    flatten_vectors(embeddings)
}

#[linkme::distributed_slice(EMBEDDERS)]
static GRPC: &dyn Embedder = &GrpcEmbedder;

#[distributed_slice(GRPC_REGISTERED_MODELS)]
static ALL_MINI_LM_L6_V2: ModelInfo = ModelInfo::new(
    0,
    "sentence-transformers/all-MiniLM-L6-v2",
    &[InputType::Text],
);

#[distributed_slice(GRPC_REGISTERED_MODELS)]
static BGE_LARGE_EN_V15: ModelInfo =
    ModelInfo::new(1, "BAAI/bge-large-en-v1.5", &[InputType::Text]);

#[distributed_slice(GRPC_REGISTERED_MODELS)]
static CLIP_VIT_BASE_PATCH32: ModelInfo = ModelInfo::new(
    2,
    "ViT-B-32",
    &[InputType::Text, InputType::Image, InputType::Multimodal],
);
