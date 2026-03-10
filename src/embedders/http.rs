#![cfg(feature = "http")]
use crate::embedders::{Embedder, Input, InputType, ModelInfo, EMBEDDERS};
use crate::utils::flatten_vectors;
use anyhow::{anyhow, bail, Result};
use linkme::distributed_slice;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;
use std::time::Duration;

pub static HTTP_EMBEDDER_ID: i32 = 3;
pub static HTTP_EMBEDDER_NAME: &str = "http";

#[distributed_slice]
pub static HTTP_REGISTERED_MODELS: [ModelInfo] = [..];

static HTTP_CLIENT: LazyLock<Client> = LazyLock::new(|| {
    Client::builder()
        .timeout(Duration::from_secs(120))
        .tcp_nodelay(true)
        .build()
        .expect("Failed to build HTTP client")
});

static BASE_URL: LazyLock<String> = LazyLock::new(|| {
    std::env::var("HTTP_EMBEDDER_ENDPOINT")
        .unwrap_or_else(|_| "http://127.0.0.1:8080/v1/embed".to_string())
});

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

struct HttpEmbedder;

impl HttpEmbedder {
    fn lookup_model_registration(model_id: i32) -> Option<&'static ModelInfo> {
        HTTP_REGISTERED_MODELS.iter().find(|m| m.id() == model_id)
    }
}

impl Embedder for HttpEmbedder {
    fn id(&self) -> i32 {
        HTTP_EMBEDDER_ID
    }

    fn name(&self) -> &'static str {
        HTTP_EMBEDDER_NAME
    }

    fn embed(&self, model_id: i32, input: Input) -> Result<(Vec<f32>, usize, usize)> {
        let model_info = Self::lookup_model_registration(model_id)
            .ok_or_else(|| anyhow!("Unknown model ID: {}", model_id))?;

        match input {
            Input::Texts(texts) => embed_texts(texts, model_info),
            _ => bail!("HTTP embedder only supports text input"),
        }
    }

    fn model_info(&self, model_name: &str) -> Option<&ModelInfo> {
        HTTP_REGISTERED_MODELS
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
    let request_body = EmbeddingRequest {
        model: model_info.name(),
        input: texts,
    };

    let response = HTTP_CLIENT
        .post(BASE_URL.as_str())
        .json(&request_body)
        .send()?
        .error_for_status()?;

    let parsed: EmbeddingResponse = response.json()?;

    let vectors: Vec<Vec<f32>> = parsed.data.into_iter().map(|d| d.embedding).collect();
    flatten_vectors(vectors)
}

#[linkme::distributed_slice(EMBEDDERS)]
static HTTP: &dyn Embedder = &HttpEmbedder;

#[distributed_slice(HTTP_REGISTERED_MODELS)]
static ALL_MINI_LM_L6_V2: ModelInfo = ModelInfo::new(
    0,
    "sentence-transformers/all-MiniLM-L6-v2",
    &[InputType::Text],
);
