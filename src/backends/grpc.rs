#![cfg(feature = "grpc")]
use crate::backends::grpc::tei::v1::embed_client::EmbedClient;
use crate::backends::grpc::tei::v1::EmbedBatchRequest;
use crate::backends::grpc::tei::v1::EmbedMultimodalRequest;
use crate::backends::{Backend, Input, InputType, ModelInfo, BACKENDS};
use crate::utils::flatten_vectors;
use anyhow::{bail, Result};
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

pub static GRPC_BACKEND_ID: i32 = 1;
pub static GRPC_BACKEND_NAME: &str = "grpc";

#[distributed_slice]
pub static GRPC_REGISTERED_MODELS: [ModelInfo] = [..];

static RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("Failed to build Tokio runtime")
});

static ENDPOINT: LazyLock<Endpoint> = LazyLock::new(|| {
    let url = std::env::var("GRPC_BACKEND_ENDPOINT")
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
    static CLIENT: std::cell::RefCell<Option<EmbedClient<Channel>>> = const { std::cell::RefCell::new(None) };
}

struct GrpcBackend;

impl GrpcBackend {
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

impl Backend for GrpcBackend {
    fn id(&self) -> i32 {
        GRPC_BACKEND_ID
    }

    fn name(&self) -> &'static str {
        GRPC_BACKEND_NAME
    }

    fn embed(&self, model_id: i32, input: Input) -> Result<(Vec<f32>, usize, usize)> {
        let model_name = self.resolve_model_name(model_id)?;

        match input {
            Input::Texts(texts) => embed_texts(texts, &model_name),
            Input::Images(images) => embed_images(images, &model_name),
            Input::Multimodal { images, texts } => embed_multimodal(images, texts, &model_name),
            _ => bail!("Unsupported input type"),
        }
    }

    fn model_info(&self, model_name: &str) -> Option<&ModelInfo> {
        GRPC_REGISTERED_MODELS
            .iter()
            .find(|m| m.name() == model_name)
    }

    fn model_info_by_id(&self, model_id: i32) -> Option<&ModelInfo> {
        GRPC_REGISTERED_MODELS.iter().find(|m| m.id() == model_id)
    }
}

fn embed_texts(texts: Vec<&str>, model_name: &str) -> Result<(Vec<f32>, usize, usize)> {
    let mut client = GrpcBackend::grpc_client()?;

    let response = RUNTIME.block_on(async {
        let request = EmbedBatchRequest {
            inputs: texts.iter().map(|&s| s.to_string()).collect(),
            truncate: true,
            normalize: true,
            truncation_direction: 0,
            prompt_name: None,
            dimensions: None,
            model: model_name.to_string(),
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

fn embed_images(images: Vec<&[u8]>, model_name: &str) -> Result<(Vec<f32>, usize, usize)> {
    let mut client = GrpcBackend::grpc_client()?;

    let response = RUNTIME.block_on(async {
        let request = EmbedMultimodalRequest {
            model: Some(model_name.to_string()),
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
    model_name: &str,
) -> Result<(Vec<f32>, usize, usize)> {
    let mut client = GrpcBackend::grpc_client()?;

    let response = RUNTIME.block_on(async {
        let request = EmbedMultimodalRequest {
            model: Some(model_name.to_string()),
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

#[linkme::distributed_slice(BACKENDS)]
static GRPC: &dyn Backend = &GrpcBackend;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::{BackendRegistry, InputType};

    #[test]
    fn grpc_backend_is_registered_in_global_slice() {
        // Act
        let id = BackendRegistry::lookup_backend_id(GRPC_BACKEND_NAME);

        // Assert
        assert_eq!(id, Some(GRPC_BACKEND_ID));
    }

    #[test]
    fn grpc_lookup_backend_by_id_succeeds() {
        // Act
        let result = BackendRegistry::lookup_backend(GRPC_BACKEND_ID);

        // Assert
        assert!(result.is_some());
    }

    #[test]
    fn all_registered_models_have_unique_ids() {
        // Act
        let ids: Vec<i32> = GRPC_REGISTERED_MODELS.iter().map(|m| m.id()).collect();

        // Assert
        let unique: std::collections::HashSet<i32> = ids.iter().copied().collect();
        assert_eq!(ids.len(), unique.len());
    }

    #[test]
    fn all_registered_models_have_unique_names() {
        // Act
        let names: Vec<&str> = GRPC_REGISTERED_MODELS.iter().map(|m| m.name()).collect();

        // Assert
        let unique: std::collections::HashSet<&str> = names.iter().copied().collect();
        assert_eq!(names.len(), unique.len());
    }

    #[test]
    fn clip_vit_base_supports_text_image_and_multimodal() {
        // Act
        let clip = GRPC_REGISTERED_MODELS
            .iter()
            .find(|m| m.name().contains("ViT-B-32"))
            .expect("CLIP ViT-B-32 not found");

        // Assert
        assert!(clip.supports_input_type(InputType::Text));
        assert!(clip.supports_input_type(InputType::Image));
        assert!(clip.supports_input_type(InputType::Multimodal));
    }

    #[test]
    fn text_only_models_do_not_support_image() {
        // Assert
        for m in GRPC_REGISTERED_MODELS.iter().filter(|m| {
            m.supports_input_type(InputType::Text)
                && !m.name().contains("ViT")
                && !m.name().contains("clip")
                && !m.name().contains("siglip")
        }) {
            assert!(
                !m.supports_input_type(InputType::Image),
                "model {} unexpectedly supports Image",
                m.name()
            );
        }
    }

    #[cfg(not(feature = "dynamic_model_loading"))]
    #[test]
    fn resolve_model_name_succeeds_for_all_registered_models() {
        // Arrange
        let backend = BackendRegistry::lookup_backend(GRPC_BACKEND_ID).unwrap();

        // Assert
        for m in GRPC_REGISTERED_MODELS.iter() {
            assert!(
                backend.resolve_model_name(m.id()).is_ok(),
                "resolve_model_name failed for id {}",
                m.id()
            );
        }
    }

    #[cfg(not(feature = "dynamic_model_loading"))]
    #[test]
    fn resolve_model_name_fails_for_unknown_id() {
        // Arrange
        let backend = BackendRegistry::lookup_backend(GRPC_BACKEND_ID).unwrap();

        // Act
        let result = backend.resolve_model_name(i32::MAX);

        // Assert
        assert!(result.is_err());
    }
}
