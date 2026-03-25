#![cfg(feature = "http")]
use crate::backends::{Backend, Input, InputType, ModelInfo, BACKENDS};
use crate::utils::flatten_vectors;
use anyhow::{bail, Result};
use linkme::distributed_slice;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;
use std::time::Duration;

pub static HTTP_BACKEND_ID: i32 = 3;
pub static HTTP_BACKEND_NAME: &str = "http";

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
    std::env::var("HTTP_BACKEND_ENDPOINT")
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

struct HttpBackend;

impl HttpBackend {}

impl Backend for HttpBackend {
    fn id(&self) -> i32 {
        HTTP_BACKEND_ID
    }

    fn name(&self) -> &'static str {
        HTTP_BACKEND_NAME
    }

    fn embed(&self, model_id: i32, input: Input) -> Result<(Vec<f32>, usize, usize)> {
        let model_name = self.resolve_model_name(model_id)?;

        match input {
            Input::Texts(texts) => embed_texts(texts, &model_name),
            _ => bail!("HTTP backend only supports text input"),
        }
    }

    fn model_info(&self, model_name: &str) -> Option<&ModelInfo> {
        HTTP_REGISTERED_MODELS
            .iter()
            .find(|m| m.name() == model_name)
    }

    fn model_info_by_id(&self, model_id: i32) -> Option<&ModelInfo> {
        HTTP_REGISTERED_MODELS.iter().find(|m| m.id() == model_id)
    }

    fn supports_input_for_model(&self, model_id: i32, input_type: InputType) -> bool {
        input_type == InputType::Text && self.resolve_model_name(model_id).is_ok()
    }
}

fn embed_texts(texts: Vec<&str>, model_name: &str) -> Result<(Vec<f32>, usize, usize)> {
    let request_body = EmbeddingRequest {
        model: model_name,
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

#[linkme::distributed_slice(BACKENDS)]
static HTTP: &dyn Backend = &HttpBackend;

#[distributed_slice(HTTP_REGISTERED_MODELS)]
static ALL_MINI_LM_L6_V2: ModelInfo = ModelInfo::new(
    0,
    "sentence-transformers/all-MiniLM-L6-v2",
    &[InputType::Text],
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::{BackendRegistry, InputType};

    #[test]
    fn http_backend_is_registered() {
        // Act
        let id = BackendRegistry::lookup_backend_id(HTTP_BACKEND_NAME);

        // Assert
        assert_eq!(id, Some(HTTP_BACKEND_ID));
    }

    #[test]
    fn http_backend_lookup_by_id_succeeds() {
        // Act
        let result = BackendRegistry::lookup_backend(HTTP_BACKEND_ID);

        // Assert
        assert!(result.is_some());
    }

    #[test]
    fn all_http_models_have_unique_ids() {
        // Act
        let ids: Vec<i32> = HTTP_REGISTERED_MODELS.iter().map(|m| m.id()).collect();

        // Assert
        let unique: std::collections::HashSet<i32> = ids.iter().copied().collect();
        assert_eq!(ids.len(), unique.len());
    }

    #[test]
    fn all_http_models_support_text_input() {
        // Assert
        for m in HTTP_REGISTERED_MODELS.iter() {
            assert!(
                m.supports_input_type(InputType::Text),
                "HTTP model {} must support Text",
                m.name()
            );
        }
    }

    #[test]
    fn http_backend_supports_text_for_all_registered_models() {
        // Arrange
        let backend = BackendRegistry::lookup_backend(HTTP_BACKEND_ID).unwrap();

        // Assert
        for m in HTTP_REGISTERED_MODELS.iter() {
            assert!(
                backend.supports_input_for_model(m.id(), InputType::Text),
                "supports_input_for_model returned false for {} / Text",
                m.name()
            );
        }
    }

    #[test]
    fn http_backend_does_not_support_image_for_any_model() {
        // Arrange
        let backend = BackendRegistry::lookup_backend(HTTP_BACKEND_ID).unwrap();

        // Assert
        for m in HTTP_REGISTERED_MODELS.iter() {
            assert!(
                !backend.supports_input_for_model(m.id(), InputType::Image),
                "HTTP backend should not support Image for {}",
                m.name()
            );
        }
    }

    #[test]
    fn http_backend_does_not_support_multimodal() {
        // Arrange
        let backend = BackendRegistry::lookup_backend(HTTP_BACKEND_ID).unwrap();

        // Assert
        for m in HTTP_REGISTERED_MODELS.iter() {
            assert!(
                !backend.supports_input_for_model(m.id(), InputType::Multimodal),
                "HTTP backend should not support Multimodal for {}",
                m.name()
            );
        }
    }

    #[test]
    fn http_embed_returns_error_for_image_input() {
        // Arrange
        let backend = BackendRegistry::lookup_backend(HTTP_BACKEND_ID).unwrap();
        let dummy_img: &[u8] = &[0xFF, 0xD8, 0xFF];

        // Act
        let result = backend.embed(
            HTTP_REGISTERED_MODELS[0].id(),
            Input::Images(vec![dummy_img]),
        );

        // Assert
        assert!(
            result.is_err(),
            "expected error for Image input on HTTP backend"
        );
    }
}
