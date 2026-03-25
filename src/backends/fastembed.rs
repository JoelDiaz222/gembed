#![cfg(feature = "fastembed")]
use crate::backends::{Backend, Input, InputType, ModelInfo, BACKENDS};
use crate::utils::flatten_vectors;
use anyhow::{bail, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use linkme::distributed_slice;
use std::{cell::RefCell, collections::HashMap, path::PathBuf};

pub static FASTEMBED_BACKEND_ID: i32 = 0;
pub static FASTEMBED_BACKEND_NAME: &str = "fastembed";

struct ModelRegistration {
    pub info: ModelInfo,
    #[allow(dead_code)]
    pub embedding_model: EmbeddingModel,
}

#[distributed_slice]
pub static FASTEMBED_REGISTERED_MODELS: [ModelRegistration] = [..];

thread_local! {
    static FASTEMBED_MODELS: RefCell<HashMap<i32, TextEmbedding>> = RefCell::new(HashMap::new());
}

struct FastEmbedBackend;

impl FastEmbedBackend {
    fn lookup_model_registration(model_id: i32) -> Option<&'static ModelRegistration> {
        FASTEMBED_REGISTERED_MODELS
            .iter()
            .find(|reg| reg.info.id() == model_id)
    }
}

impl Backend for FastEmbedBackend {
    fn id(&self) -> i32 {
        FASTEMBED_BACKEND_ID
    }

    fn name(&self) -> &'static str {
        FASTEMBED_BACKEND_NAME
    }

    #[cfg(not(feature = "dynamic_model_loading"))]
    fn embed(&self, model_id: i32, input: Input) -> Result<(Vec<f32>, usize, usize)> {
        let text_slices = match input {
            Input::Texts(texts) => texts,
            _ => bail!("Unsupported input type"),
        };

        let model_name = self.resolve_model_name(model_id)?;
        let registration = FASTEMBED_REGISTERED_MODELS
            .iter()
            .find(|reg| reg.info.name() == model_name)
            .ok_or_else(|| anyhow::anyhow!("Invalid model: {}", model_name))?;

        FASTEMBED_MODELS.with(|cell| {
            let mut models = cell.borrow_mut();
            let model_instance = models.entry(model_id).or_insert_with(|| {
                TextEmbedding::try_new(
                    InitOptions::new(registration.embedding_model.clone())
                        .with_cache_dir(PathBuf::from("./fastembed_models")),
                )
                .expect("Failed to initialize model")
            });
            let embeddings = model_instance.embed(text_slices, None)?;
            flatten_vectors(embeddings)
        })
    }

    #[cfg(feature = "dynamic_model_loading")]
    fn embed(&self, model_id: i32, input: Input) -> Result<(Vec<f32>, usize, usize)> {
        let text_slices = match input {
            Input::Texts(texts) => texts,
            _ => bail!("Unsupported input type"),
        };

        let model_name = self.resolve_model_name(model_id)?;
        let model_enum = model_name
            .parse::<EmbeddingModel>()
            .map_err(|_| anyhow::anyhow!("Failed to parse model name: {}", model_name))?;

        FASTEMBED_MODELS.with(|cell| {
            let mut models = cell.borrow_mut();
            let model_instance = models.entry(model_id).or_insert_with(|| {
                TextEmbedding::try_new(
                    InitOptions::new(model_enum)
                        .with_cache_dir(PathBuf::from("./fastembed_models")),
                )
                .expect("Failed to initialize model")
            });
            let embeddings = model_instance.embed(text_slices, None)?;
            crate::utils::flatten_vectors(embeddings)
        })
    }

    fn model_info(&self, model_name: &str) -> Option<&ModelInfo> {
        FASTEMBED_REGISTERED_MODELS
            .iter()
            .find(|reg| reg.info.name() == model_name)
            .map(|reg| &reg.info)
    }

    fn model_info_by_id(&self, model_id: i32) -> Option<&ModelInfo> {
        Self::lookup_model_registration(model_id).map(|reg| &reg.info)
    }

    fn supports_input_for_model(&self, model_id: i32, input_type: InputType) -> bool {
        input_type == InputType::Text && self.resolve_model_name(model_id).is_ok()
    }
}

#[linkme::distributed_slice(BACKENDS)]
static FASTEMBED: &dyn Backend = &FastEmbedBackend;

#[distributed_slice(FASTEMBED_REGISTERED_MODELS)]
static ALL_MINI_LM_L6_V2: ModelRegistration = ModelRegistration {
    info: ModelInfo::new(0, "Qdrant/all-MiniLM-L6-v2-onnx", &[InputType::Text]),
    embedding_model: EmbeddingModel::AllMiniLML6V2,
};

#[distributed_slice(FASTEMBED_REGISTERED_MODELS)]
static BGE_LARGE_EN_V1_5: ModelRegistration = ModelRegistration {
    info: ModelInfo::new(1, "Xenova/bge-large-en-v1.5", &[InputType::Text]),
    embedding_model: EmbeddingModel::BGELargeENV15,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::{BackendRegistry, InputType};

    #[test]
    fn fastembed_backend_is_registered() {
        // Act
        let id = BackendRegistry::lookup_backend_id(FASTEMBED_BACKEND_NAME);

        // Assert
        assert_eq!(id, Some(FASTEMBED_BACKEND_ID));
    }

    #[test]
    fn fastembed_lookup_backend_by_id_succeeds() {
        // Act
        let b = BackendRegistry::lookup_backend(FASTEMBED_BACKEND_ID);

        // Assert
        assert!(b.is_some());
    }

    #[test]
    fn all_registered_models_have_unique_ids() {
        // Act
        let ids: Vec<i32> = FASTEMBED_REGISTERED_MODELS
            .iter()
            .map(|r| r.info.id())
            .collect();

        // Assert
        let unique: std::collections::HashSet<i32> = ids.iter().copied().collect();
        assert_eq!(
            ids.len(),
            unique.len(),
            "duplicate model IDs in FASTEMBED_REGISTERED_MODELS"
        );
    }

    #[test]
    fn all_registered_models_have_unique_names() {
        // Act
        let names: Vec<&str> = FASTEMBED_REGISTERED_MODELS
            .iter()
            .map(|r| r.info.name())
            .collect();

        // Assert
        let unique: std::collections::HashSet<&str> = names.iter().copied().collect();
        assert_eq!(
            names.len(),
            unique.len(),
            "duplicate model names in FASTEMBED_REGISTERED_MODELS"
        );
    }

    #[test]
    fn all_registered_models_support_text_input() {
        // Assert
        for reg in FASTEMBED_REGISTERED_MODELS.iter() {
            assert!(
                reg.info.supports_input_type(InputType::Text),
                "model {} must support Text input",
                reg.info.name()
            );
        }
    }

    #[test]
    fn no_registered_model_supports_image_input() {
        // Assert
        for reg in FASTEMBED_REGISTERED_MODELS.iter() {
            assert!(
                !reg.info.supports_input_type(InputType::Image),
                "model {} should NOT support Image input",
                reg.info.name()
            );
        }
    }

    #[cfg(not(feature = "dynamic_model_loading"))]
    #[test]
    fn model_info_lookup_by_name_succeeds_for_all_models() {
        // Arrange
        let backend = BackendRegistry::lookup_backend(FASTEMBED_BACKEND_ID).unwrap();

        // Assert
        for reg in FASTEMBED_REGISTERED_MODELS.iter() {
            assert!(
                backend.model_info(reg.info.name()).is_some(),
                "model_info returned None for '{}'",
                reg.info.name()
            );
        }
    }

    #[cfg(not(feature = "dynamic_model_loading"))]
    #[test]
    fn model_info_lookup_by_id_succeeds_for_all_models() {
        // Arrange
        let backend = BackendRegistry::lookup_backend(FASTEMBED_BACKEND_ID).unwrap();

        // Assert
        for reg in FASTEMBED_REGISTERED_MODELS.iter() {
            assert!(
                backend.model_info_by_id(reg.info.id()).is_some(),
                "model_info_by_id returned None for id {}",
                reg.info.id()
            );
        }
    }

    #[test]
    fn supports_input_for_model_returns_true_for_text() {
        // Arrange
        let backend = BackendRegistry::lookup_backend(FASTEMBED_BACKEND_ID).unwrap();

        // Assert
        for reg in FASTEMBED_REGISTERED_MODELS.iter() {
            assert!(
                backend.supports_input_for_model(reg.info.id(), InputType::Text),
                "model {} should support Text",
                reg.info.name()
            );
        }
    }

    #[test]
    fn supports_input_for_model_returns_false_for_image() {
        // Arrange
        let backend = BackendRegistry::lookup_backend(FASTEMBED_BACKEND_ID).unwrap();

        // Assert
        for reg in FASTEMBED_REGISTERED_MODELS.iter() {
            assert!(
                !backend.supports_input_for_model(reg.info.id(), InputType::Image),
                "model {} should NOT support Image",
                reg.info.name()
            );
        }
    }
}
