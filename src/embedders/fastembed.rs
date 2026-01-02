#![cfg(feature = "fastembed")]
use crate::embedders::{Embedder, Input, InputType, ModelInfo, EMBEDDERS};
use anyhow::{bail, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use linkme::distributed_slice;
use std::{cell::RefCell, collections::HashMap, path::PathBuf};

pub static FASTEMBED_EMBEDDER_ID: i32 = 0;
pub static FASTEMBED_EMBEDDER_NAME: &str = "fastembed";

struct ModelRegistration {
    pub info: ModelInfo,
    pub embedding_model: EmbeddingModel,
}

#[distributed_slice]
pub static FASTEMBED_REGISTERED_MODELS: [ModelRegistration] = [..];

thread_local! {
    static FASTEMBED_MODELS: RefCell<HashMap<i32, TextEmbedding>> = RefCell::new(HashMap::new());
}

struct FastEmbedder;

impl FastEmbedder {
    fn lookup_model_registration(model_id: i32) -> Option<&'static ModelRegistration> {
        FASTEMBED_REGISTERED_MODELS
            .iter()
            .find(|reg| reg.info.id() == model_id)
    }
}

impl Embedder for FastEmbedder {
    fn id(&self) -> i32 {
        FASTEMBED_EMBEDDER_ID
    }

    fn name(&self) -> &'static str {
        FASTEMBED_EMBEDDER_NAME
    }

    fn embed(&self, model_id: i32, input: Input) -> Result<(Vec<f32>, usize, usize)> {
        let text_slices = match input {
            Input::Texts(texts) => texts,
            _ => bail!("Unsupported input type"),
        };

        let registration = Self::lookup_model_registration(model_id)
            .ok_or_else(|| anyhow::anyhow!("Invalid model ID: {}", model_id))?;

        FASTEMBED_MODELS.with(|cell| {
            let mut models = cell.borrow_mut();
            let model_instance = models.entry(model_id).or_insert_with(|| {
                TextEmbedding::try_new(
                    InitOptions::new(registration.embedding_model.clone())
                        .with_cache_dir(PathBuf::from("./fastembed_models")),
                )
                .expect("Failed to initialize model")
            });
            model_instance.embed_flat(text_slices, None)
        })
    }

    fn model_info(&self, model_name: &str) -> Option<&ModelInfo> {
        FASTEMBED_REGISTERED_MODELS
            .iter()
            .find(|reg| reg.info.name() == model_name)
            .map(|reg| &reg.info)
    }

    fn supports_input_for_model(&self, model_id: i32, input_type: InputType) -> bool {
        Self::lookup_model_registration(model_id)
            .map(|reg| reg.info.supports_input_type(input_type))
            .unwrap_or(false)
    }
}

#[linkme::distributed_slice(EMBEDDERS)]
static FASTEMBED: &dyn Embedder = &FastEmbedder;

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
