#![cfg(feature = "fastembed")]
use crate::embedders::{EMBEDDERS, Embedder, Input, InputType, ModelInfo};
use anyhow::Result;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::str::FromStr;
use std::{cell::RefCell, collections::HashMap, path::PathBuf};

pub static EMBED_METHOD_FASTEMBED_ID: i32 = 0;
pub static EMBED_METHOD_FASTEMBED_NAME: &str = "fastembed";

thread_local! {
    static FASTEMBED_MODELS: RefCell<HashMap<i32, TextEmbedding>> = RefCell::new(HashMap::new());
}

struct FastEmbedder;

struct ModelDef {
    model: &'static ModelInfo,
    embedding_model: EmbeddingModel,
}

impl FastEmbedder {
    const MODELS: &'static [ModelInfo] = &[
        ModelInfo::new(0, "AllMiniLML6V2", &[InputType::Text]),
        ModelInfo::new(1, "BGELargeENV15", &[InputType::Text]),
    ];

    fn get_model_def(model_id: i32) -> Option<ModelDef> {
        match model_id {
            0 => Some(ModelDef {
                model: &Self::MODELS[0],
                embedding_model: EmbeddingModel::AllMiniLML6V2,
            }),
            1 => Some(ModelDef {
                model: &Self::MODELS[1],
                embedding_model: EmbeddingModel::BGELargeENV15,
            }),
            _ => None,
        }
    }
}

impl Embedder for FastEmbedder {
    fn method_id(&self) -> i32 {
        EMBED_METHOD_FASTEMBED_ID
    }

    fn method_name(&self) -> &'static str {
        EMBED_METHOD_FASTEMBED_NAME
    }

    fn embed(&self, model_id: i32, input: Input) -> Result<(Vec<f32>, usize, usize)> {
        let text_slices = match input {
            Input::Texts(texts) => texts,
        };

        let model_def = Self::get_model_def(model_id)
            .ok_or_else(|| anyhow::anyhow!("Invalid model ID: {}", model_id))?;

        FASTEMBED_MODELS.with(|cell| {
            let mut models = cell.borrow_mut();
            let model_instance = models.entry(model_id).or_insert_with(|| {
                TextEmbedding::try_new(
                    InitOptions::new(model_def.embedding_model)
                        .with_cache_dir(PathBuf::from("./fastembed_models")),
                )
                .expect("Failed to initialize model")
            });
            model_instance.embed_flat(text_slices, None)
        })
    }

    fn get_model(&self, model_name: &str) -> Option<&ModelInfo> {
        let parsed = EmbeddingModel::from_str(model_name).ok()?;

        for model_def in [Self::get_model_def(0), Self::get_model_def(1)]
            .into_iter()
            .flatten()
        {
            if model_def.embedding_model == parsed {
                return Some(model_def.model);
            }
        }
        None
    }

    fn supports_model_id(&self, model_id: i32, input_type: InputType) -> bool {
        Self::MODELS
            .iter()
            .find(|m| m.id == model_id)
            .map(|m| m.supports_input_type(input_type))
            .unwrap_or(false)
    }
}

#[linkme::distributed_slice(EMBEDDERS)]
static FASTEMBED: &dyn Embedder = &FastEmbedder;
