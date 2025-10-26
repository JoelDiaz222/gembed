#![cfg(feature = "fastembed")]
use crate::embedders::{EmbedMethod, Embedder, EMBEDDERS};
use anyhow::Result;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::str::FromStr;
use std::{cell::RefCell, collections::HashMap, path::PathBuf};

#[unsafe(no_mangle)]
pub static EMBED_METHOD_FASTEMBED: i32 = EmbedMethod::FastEmbed as i32;

thread_local! {
    static FASTEMBED_MODELS: RefCell<HashMap<i32, TextEmbedding>> = RefCell::new(HashMap::new());
}

struct FastEmbedder;

impl FastEmbedder {
    const MODELS: &'static [(i32, EmbeddingModel)] = &[
        (0, EmbeddingModel::AllMiniLML6V2),
        (1, EmbeddingModel::BGELargeENV15),
    ];

    fn get_embedding_model(model_id: i32) -> Option<EmbeddingModel> {
        Self::MODELS
            .iter()
            .find(|(id, _)| *id == model_id)
            .map(|(_, model)| model.clone())
    }
}

impl Embedder for FastEmbedder {
    fn method(&self) -> EmbedMethod {
        EmbedMethod::FastEmbed
    }

    fn embed(&self, model_id: i32, text_slices: Vec<&str>) -> Result<(Vec<f32>, usize, usize)> {
        let embedding_model = Self::get_embedding_model(model_id)
            .ok_or_else(|| anyhow::anyhow!("Invalid model ID: {}", model_id))?;

        FASTEMBED_MODELS.with(|cell| {
            let mut models = cell.borrow_mut();

            let model_instance = models.entry(model_id).or_insert_with(|| {
                TextEmbedding::try_new(
                    InitOptions::new(embedding_model)
                        .with_cache_dir(PathBuf::from("./fastembed_models")),
                )
                .expect("Failed to initialize model")
            });

            model_instance.embed_flat(text_slices, None)
        })
    }

    fn get_model_id(&self, model: &str) -> Option<i32> {
        let parsed = EmbeddingModel::from_str(model).ok()?;

        Self::MODELS
            .iter()
            .find(|(_, m)| *m == parsed)
            .map(|(id, _)| *id)
    }

    fn supports_model_id(&self, model_id: i32) -> bool {
        Self::get_embedding_model(model_id).is_some()
    }
}

#[linkme::distributed_slice(EMBEDDERS)]
static FASTEMBED: &dyn Embedder = &FastEmbedder;
