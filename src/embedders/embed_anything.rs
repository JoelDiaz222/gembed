#![cfg(feature = "embed_anything")]
use crate::embedders::{Embedder, Input, InputType, ModelInfo, EMBEDDERS};
use crate::utils::{detect_image_format, flatten_vectors};
use anyhow::{anyhow, bail, Result};
use embed_anything::embed_image_directory;
use embed_anything::embeddings::embed::{Embedder as EAEmbedder, EmbedderBuilder};
use embed_anything::embeddings::local::text_embedding::ONNXModel;
use linkme::distributed_slice;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::write;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::runtime::Runtime;

pub static EMBED_ANYTHING_EMBEDDER_ID: i32 = 2;
pub static EMBED_ANYTHING_EMBEDDER_NAME: &str = "embed_anything";

struct ModelDef {
    architecture: &'static str,
    onnx_model: Option<ONNXModel>,
    hf_model_id: Option<&'static str>,
}

struct ModelRegistration {
    info: ModelInfo,
    def: ModelDef,
}

#[distributed_slice]
pub static EMBED_ANYTHING_REGISTERED_MODELS: [ModelRegistration] = [..];

thread_local! {
    static RUNTIME: RefCell<Option<Runtime>> = RefCell::new(None);
    static EMBED_ANYTHING_MODELS: RefCell<HashMap<i32, Arc<EAEmbedder>>> = RefCell::new(HashMap::new());
}

struct EmbedAnythingEmbedder;

impl EmbedAnythingEmbedder {
    fn lookup_model_registration(model_id: i32) -> Option<&'static ModelRegistration> {
        EMBED_ANYTHING_REGISTERED_MODELS
            .iter()
            .find(|reg| reg.info.id() == model_id)
    }

    fn runtime() -> Result<&'static Runtime> {
        RUNTIME.with(|cell| {
            let mut runtime_opt = cell.borrow_mut();
            if runtime_opt.is_none() {
                *runtime_opt = Some(Runtime::new()?);
            }

            // SAFETY: We're returning a reference that's tied to thread-local storage
            // The runtime lives for the entire thread lifetime
            Ok(unsafe { &*(runtime_opt.as_ref().unwrap() as *const Runtime) })
        })
    }

    fn embedder(model_id: i32) -> Result<Arc<EAEmbedder>> {
        let registration = Self::lookup_model_registration(model_id)
            .ok_or_else(|| anyhow!("Invalid model ID: {}", model_id))?;

        let model_def = &registration.def;

        EMBED_ANYTHING_MODELS.with(|cell| {
            let mut models = cell.borrow_mut();
            if let Some(embedder) = models.get(&model_id) {
                return Ok(Arc::clone(embedder));
            }

            let builder = EmbedderBuilder::new().model_architecture(model_def.architecture);

            let embedder = if let Some(onnx_model) = model_def.onnx_model {
                builder
                    .onnx_model_id(Some(onnx_model))
                    .from_pretrained_onnx()?
            } else if let Some(hf_id) = model_def.hf_model_id {
                builder.model_id(Some(hf_id)).from_pretrained_hf()?
            } else {
                bail!("No model configuration found");
            };

            let arc_embedder = Arc::new(embedder);
            models.insert(model_id, Arc::clone(&arc_embedder));
            Ok(arc_embedder)
        })
    }
}

impl Embedder for EmbedAnythingEmbedder {
    fn id(&self) -> i32 {
        EMBED_ANYTHING_EMBEDDER_ID
    }

    fn name(&self) -> &'static str {
        EMBED_ANYTHING_EMBEDDER_NAME
    }

    fn embed(&self, model_id: i32, input: Input) -> Result<(Vec<f32>, usize, usize)> {
        let embedder = Self::embedder(model_id)?;
        let runtime = Self::runtime()?;

        match input {
            Input::Texts(texts) => embed_texts(texts, &embedder, runtime),
            Input::Image(image) => embed_image(image, &embedder, runtime),
            Input::Multimodal { image, texts } => {
                embed_multimodal(image, texts, &embedder, runtime)
            }
        }
    }

    fn model_info(&self, model_name: &str) -> Option<&ModelInfo> {
        EMBED_ANYTHING_REGISTERED_MODELS
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

fn embed_texts(
    texts: Vec<&str>,
    embedder: &Arc<EAEmbedder>,
    runtime: &Runtime,
) -> Result<(Vec<f32>, usize, usize)> {
    let result = runtime.block_on(embedder.embed(&texts, None, None))?;

    let vectors: Vec<Vec<f32>> = result
        .into_iter()
        .map(|e| e.to_dense())
        .collect::<Result<_, _>>()?;

    flatten_vectors(vectors)
}

fn embed_image(
    image: &[u8],
    embedder: &Arc<EAEmbedder>,
    runtime: &Runtime,
) -> Result<(Vec<f32>, usize, usize)> {
    let tmp_dir = TempDir::new()?;
    let tmp_path = tmp_dir.path();

    let extension = detect_image_format(image)?;
    let file_path = tmp_path.join(format!("image.{}", extension));
    write(&file_path, image)?;

    let result = runtime
        .block_on(embed_image_directory(
            tmp_path.to_path_buf(),
            embedder,
            None,
            None,
        ))?
        .ok_or_else(|| anyhow!("No images were processed"))?;

    let embedding = result
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("No embedding returned"))?
        .embedding
        .to_dense()?;

    let dim = embedding.len();
    Ok((embedding, 1, dim))
}

fn embed_multimodal(
    image: Option<&[u8]>,
    texts: Vec<&str>,
    embedder: &Arc<EAEmbedder>,
    runtime: &Runtime,
) -> Result<(Vec<f32>, usize, usize)> {
    let mut all_embeddings = Vec::new();

    if let Some(img_bytes) = image {
        let (img_embedding, _, _) = embed_image(img_bytes, embedder, runtime)?;
        all_embeddings.push(img_embedding);
    }

    if !texts.is_empty() {
        let (text_embeddings, n_text, dim) = embed_texts(texts, embedder, runtime)?;
        for i in 0..n_text {
            let start = i * dim;
            let end = start + dim;
            all_embeddings.push(text_embeddings[start..end].to_vec());
        }
    }

    flatten_vectors(all_embeddings)
}

#[linkme::distributed_slice(EMBEDDERS)]
static EMBED_ANYTHING: &dyn Embedder = &EmbedAnythingEmbedder;

#[distributed_slice(EMBED_ANYTHING_REGISTERED_MODELS)]
static ALL_MINI_LM_L6_V2: ModelRegistration = ModelRegistration {
    info: ModelInfo::new(
        0,
        "sentence-transformers/all-MiniLM-L6-v2",
        &[InputType::Text],
    ),
    def: ModelDef {
        architecture: "bert",
        onnx_model: None,
        hf_model_id: Some("sentence-transformers/all-MiniLM-L6-v2"),
    },
};

#[distributed_slice(EMBED_ANYTHING_REGISTERED_MODELS)]
static ALL_MINI_LM_L6_V2_ONNX: ModelRegistration = ModelRegistration {
    info: ModelInfo::new(1, "Qdrant/all-MiniLM-L6-v2-onnx", &[InputType::Text]),
    def: ModelDef {
        architecture: "bert",
        onnx_model: Some(ONNXModel::AllMiniLML6V2),
        hf_model_id: None,
    },
};

#[distributed_slice(EMBED_ANYTHING_REGISTERED_MODELS)]
static CLIP_VIT_BASE_PATCH32: ModelRegistration = ModelRegistration {
    info: ModelInfo::new(
        2,
        "openai/clip-vit-base-patch32",
        &[InputType::Text, InputType::Image, InputType::Multimodal],
    ),
    def: ModelDef {
        architecture: "clip",
        onnx_model: None,
        hf_model_id: Some("openai/clip-vit-base-patch32"),
    },
};
