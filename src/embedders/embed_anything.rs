#![cfg(feature = "embed_anything")]
use crate::embedders::{Embedder, Input, InputType, ModelInfo, EMBEDDERS};
use crate::utils::{detect_image_format, flatten_vectors, supports_input_for_model};
use anyhow::{anyhow, bail, Result};
use embed_anything::embed_image_directory;
use embed_anything::embeddings::embed::{Embedder as EAEmbedder, EmbedderBuilder};
use embed_anything::embeddings::local::text_embedding::ONNXModel;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::write;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::runtime::Runtime;

pub static EMBED_METHOD_EMBED_ANYTHING_ID: i32 = 2;
pub static EMBED_METHOD_EMBED_ANYTHING_NAME: &str = "embed_anything";

thread_local! {
    static RUNTIME: RefCell<Option<Runtime>> = RefCell::new(None);
    static EMBED_ANYTHING_MODELS: RefCell<HashMap<i32, Arc<EAEmbedder>>> = RefCell::new(HashMap::new());
}

struct EmbedAnythingEmbedder;

struct ModelDef {
    architecture: &'static str,
    onnx_model: Option<ONNXModel>,
    hf_model_id: Option<&'static str>,
}

impl EmbedAnythingEmbedder {
    const MODELS: &'static [ModelInfo] = &[
        ModelInfo::new(0, "Qdrant/all-MiniLM-L6-v2-onnx", &[InputType::Text]),
        ModelInfo::new(1, "Xenova/bge-large-en-v1.5", &[InputType::Text]),
        ModelInfo::new(
            2,
            "openai/clip-vit-base-patch32",
            &[InputType::Text, InputType::Image, InputType::Multimodal],
        ),
    ];

    fn model_def(model_id: i32) -> Option<ModelDef> {
        match model_id {
            0 => Some(ModelDef {
                architecture: "bert",
                onnx_model: Some(ONNXModel::AllMiniLML6V2),
                hf_model_id: None,
            }),
            1 => Some(ModelDef {
                architecture: "bert",
                onnx_model: Some(ONNXModel::BGELargeENV15),
                hf_model_id: None,
            }),
            2 => Some(ModelDef {
                architecture: "clip",
                onnx_model: None,
                hf_model_id: Some("openai/clip-vit-base-patch32"),
            }),
            _ => None,
        }
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
        let model_def =
            Self::model_def(model_id).ok_or_else(|| anyhow!("Invalid model ID: {}", model_id))?;

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
        EMBED_METHOD_EMBED_ANYTHING_ID
    }

    fn name(&self) -> &'static str {
        EMBED_METHOD_EMBED_ANYTHING_NAME
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
        Self::MODELS.iter().find(|m| m.name() == model_name)
    }

    fn supports_input_for_model(&self, model_id: i32, input_type: InputType) -> bool {
        supports_input_for_model(Self::MODELS, model_id, input_type)
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
