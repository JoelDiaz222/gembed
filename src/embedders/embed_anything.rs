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
            Input::Images(images) => embed_images(images, &embedder, runtime),
            Input::Multimodal { images, texts } => {
                embed_multimodal(images, texts, &embedder, runtime)
            }
            Input::ImageDirectories(paths) => embed_image_directories(paths, &embedder, runtime),
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

fn embed_image_directories(
    paths: Vec<&str>,
    embedder: &Arc<EAEmbedder>,
    runtime: &Runtime,
) -> Result<(Vec<f32>, usize, usize)> {
    let mut all_embeddings = Vec::new();

    for path in paths {
        let result = runtime
            .block_on(embed_image_directory(
                std::path::PathBuf::from(path),
                embedder,
                None,
                None,
            ))?
            .ok_or_else(|| anyhow!("No images were processed in directory: {}", path))?;

        for e in result {
            all_embeddings.push(e.embedding.to_dense()?);
        }
    }

    flatten_vectors(all_embeddings)
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

fn embed_images(
    images: Vec<&[u8]>,
    embedder: &Arc<EAEmbedder>,
    runtime: &Runtime,
) -> Result<(Vec<f32>, usize, usize)> {
    if images.is_empty() {
        return Ok((vec![], 0, 0));
    }

    let tmp_dir = TempDir::new()?;
    let tmp_path = tmp_dir.path();

    for (i, image) in images.iter().enumerate() {
        let extension = detect_image_format(image)?;
        let file_path = tmp_path.join(format!("{:06}.{}", i, extension));
        write(&file_path, image)?;
    }

    let result = runtime
        .block_on(embed_image_directory(
            tmp_path.to_path_buf(),
            embedder,
            None,
            None,
        ))?
        .ok_or_else(|| anyhow!("No images were processed"))?;

    // Sort results by filename to ensure order matches input
    let mut embeddings_with_idx: Vec<(usize, Vec<f32>)> = result
        .into_iter()
        .map(|e| {
            let metadata = e.metadata.as_ref();
            let filename_str = metadata
                .and_then(|m| m.get("file_name"))
                .ok_or_else(|| anyhow!("Missing file_name in metadata"))?;

            let stem = std::path::Path::new(filename_str)
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or_else(|| anyhow!("Invalid filename in metadata: {}", filename_str))?;

            let idx = stem.parse()?;
            Ok((idx, e.embedding.to_dense()?))
        })
        .collect::<Result<Vec<_>>>()?;

    embeddings_with_idx.sort_by_key(|k| k.0);
    let vectors: Vec<Vec<f32>> = embeddings_with_idx.into_iter().map(|(_, v)| v).collect();
    flatten_vectors(vectors)
}

fn embed_multimodal(
    images: Vec<&[u8]>,
    texts: Vec<&str>,
    embedder: &Arc<EAEmbedder>,
    runtime: &Runtime,
) -> Result<(Vec<f32>, usize, usize)> {
    let mut all_embeddings = Vec::new();

    if !images.is_empty() {
        let (img_embeddings, n_img, dim) = embed_images(images, embedder, runtime)?;
        for i in 0..n_img {
            all_embeddings.push(img_embeddings[i * dim..(i + 1) * dim].to_vec());
        }
    }

    if !texts.is_empty() {
        let (text_embeddings, n_texts, dim) = embed_texts(texts, embedder, runtime)?;
        for i in 0..n_texts {
            all_embeddings.push(text_embeddings[i * dim..(i + 1) * dim].to_vec());
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
static BGE_LARGE_EN_V15: ModelRegistration = ModelRegistration {
    info: ModelInfo::new(1, "BAAI/bge-large-en-v1.5", &[InputType::Text]),
    def: ModelDef {
        architecture: "bert",
        onnx_model: None,
        hf_model_id: Some("BAAI/bge-large-en-v1.5"),
    },
};

#[distributed_slice(EMBED_ANYTHING_REGISTERED_MODELS)]
static CLIP_VIT_BASE_PATCH32: ModelRegistration = ModelRegistration {
    info: ModelInfo::new(
        2,
        "openai/clip-vit-base-patch32",
        &[
            InputType::Text,
            InputType::Image,
            InputType::Multimodal,
            InputType::ImageDirectory,
        ],
    ),
    def: ModelDef {
        architecture: "clip",
        onnx_model: None,
        hf_model_id: Some("openai/clip-vit-base-patch32"),
    },
};

#[distributed_slice(EMBED_ANYTHING_REGISTERED_MODELS)]
static CLIP_VIT_LARGE_PATCH14: ModelRegistration = ModelRegistration {
    info: ModelInfo::new(
        3,
        "openai/clip-vit-large-patch14",
        &[
            InputType::Text,
            InputType::Image,
            InputType::Multimodal,
            InputType::ImageDirectory,
        ],
    ),
    def: ModelDef {
        architecture: "clip",
        onnx_model: None,
        hf_model_id: Some("openai/clip-vit-large-patch14"),
    },
};

#[distributed_slice(EMBED_ANYTHING_REGISTERED_MODELS)]
static SIGLIP_LARGE_PATCH16_384: ModelRegistration = ModelRegistration {
    info: ModelInfo::new(
        4,
        "google/siglip-large-patch16-384",
        &[
            InputType::Text,
            InputType::Image,
            InputType::Multimodal,
            InputType::ImageDirectory,
        ],
    ),
    def: ModelDef {
        architecture: "siglip",
        onnx_model: None,
        hf_model_id: Some("google/siglip-large-patch16-384"),
    },
};
