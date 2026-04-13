#![cfg(feature = "embed_anything")]
use crate::backends::{Backend, Input, InputType, ModelInfo, BACKENDS};
use crate::telemetry::tlog;
use crate::utils::{detect_image_format, flatten_vectors};
use anyhow::{anyhow, Result};
use embed_anything::embed_image_directory;
use embed_anything::embeddings::embed::{Embedder as EABackend, EmbedderBuilder};
use embed_anything::embeddings::local::text_embedding::ONNXModel;
use linkme::distributed_slice;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::write;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::runtime::Runtime;

static EMBED_ANYTHING_BACKEND_ID: i32 = 2;
static EMBED_ANYTHING_BACKEND_NAME: &str = "embed_anything";

struct ModelDef {
    #[allow(dead_code)]
    architecture: &'static str,
    #[allow(dead_code)]
    onnx_model: Option<ONNXModel>,
    #[allow(dead_code)]
    hf_model_id: Option<&'static str>,
}

struct ModelRegistration {
    info: ModelInfo,
    #[allow(dead_code)]
    def: ModelDef,
}

#[distributed_slice]
static EMBED_ANYTHING_REGISTERED_MODELS: [ModelRegistration] = [..];

thread_local! {
    static RUNTIME: RefCell<Option<Runtime>> = const { RefCell::new(None) };
    static EMBED_ANYTHING_MODELS: RefCell<HashMap<i32, Arc<EABackend>>> = RefCell::new(HashMap::new());
}

struct EmbedAnythingBackend;

impl EmbedAnythingBackend {
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

    #[cfg(not(feature = "dynamic_model_loading"))]
    fn backend(model_id: i32, model_name: String) -> Result<Arc<EABackend>> {
        let registration = EMBED_ANYTHING_REGISTERED_MODELS
            .iter()
            .find(|reg| reg.info.name() == model_name)
            .ok_or_else(|| anyhow::anyhow!("Invalid model: {}", model_name))?;

        EMBED_ANYTHING_MODELS.with(|cell| {
            let mut models = cell.borrow_mut();
            if let Some(backend) = models.get(&model_id) {
                tlog("rs_ea_model_cache_hit", 0);
                return Ok(Arc::clone(backend));
            }

            tlog("rs_ea_model_load_start", 0);

            let model_def = &registration.def;
            let builder = EmbedderBuilder::new().model_architecture(model_def.architecture);

            let backend = if let Some(onnx_model) = model_def.onnx_model {
                builder
                    .onnx_model_id(Some(onnx_model))
                    .from_pretrained_onnx()?
            } else if let Some(hf_id) = model_def.hf_model_id {
                builder.model_id(Some(hf_id)).from_pretrained_hf()?
            } else {
                anyhow::bail!("No model configuration found");
            };

            tlog("rs_ea_model_load_done", 0);

            let arc_backend = Arc::new(backend);
            models.insert(model_id, Arc::clone(&arc_backend));
            Ok(arc_backend)
        })
    }

    #[cfg(feature = "dynamic_model_loading")]
    fn backend(model_id: i32, model_name: String) -> Result<Arc<EABackend>> {
        EMBED_ANYTHING_MODELS.with(|cell| {
            let mut models = cell.borrow_mut();
            if let Some(backend) = models.get(&model_id) {
                return Ok(Arc::clone(backend));
            }

            let backend = EmbedderBuilder::new()
                .model_architecture("bert")
                .model_id(Some(&model_name))
                .from_pretrained_hf()?;

            let arc_backend = Arc::new(backend);
            models.insert(model_id, Arc::clone(&arc_backend));
            Ok(arc_backend)
        })
    }
}

impl Backend for EmbedAnythingBackend {
    fn id(&self) -> i32 {
        EMBED_ANYTHING_BACKEND_ID
    }

    fn name(&self) -> &'static str {
        EMBED_ANYTHING_BACKEND_NAME
    }

    fn embed(&self, model_id: i32, input: Input) -> Result<(Vec<f32>, usize, usize)> {
        let backend = Self::backend(model_id, self.resolve_model_name(model_id)?)?;
        let runtime = Self::runtime()?;

        match input {
            Input::Texts(texts) => embed_texts(texts, &backend, runtime),
            Input::Images(images) => embed_images(images, &backend, runtime),
            Input::Multimodal { images, texts } => {
                embed_multimodal(images, texts, &backend, runtime)
            }
            Input::ImageDirectories(paths) => embed_image_directories(paths, &backend, runtime),
        }
    }

    fn model_info(&self, model_name: &str) -> Option<&ModelInfo> {
        EMBED_ANYTHING_REGISTERED_MODELS
            .iter()
            .find(|reg| reg.info.name() == model_name)
            .map(|reg| &reg.info)
    }

    fn model_info_by_id(&self, model_id: i32) -> Option<&ModelInfo> {
        Self::lookup_model_registration(model_id).map(|reg| &reg.info)
    }
}

fn embed_image_directories(
    paths: Vec<&str>,
    backend: &Arc<EABackend>,
    runtime: &Runtime,
) -> Result<(Vec<f32>, usize, usize)> {
    let mut all_embeddings = Vec::new();

    for path in paths {
        let result = runtime
            .block_on(embed_image_directory(
                std::path::PathBuf::from(path),
                backend,
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
    backend: &Arc<EABackend>,
    runtime: &Runtime,
) -> Result<(Vec<f32>, usize, usize)> {
    tlog("rs_ea_embed_texts_start", texts.len());

    let result = runtime.block_on(backend.embed(&texts, None, None))?;

    tlog("rs_ea_embed_texts_done", texts.len());

    let vectors: Vec<Vec<f32>> = result
        .into_iter()
        .map(|e| e.to_dense())
        .collect::<Result<_, _>>()?;

    flatten_vectors(vectors)
}

fn embed_images(
    images: Vec<&[u8]>,
    backend: &Arc<EABackend>,
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
            backend,
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
    backend: &Arc<EABackend>,
    runtime: &Runtime,
) -> Result<(Vec<f32>, usize, usize)> {
    let mut all_embeddings = Vec::new();

    if !images.is_empty() {
        let (img_embeddings, n_img, dim) = embed_images(images, backend, runtime)?;
        for i in 0..n_img {
            all_embeddings.push(img_embeddings[i * dim..(i + 1) * dim].to_vec());
        }
    }

    if !texts.is_empty() {
        let (text_embeddings, n_texts, dim) = embed_texts(texts, backend, runtime)?;
        for i in 0..n_texts {
            all_embeddings.push(text_embeddings[i * dim..(i + 1) * dim].to_vec());
        }
    }

    flatten_vectors(all_embeddings)
}

#[linkme::distributed_slice(BACKENDS)]
static EMBED_ANYTHING: &dyn Backend = &EmbedAnythingBackend;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::{BackendRegistry, InputType};

    #[test]
    fn embed_anything_backend_is_registered() {
        // Act
        let id = BackendRegistry::lookup_backend_id(EMBED_ANYTHING_BACKEND_NAME);

        // Assert
        assert_eq!(id, Some(EMBED_ANYTHING_BACKEND_ID));
    }

    #[test]
    fn all_models_have_unique_ids() {
        // Act
        let ids: Vec<i32> = EMBED_ANYTHING_REGISTERED_MODELS
            .iter()
            .map(|r| r.info.id())
            .collect();

        // Assert
        let unique: std::collections::HashSet<i32> = ids.iter().copied().collect();
        assert_eq!(ids.len(), unique.len());
    }

    #[test]
    fn clip_supports_multimodal_and_directory() {
        // Act
        let clip = EMBED_ANYTHING_REGISTERED_MODELS
            .iter()
            .find(|m| m.info.name().contains("clip-vit-base"))
            .expect("CLIP not found");

        // Assert
        assert!(clip.info.supports_input_type(InputType::Multimodal));
        assert!(clip.info.supports_input_type(InputType::ImageDirectory));
    }

    #[test]
    fn mini_lm_is_text_only() {
        // Act
        let mini = EMBED_ANYTHING_REGISTERED_MODELS
            .iter()
            .find(|m| m.info.name().contains("all-MiniLM"))
            .expect("MiniLM not found");

        // Assert
        assert!(mini.info.supports_input_type(InputType::Text));
        assert!(!mini.info.supports_input_type(InputType::Image));
    }

    mod runtime_pool {
        use super::*;

        #[test]
        fn runtime_is_initialized_successfully() {
            // Act
            let rt = EmbedAnythingBackend::runtime();

            // Assert
            assert!(rt.is_ok());
        }

        #[test]
        fn multiple_calls_return_consistent_runtime() {
            // Act
            let rt1 = EmbedAnythingBackend::runtime().unwrap() as *const _;
            let rt2 = EmbedAnythingBackend::runtime().unwrap() as *const _;

            // Assert
            assert_eq!(rt1, rt2);
        }
    }
}
