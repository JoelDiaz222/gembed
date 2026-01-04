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
use std::fs::{write, OpenOptions};
use std::io::Write;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::runtime::Runtime;

pub static EMBED_ANYTHING_EMBEDDER_ID: i32 = 2;
pub static EMBED_ANYTHING_EMBEDDER_NAME: &str = "embed_anything";

// Logging helper
fn log_to_file(msg: &str) {
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/pg_gembed_debug.log")
    {
        let _ = writeln!(
            file,
            "[{}] {}",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
            msg
        );
    }
}

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
        log_to_file("Creating/getting runtime");
        RUNTIME.with(|cell| {
            let mut runtime_opt = cell.borrow_mut();
            if runtime_opt.is_none() {
                log_to_file("Initializing new tokio runtime");
                *runtime_opt = Some(Runtime::new()?);
                log_to_file("Tokio runtime initialized successfully");
            }

            // SAFETY: We're returning a reference that's tied to thread-local storage
            // The runtime lives for the entire thread lifetime
            Ok(unsafe { &*(runtime_opt.as_ref().unwrap() as *const Runtime) })
        })
    }

    fn embedder(model_id: i32) -> Result<Arc<EAEmbedder>> {
        log_to_file(&format!("Getting embedder for model_id: {}", model_id));

        let registration = Self::lookup_model_registration(model_id)
            .ok_or_else(|| anyhow!("Invalid model ID: {}", model_id))?;

        log_to_file(&format!(
            "Model registration found: {}",
            registration.info.name()
        ));

        let model_def = &registration.def;

        EMBED_ANYTHING_MODELS.with(|cell| {
            let mut models = cell.borrow_mut();
            if let Some(embedder) = models.get(&model_id) {
                log_to_file("Using cached embedder");
                return Ok(Arc::clone(embedder));
            }

            log_to_file(&format!(
                "Building new embedder with architecture: {}",
                model_def.architecture
            ));

            let builder = EmbedderBuilder::new().model_architecture(model_def.architecture);

            let embedder = if let Some(onnx_model) = model_def.onnx_model {
                log_to_file(&format!("Loading ONNX model: {:?}", onnx_model));
                log_to_file(&format!("HF_HOME env: {:?}", std::env::var("HF_HOME")));

                let result = builder
                    .onnx_model_id(Some(onnx_model))
                    .from_pretrained_onnx();

                match &result {
                    Ok(_) => log_to_file("ONNX model loaded successfully"),
                    Err(e) => log_to_file(&format!("ONNX model loading failed: {}", e)),
                }

                result?
            } else if let Some(hf_id) = model_def.hf_model_id {
                log_to_file(&format!("Loading HuggingFace model: {}", hf_id));
                log_to_file(&format!("HF_HOME env: {:?}", std::env::var("HF_HOME")));

                let result = builder.model_id(Some(hf_id)).from_pretrained_hf();

                match &result {
                    Ok(_) => log_to_file("HuggingFace model loaded successfully"),
                    Err(e) => log_to_file(&format!("HuggingFace model loading failed: {}", e)),
                }

                result?
            } else {
                log_to_file("ERROR: No model configuration found");
                bail!("No model configuration found");
            };

            log_to_file("Caching embedder");
            let arc_embedder = Arc::new(embedder);
            models.insert(model_id, Arc::clone(&arc_embedder));
            log_to_file("Embedder cached successfully");
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
        log_to_file(&format!("embed() called with model_id: {}", model_id));

        let embedder = Self::embedder(model_id)?;
        log_to_file("Embedder retrieved successfully");

        let runtime = Self::runtime()?;
        log_to_file("Runtime retrieved successfully");

        let result = match input {
            Input::Texts(ref texts) => {
                log_to_file(&format!("Embedding {} text(s)", texts.len()));
                embed_texts(texts.to_vec(), &embedder, runtime)
            }
            Input::Image(image) => {
                log_to_file(&format!("Embedding image ({} bytes)", image.len()));
                embed_image(image, &embedder, runtime)
            }
            Input::Multimodal { image, ref texts } => {
                log_to_file(&format!(
                    "Embedding multimodal (image: {}, texts: {})",
                    image.is_some(),
                    texts.len()
                ));
                embed_multimodal(image, texts.to_vec(), &embedder, runtime)
            }
        };

        match &result {
            Ok((flat, n, d)) => {
                log_to_file(&format!("Embedding successful: {} vectors of dim {}", n, d))
            }
            Err(e) => log_to_file(&format!("Embedding failed: {}", e)),
        }

        result
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
    log_to_file("embed_texts: Starting");
    let result = runtime.block_on(embedder.embed(&texts, None, None))?;
    log_to_file("embed_texts: Embed completed");

    let vectors: Vec<Vec<f32>> = result
        .into_iter()
        .map(|e| e.to_dense())
        .collect::<Result<_, _>>()?;

    log_to_file(&format!(
        "embed_texts: Converted to {} dense vectors",
        vectors.len()
    ));

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
