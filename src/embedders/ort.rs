// Setup:
//
// In order for this embedder to work, the official Microsoft ONNX Runtime release has to be
// downloaded:
//
//    wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.3/onnxruntime-linux-x64-1.24.3.tgz
//    tar xzf onnxruntime-linux-x64-1.24.3.tgz
//    sudo cp onnxruntime-linux-x64-1.24.3/lib/libonnxruntime.so.1.24.3 /usr/local/lib/
//    sudo ln -s /usr/local/lib/libonnxruntime.so.1.24.3 /usr/local/lib/libonnxruntime.so
//    sudo ldconfig
//
// The database adapters need to link against this dynamic library. For example:
//     ORT_LIB_DIR = /usr/local/lib
//
//    SHLIB_LINK = \
// 	      -L$(GEMBED_TARGET) \
// 	      -lgembed \
// 	      -L$(ORT_LIB_DIR) \
// 	      -lonnxruntime \
// 	      -Wl,-rpath,$(ORT_LIB_DIR)

#![cfg(feature = "ort")]
use crate::embedders::{Embedder, Input, InputType, ModelInfo, EMBEDDERS};
use crate::utils::flatten_vectors;
use anyhow::{bail, Result};
use image::imageops::FilterType;
use linkme::distributed_slice;
use ort::session::Session;
use ort::value::Value;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub static ORT_EMBEDDER_ID: i32 = 3;
pub static ORT_EMBEDDER_NAME: &str = "ort";

const DEFAULT_MODEL_BASE_DIR: &str = "/path/to/onnx_models";

#[distributed_slice]
pub static ORT_REGISTERED_MODELS: [ModelInfo] = [..];

thread_local! {
    static ORT_SESSIONS: RefCell<HashMap<i32, Session>> = RefCell::new(HashMap::new());
}

struct OrtEmbedder;

impl OrtEmbedder {
    fn lookup_model_registration(model_id: i32) -> Option<&'static ModelInfo> {
        ORT_REGISTERED_MODELS
            .iter()
            .find(|m| m.id() == model_id)
    }

    fn model_path(model_name: &str) -> PathBuf {
        PathBuf::from(DEFAULT_MODEL_BASE_DIR)
            .join(model_name)
            .join("model.onnx")
    }

    fn preprocess(image_path: &Path) -> Result<Vec<f32>> {
        let img = image::open(image_path)?
            .resize_to_fill(224, 224, FilterType::Triangle)
            .to_rgb8();

        let mut data = vec![0f32; 3 * 224 * 224];
        for (x, y, pixel) in img.enumerate_pixels() {
            for c in 0..3 {
                data[c * 224 * 224 + y as usize * 224 + x as usize] =
                    (pixel[c] as f32 / 255.0) * 2.0 - 1.0;
            }
        }
        Ok(data)
    }

    fn embed_image(session: &mut Session, path: &Path) -> Result<Vec<f32>> {
        let data = Self::preprocess(path)?;
        let input_value = Value::from_array(([1usize, 3, 224, 224], data.into_boxed_slice()))?;
        let input_name = session.inputs[0].name.clone();
        let outputs = session.run(HashMap::from([(
            input_name.as_str(),
            input_value,
        )]))?;

        let output = outputs.iter().next().unwrap().1;
        let (shape, data) = output.try_extract_tensor::<f32>()?;
        let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let embedding = match shape.as_slice() {
            [_, dim] => data[..*dim].to_vec(),
            [_, _seq, dim] => data[..*dim].to_vec(),
            _ => bail!("Unexpected output shape: {:?}", shape),
        };
        Ok(embedding)
    }
}

impl Embedder for OrtEmbedder {
    fn id(&self) -> i32 {
        ORT_EMBEDDER_ID
    }

    fn name(&self) -> &'static str {
        ORT_EMBEDDER_NAME
    }

    fn embed(&self, model_id: i32, input: Input) -> Result<(Vec<f32>, usize, usize)> {
        let model_info = Self::lookup_model_registration(model_id)
            .ok_or_else(|| anyhow::anyhow!("Invalid model ID: {}", model_id))?;

        ORT_SESSIONS.with(|cell| {
            let mut sessions = cell.borrow_mut();
            if !sessions.contains_key(&model_id) {
                let path = Self::model_path(model_info.name());
                let session = Session::builder()?.commit_from_file(&path)?;
                sessions.insert(model_id, session);
            }

            let session = sessions.get_mut(&model_id).unwrap();

            match input {
                Input::Images(images) => embed_images(session, images),
                Input::ImageDirectories(paths) => embed_image_directories(session, paths),
                _ => bail!("Input type not supported"),
            }
        })
    }

    fn model_info(&self, model_name: &str) -> Option<&ModelInfo> {
        ORT_REGISTERED_MODELS
            .iter()
            .find(|m| m.name() == model_name)
    }

    fn supports_input_for_model(&self, model_id: i32, input_type: InputType) -> bool {
        Self::lookup_model_registration(model_id)
            .map(|m| m.supports_input_type(input_type))
            .unwrap_or(false)
    }
}

fn embed_images(session: &mut Session, images: Vec<&[u8]>) -> Result<(Vec<f32>, usize, usize)> {
    if images.is_empty() {
        return Ok((vec![], 0, 0));
    }

    let tmp_dir = tempfile::TempDir::new()?;
    let tmp_path = tmp_dir.path();

    for (i, image) in images.iter().enumerate() {
        let extension = crate::utils::detect_image_format(image)?;
        let file_path = tmp_path.join(format!("{:06}.{}", i, extension));
        std::fs::write(&file_path, image)?;
    }

    let mut image_paths: Vec<PathBuf> = std::fs::read_dir(tmp_path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    image_paths.sort();

    let mut all_embeddings = Vec::new();
    for path in &image_paths {
        all_embeddings.push(OrtEmbedder::embed_image(session, path)?);
    }

    flatten_vectors(all_embeddings)
}

fn embed_image_directories(
    session: &mut Session,
    paths: Vec<&str>,
) -> Result<(Vec<f32>, usize, usize)> {
    let mut all_embeddings = Vec::new();

    for path in paths {
        let dir = Path::new(path);
        let image_paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| anyhow::anyhow!("Failed to read directory {}: {}", path, e))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| {
                        matches!(
                            e.to_lowercase().as_str(),
                            "jpg" | "jpeg" | "png" | "webp" | "bmp"
                        )
                    })
                    .unwrap_or(false)
            })
            .collect();

        for image_path in &image_paths {
            all_embeddings.push(OrtEmbedder::embed_image(session, image_path)?);
        }
    }

    flatten_vectors(all_embeddings)
}

#[linkme::distributed_slice(EMBEDDERS)]
static ORT: &dyn Embedder = &OrtEmbedder;

#[distributed_slice(ORT_REGISTERED_MODELS)]
static CLIP_VIT_BASE_PATCH32: ModelInfo = ModelInfo::new(
    0,
    "openai/clip-vit-base-patch32",
    &[InputType::Image, InputType::ImageDirectory],
);
