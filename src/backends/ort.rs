// Setup:
//
// In order for this backend to work, the official Microsoft ONNX Runtime release has to be
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
use crate::backends::{Backend, Input, InputType, ModelInfo, BACKENDS};
use crate::utils::flatten_vectors;
use anyhow::{bail, Result};
use image::imageops::FilterType;
use linkme::distributed_slice;
use ort::session::Session;
use ort::value::Value;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

static ORT_BACKEND_ID: i32 = 4;
static ORT_BACKEND_NAME: &str = "ort";

const DEFAULT_MODEL_BASE_DIR: &str = "/path/to/onnx_models";

#[derive(Clone)]
enum Normalization {
    /// CLIP: pixel / 255 * 2 - 1  →  [-1, 1]
    Affine,
    /// SigLIP: (pixel / 255 - mean) / std
    MeanStd { mean: [f32; 3], std: [f32; 3] },
}

#[derive(Clone)]
struct ModelDef {
    image_size: usize,
    normalization: Normalization,
}

struct ModelRegistration {
    info: ModelInfo,
    def: ModelDef,
}

#[distributed_slice]
static ORT_REGISTERED_MODELS: [ModelRegistration] = [..];

thread_local! {
    static ORT_SESSIONS: RefCell<HashMap<i32, Session>> = RefCell::new(HashMap::new());
}

struct OrtBackend;

impl OrtBackend {
    fn lookup_model_registration(model_id: i32) -> Option<&'static ModelRegistration> {
        ORT_REGISTERED_MODELS
            .iter()
            .find(|m| m.info.id() == model_id)
    }

    fn model_path(model_name: &str) -> PathBuf {
        PathBuf::from(DEFAULT_MODEL_BASE_DIR)
            .join(model_name)
            .join("model.onnx")
    }

    fn preprocess(image_path: &Path, def: &ModelDef) -> Result<Vec<f32>> {
        let size = def.image_size as u32;
        let img = image::open(image_path)?
            .resize_to_fill(size, size, FilterType::Triangle)
            .to_rgb8();

        let n = def.image_size;
        let mut data = vec![0f32; 3 * n * n];
        for (x, y, pixel) in img.enumerate_pixels() {
            for c in 0..3 {
                let val = pixel[c] as f32 / 255.0;
                data[c * n * n + y as usize * n + x as usize] = match &def.normalization {
                    Normalization::Affine => val * 2.0 - 1.0,
                    Normalization::MeanStd { mean, std } => (val - mean[c]) / std[c],
                };
            }
        }
        Ok(data)
    }

    fn embed_image(session: &mut Session, path: &Path, def: &ModelDef) -> Result<Vec<f32>> {
        let n = def.image_size;
        let data = Self::preprocess(path, def)?;
        let input_value = Value::from_array(([1usize, 3, n, n], data.into_boxed_slice()))?;
        let input_name = session.inputs[0].name.clone();
        let outputs = session.run(HashMap::from([(input_name.as_str(), input_value)]))?;

        let output = outputs.iter().next().unwrap().1;
        let (shape, data) = output.try_extract_tensor::<f32>()?;
        let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let embedding = match shape.as_slice() {
            [_, dim] => data[..*dim].to_vec(),
            [_, _seq, dim] => data[..*dim].to_vec(),
            _ => bail!("Unexpected output shape: {:?}", shape),
        };

        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        Ok(embedding.iter().map(|x| x / norm).collect())
    }
}

impl Backend for OrtBackend {
    fn id(&self) -> i32 {
        ORT_BACKEND_ID
    }

    fn name(&self) -> &'static str {
        ORT_BACKEND_NAME
    }

    #[cfg(not(feature = "dynamic_model_loading"))]
    fn embed(&self, model_id: i32, input: Input) -> Result<(Vec<f32>, usize, usize)> {
        let model_name = self.resolve_model_name(model_id)?;

        let registration = ORT_REGISTERED_MODELS
            .iter()
            .find(|reg| reg.info.name() == model_name)
            .ok_or_else(|| anyhow::anyhow!("Invalid model: {}", model_name))?;

        ORT_SESSIONS.with(|cell| {
            let mut sessions = cell.borrow_mut();
            if !sessions.contains_key(&model_id) {
                let path = Self::model_path(&model_name);
                let session = Session::builder()?.commit_from_file(&path)?;
                sessions.insert(model_id, session);
            }

            let session = sessions.get_mut(&model_id).unwrap();
            let def = registration.def.clone();

            match input {
                Input::Images(images) => embed_images(session, images, &def),
                Input::ImageDirectories(paths) => embed_image_directories(session, paths, &def),
                _ => bail!("Input type not supported"),
            }
        })
    }

    #[cfg(feature = "dynamic_model_loading")]
    fn embed(&self, model_id: i32, input: Input) -> Result<(Vec<f32>, usize, usize)> {
        let model_name = self.resolve_model_name(model_id)?;

        ORT_SESSIONS.with(|cell| {
            let mut sessions = cell.borrow_mut();
            if !sessions.contains_key(&model_id) {
                let path = Self::model_path(&model_name);
                let session = Session::builder()?.commit_from_file(&path)?;
                sessions.insert(model_id, session);
            }

            let session = sessions.get_mut(&model_id).unwrap();
            let def = ModelDef {
                image_size: 224,
                normalization: Normalization::Affine,
            };

            match input {
                Input::Images(images) => embed_images(session, images, &def),
                Input::ImageDirectories(paths) => embed_image_directories(session, paths, &def),
                _ => bail!("Input type not supported"),
            }
        })
    }

    fn model_info(&self, model_name: &str) -> Option<&ModelInfo> {
        ORT_REGISTERED_MODELS
            .iter()
            .find(|m| m.info.name() == model_name)
            .map(|m| &m.info)
    }

    fn model_info_by_id(&self, model_id: i32) -> Option<&ModelInfo> {
        Self::lookup_model_registration(model_id).map(|reg| &reg.info)
    }
}

fn embed_images(
    session: &mut Session,
    images: Vec<&[u8]>,
    def: &ModelDef,
) -> Result<(Vec<f32>, usize, usize)> {
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
        all_embeddings.push(OrtBackend::embed_image(session, path, def)?);
    }

    flatten_vectors(all_embeddings)
}

fn embed_image_directories(
    session: &mut Session,
    paths: Vec<&str>,
    def: &ModelDef,
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
            all_embeddings.push(OrtBackend::embed_image(session, image_path, def)?);
        }
    }

    flatten_vectors(all_embeddings)
}

#[linkme::distributed_slice(BACKENDS)]
static ORT: &dyn Backend = &OrtBackend;

#[distributed_slice(ORT_REGISTERED_MODELS)]
static CLIP_VIT_BASE_PATCH32: ModelRegistration = ModelRegistration {
    info: ModelInfo::new(
        0,
        "openai/clip-vit-base-patch32",
        &[InputType::Image, InputType::ImageDirectory],
    ),
    def: ModelDef {
        image_size: 224,
        normalization: Normalization::Affine,
    },
};

#[distributed_slice(ORT_REGISTERED_MODELS)]
static CLIP_VIT_LARGE_PATCH14: ModelRegistration = ModelRegistration {
    info: ModelInfo::new(
        1,
        "openai/clip-vit-large-patch14",
        &[InputType::Image, InputType::ImageDirectory],
    ),
    def: ModelDef {
        image_size: 224,
        normalization: Normalization::Affine,
    },
};

#[distributed_slice(ORT_REGISTERED_MODELS)]
static SIGLIP_LARGE_PATCH16_384: ModelRegistration = ModelRegistration {
    info: ModelInfo::new(
        2,
        "google/siglip-large-patch16-384",
        &[InputType::Image, InputType::ImageDirectory],
    ),
    def: ModelDef {
        image_size: 384,
        normalization: Normalization::MeanStd {
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        },
    },
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::{BackendRegistry, InputType};

    #[test]
    fn ort_backend_is_registered() {
        assert_eq!(
            BackendRegistry::lookup_backend_id(ORT_BACKEND_NAME),
            Some(ORT_BACKEND_ID)
        );
    }

    #[test]
    fn all_ort_models_have_unique_ids() {
        // Act
        let ids: Vec<i32> = ORT_REGISTERED_MODELS.iter().map(|m| m.info.id()).collect();

        // Assert
        let unique: std::collections::HashSet<i32> = ids.iter().copied().collect();
        assert_eq!(ids.len(), unique.len());
    }

    #[test]
    fn all_ort_models_have_unique_names() {
        // Act
        let names: Vec<&str> = ORT_REGISTERED_MODELS
            .iter()
            .map(|m| m.info.name())
            .collect();

        // Assert
        let unique: std::collections::HashSet<&str> = names.iter().copied().collect();
        assert_eq!(names.len(), unique.len());
    }

    #[test]
    fn all_ort_models_support_image_input() {
        // Assert
        for m in ORT_REGISTERED_MODELS.iter() {
            assert!(
                m.info.supports_input_type(InputType::Image),
                "ORT model {} must support Image",
                m.info.name()
            );
        }
    }

    #[test]
    fn clip_models_use_affine_normalization() {
        // Assert
        for m in ORT_REGISTERED_MODELS
            .iter()
            .filter(|m| m.info.name().contains("clip"))
        {
            match &m.def.normalization {
                Normalization::Affine => {}
                _ => panic!(
                    "CLIP model {} should use Affine normalization",
                    m.info.name()
                ),
            }
        }
    }

    #[test]
    fn siglip_models_use_mean_std_normalization() {
        // Assert
        for m in ORT_REGISTERED_MODELS
            .iter()
            .filter(|m| m.info.name().contains("siglip"))
        {
            match &m.def.normalization {
                Normalization::MeanStd { .. } => {}
                _ => panic!(
                    "SigLIP model {} should use MeanStd normalization",
                    m.info.name()
                ),
            }
        }
    }

    mod preprocess_helper {
        use super::*;
        use std::path::Path;

        #[test]
        fn preprocess_returns_correct_length() {
            // Arrange
            let tmp = tempfile::Builder::new().suffix(".png").tempfile().unwrap();
            let img = image::RgbImage::new(1, 1);
            img.save(tmp.path()).unwrap();

            let def = ModelDef {
                image_size: 224,
                normalization: Normalization::Affine,
            };

            // Act
            let result = OrtBackend::preprocess(tmp.path(), &def).unwrap();

            // Assert
            // 3 channels * 224 * 224
            assert_eq!(result.len(), 3 * 224 * 224);
        }

        #[test]
        fn preprocess_fails_for_missing_file() {
            // Arrange
            let def = ModelDef {
                image_size: 10,
                normalization: Normalization::Affine,
            };

            // Act
            let result = OrtBackend::preprocess(Path::new("__missing__"), &def);

            // Assert
            assert!(result.is_err());
        }
    }
}
