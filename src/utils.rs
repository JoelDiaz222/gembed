use crate::embedders::{InputType, ModelInfo};
use anyhow::anyhow;
use std::os::raw::{c_char, c_float, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};

pub const EXIT_SUCCESS: c_int = 0;
pub const GENERIC_ERROR: c_int = -1;
pub const ERR_INVALID_POINTER: c_int = -2;
pub const ERR_EMPTY_INPUT: c_int = -3;
pub const ERR_INVALID_UTF8: c_int = -4;
pub const ERR_INVALID_EMBEDDER: c_int = -5;
pub const ERR_MODEL_NOT_ALLOWED: c_int = -6;

#[repr(C)]
pub struct StringSlice {
    pub ptr: *const c_char,
    pub len: usize,
}

#[repr(C)]
pub struct ByteSlice {
    pub ptr: *const u8,
    pub len: usize,
}

#[repr(C)]
pub struct InputData {
    pub input_type: InputType,
    pub binary_data: *const ByteSlice,
    pub n_binary: usize,
    pub text_data: *const StringSlice,
    pub n_text: usize,
}

#[repr(C)]
pub struct EmbeddingBatch {
    pub data: *mut c_float,
    pub n_vectors: usize,
    pub dim: usize,
}

pub fn ffi_guard<F>(f: F) -> c_int
where
    F: FnOnce() -> c_int,
{
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| GENERIC_ERROR)
}

pub fn supports_input_for_model(
    models: &[ModelInfo],
    model_id: i32,
    input_type: InputType,
) -> bool {
    models
        .iter()
        .find(|m| m.id() == model_id)
        .map(|m| m.supports_input_type(input_type))
        .unwrap_or(false)
}

pub fn flatten_vectors(vectors: Vec<Vec<f32>>) -> anyhow::Result<(Vec<f32>, usize, usize)> {
    let n_vectors = vectors.len();
    let dim = vectors.first().map(|v| v.len()).unwrap_or(0);
    let total = n_vectors * dim;
    let mut flat: Vec<f32> = Vec::with_capacity(total);

    for v in vectors {
        flat.extend_from_slice(&v);
    }

    Ok((flat, n_vectors, dim))
}

#[cfg(feature = "embed_anything")]
pub fn detect_image_format(bytes: &[u8]) -> anyhow::Result<&'static str> {
    if bytes.len() < 12 {
        return Err(anyhow!("Image data too short to detect format"));
    }

    match bytes {
        [0xFF, 0xD8, 0xFF, ..] => Ok("jpg"),
        [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, ..] => Ok("png"),
        [
            0x52,
            0x49,
            0x46,
            0x46,
            _,
            _,
            _,
            _,
            0x57,
            0x45,
            0x42,
            0x50,
            ..,
        ] => Ok("webp"),
        [0x47, 0x49, 0x46, 0x38, b'7' | b'9', 0x61, ..] => Ok("gif"),
        [0x42, 0x4D, ..] => Ok("bmp"),
        _ => Err(anyhow!("Unknown or unsupported image format")),
    }
}
