use crate::backends::{Input, InputType};
use crate::get_text_slices;
use anyhow::anyhow;
use std::os::raw::{c_char, c_float, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

pub const EXIT_SUCCESS: c_int = 0;
pub const GENERIC_ERROR: c_int = -1;
pub const ERR_INVALID_POINTER: c_int = -2;
pub const ERR_EMPTY_INPUT: c_int = -3;
pub const ERR_INVALID_UTF8: c_int = -4;
pub const ERR_INVALID_BACKEND: c_int = -5;
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
    pub n_binaries: usize,
    pub text_data: *const StringSlice,
    pub n_texts: usize,
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

/// Build the input based on input_type
pub fn build_input(input_data: &'_ InputData) -> anyhow::Result<Input<'_>, c_int> {
    let input = match input_data.input_type {
        InputType::Text => {
            if input_data.text_data.is_null() || input_data.n_texts == 0 {
                return Err(ERR_EMPTY_INPUT);
            }
            let texts = match unsafe { get_text_slices(input_data.text_data, input_data.n_texts) } {
                Ok(v) => v,
                Err(_) => return Err(ERR_INVALID_UTF8),
            };
            Input::Texts(texts)
        }
        InputType::Image => {
            if input_data.binary_data.is_null() || input_data.n_binaries == 0 {
                return Err(ERR_EMPTY_INPUT);
            }

            let slices =
                unsafe { slice::from_raw_parts(input_data.binary_data, input_data.n_binaries) };
            let mut images = Vec::with_capacity(input_data.n_binaries);
            for s in slices {
                if s.ptr.is_null() || s.len == 0 {
                    return Err(ERR_EMPTY_INPUT);
                }
                let img = unsafe { slice::from_raw_parts(s.ptr, s.len) };
                images.push(img);
            }
            Input::Images(images)
        }
        InputType::Multimodal => {
            let images = if !input_data.binary_data.is_null() && input_data.n_binaries > 0 {
                unsafe {
                    let slices =
                        slice::from_raw_parts(input_data.binary_data, input_data.n_binaries);
                    let mut imgs = Vec::with_capacity(input_data.n_binaries);
                    for s in slices {
                        if s.ptr.is_null() || s.len == 0 {
                            return Err(ERR_EMPTY_INPUT);
                        }
                        imgs.push(slice::from_raw_parts(s.ptr, s.len));
                    }
                    imgs
                }
            } else {
                vec![]
            };

            let texts = if !input_data.text_data.is_null() && input_data.n_texts > 0 {
                match unsafe { get_text_slices(input_data.text_data, input_data.n_texts) } {
                    Ok(v) => v,
                    Err(_) => return Err(ERR_INVALID_UTF8),
                }
            } else {
                vec![]
            };

            if images.is_empty() && texts.is_empty() {
                return Err(ERR_EMPTY_INPUT);
            }

            Input::Multimodal { images, texts }
        }
        InputType::ImageDirectory => {
            if input_data.text_data.is_null() || input_data.n_texts == 0 {
                return Err(ERR_EMPTY_INPUT);
            }
            let paths = match unsafe { get_text_slices(input_data.text_data, input_data.n_texts) } {
                Ok(v) => v,
                Err(_) => return Err(ERR_INVALID_UTF8),
            };
            Input::ImageDirectories(paths)
        }
    };

    Ok(input)
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
