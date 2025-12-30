mod embedders;
use crate::embedders::{EmbedderRegistry, Input, InputType};
use anyhow::Result;
use std::ffi::CStr;
use std::os::raw::{c_char, c_float, c_int};
use std::slice;

const EXIT_SUCCESS: c_int = 0;
const ERR_INVALID_POINTERS: c_int = -1;
const ERR_EMPTY_INPUT: c_int = -2;
const ERR_INVALID_UTF8: c_int = -3;
const ERR_INVALID_EMBEDDER: c_int = -4;
const ERR_MODEL_NOT_ALLOWED: c_int = -5;
const ERR_EMBEDDING_FAILED: c_int = -6;

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

/// Validates embedder by name and returns its ID
/// Returns -1 if non-existent
#[unsafe(no_mangle)]
pub extern "C" fn validate_embedder(name: *const c_char) -> c_int {
    if name.is_null() {
        return -1;
    }

    let name_str = unsafe {
        match CStr::from_ptr(name).to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        }
    };

    EmbedderRegistry::get_embedder_id(name_str).unwrap_or(-1)
}

/// Validates model string for given embedder and input type, returns model ID
/// Returns -1 if non-existent or doesn't support the input type
#[unsafe(no_mangle)]
pub extern "C" fn validate_embedding_model(
    embedder_id: c_int,
    model_name: *const c_char,
    input_type: InputType,
) -> c_int {
    if model_name.is_null() {
        return -1;
    }

    let model_str = unsafe {
        match CStr::from_ptr(model_name).to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        }
    };

    EmbedderRegistry::validate_model(embedder_id, model_str, input_type).unwrap_or(-1)
}

/// Embeds the input_data using the embedder and model specified by ID,
/// and stores the result in out_batch
#[unsafe(no_mangle)]
pub extern "C" fn generate_embeddings(
    embedder_id: c_int,
    model_id: c_int,
    input_data: *const InputData,
    out_batch: *mut EmbeddingBatch,
) -> c_int {
    if input_data.is_null() || out_batch.is_null() {
        return ERR_INVALID_POINTERS;
    }

    let input_data = unsafe { &*input_data };

    let embedder = match EmbedderRegistry::get_embedder(embedder_id) {
        Some(e) => e,
        None => return ERR_INVALID_EMBEDDER,
    };

    if !embedder.supports_input_for_model(model_id, input_data.input_type) {
        return ERR_MODEL_NOT_ALLOWED;
    }

    let input = match build_input(input_data) {
        Ok(value) => value,
        Err(value) => return value,
    };

    let result = embedder.embed(model_id, input);

    let (mut flat, n_vectors, dim) = match result {
        Ok((flat, n_vectors, dim)) if n_vectors > 0 && !flat.is_empty() => (flat, n_vectors, dim),
        _ => return ERR_EMBEDDING_FAILED,
    };

    let ptr = flat.as_mut_ptr();
    std::mem::forget(flat);

    unsafe {
        *out_batch = EmbeddingBatch {
            data: ptr,
            n_vectors,
            dim,
        };
    }

    EXIT_SUCCESS
}

/// The C caller guarantees that the strings live for the call duration
unsafe fn get_text_slices<'a>(
    inputs: *const StringSlice,
    n_inputs: usize,
) -> Result<Vec<&'a str>, ()> {
    let slices = unsafe { slice::from_raw_parts(inputs, n_inputs) };
    let mut result = Vec::with_capacity(n_inputs);

    for s in slices {
        if s.ptr.is_null() || s.len == 0 {
            return Err(());
        }

        let bytes = unsafe { slice::from_raw_parts(s.ptr as *const u8, s.len) };
        let text = std::str::from_utf8(bytes).map_err(|_| ())?;
        result.push(text);
    }

    Ok(result)
}

/// Build the input based on input_type
fn build_input(input_data: &'_ InputData) -> Result<Input<'_>, c_int> {
    let input = match input_data.input_type {
        InputType::Text => {
            if input_data.text_data.is_null() || input_data.n_text == 0 {
                return Err(ERR_EMPTY_INPUT);
            }
            let texts = match unsafe { get_text_slices(input_data.text_data, input_data.n_text) } {
                Ok(v) => v,
                Err(_) => return Err(ERR_INVALID_UTF8),
            };
            Input::Texts(texts)
        }
        InputType::Image => {
            if input_data.binary_data.is_null() || input_data.n_binary == 0 {
                return Err(ERR_EMPTY_INPUT);
            }
            let image = unsafe {
                let slice = &*input_data.binary_data;
                if slice.ptr.is_null() || slice.len == 0 {
                    return Err(ERR_EMPTY_INPUT);
                }
                std::slice::from_raw_parts(slice.ptr, slice.len)
            };
            Input::Image(image)
        }
        InputType::Multimodal => {
            let image = if !input_data.binary_data.is_null() && input_data.n_binary > 0 {
                unsafe {
                    let slice = &*input_data.binary_data;
                    if slice.ptr.is_null() || slice.len == 0 {
                        None
                    } else {
                        Some(std::slice::from_raw_parts(slice.ptr, slice.len))
                    }
                }
            } else {
                None
            };

            let texts = if !input_data.text_data.is_null() && input_data.n_text > 0 {
                match unsafe { get_text_slices(input_data.text_data, input_data.n_text) } {
                    Ok(v) => v,
                    Err(_) => return Err(ERR_INVALID_UTF8),
                }
            } else {
                vec![]
            };

            if image.is_none() && texts.is_empty() {
                return Err(ERR_EMPTY_INPUT);
            }

            Input::Multimodal { image, texts }
        }
    };
    Ok(input)
}

/// Free an embedding batch given its pointer
#[unsafe(no_mangle)]
pub extern "C" fn free_embedding_batch(batch: *mut EmbeddingBatch) {
    if batch.is_null() {
        return;
    }

    unsafe {
        let b = &mut *batch;
        if !b.data.is_null() && b.n_vectors > 0 && b.dim > 0 {
            let total = b.n_vectors * b.dim;
            drop(Vec::from_raw_parts(b.data, total, total));
            b.data = std::ptr::null_mut();
            b.n_vectors = 0;
            b.dim = 0;
        }
    }
}
