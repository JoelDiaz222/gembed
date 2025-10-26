mod embedders;

use crate::embedders::EmbedderRegistry;
use anyhow::Result;
use std::ffi::CStr;
use std::os::raw::{c_char, c_float, c_int};
use std::slice;

const ERR_INVALID_POINTERS: c_int = -1;
const ERR_EMPTY_INPUT: c_int = -2;
const ERR_INVALID_UTF8: c_int = -3;
const ERR_INVALID_METHOD: c_int = -4;
const ERR_MODEL_NOT_ALLOWED: c_int = -5;
const ERR_EMBEDDING_FAILED: c_int = -6;

#[repr(C)]
pub struct StringSlice {
    pub ptr: *const c_char,
    pub len: usize,
}

#[repr(C)]
pub struct EmbeddingBatch {
    pub data: *mut c_float,
    pub n_vectors: usize,
    pub dim: usize,
}

/// Validates method string and returns method ID
/// Returns -1 if non-existent
#[unsafe(no_mangle)]
pub extern "C" fn validate_embedding_method(method: *const c_char) -> c_int {
    if method.is_null() {
        return -1;
    }

    let method_str = unsafe {
        match CStr::from_ptr(method).to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        }
    };

    EmbedderRegistry::validate_method(method_str).unwrap_or(-1)
}

/// Validates model string for given method and returns model ID
/// Returns -1 if non-existent or unallowed
#[unsafe(no_mangle)]
pub extern "C" fn validate_embedding_model(method_id: c_int, model: *const c_char) -> c_int {
    if model.is_null() {
        return -1;
    }

    let model_str = unsafe {
        match CStr::from_ptr(model).to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        }
    };

    EmbedderRegistry::validate_model(method_id, model_str).unwrap_or(-1)
}

#[unsafe(no_mangle)]
pub extern "C" fn generate_embeddings_from_texts(
    method_id: c_int,
    model_id: c_int,
    inputs: *const StringSlice,
    n_inputs: usize,
    out_batch: *mut EmbeddingBatch,
) -> c_int {
    if inputs.is_null() || out_batch.is_null() {
        return ERR_INVALID_POINTERS;
    }

    let text_slices = unsafe { get_text_slices(inputs, n_inputs) };

    let text_slices = match text_slices {
        Ok(v) if !v.is_empty() => v,
        Ok(_) => return ERR_EMPTY_INPUT,
        Err(_) => return ERR_INVALID_UTF8,
    };

    let embedder = match EmbedderRegistry::get_embedder_by_method_id(method_id) {
        Some(e) => e,
        None => return ERR_INVALID_METHOD,
    };

    if !embedder.supports_model_id(model_id) {
        return ERR_MODEL_NOT_ALLOWED;
    }

    let result = embedder.embed(model_id, text_slices);

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

    0
}

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
