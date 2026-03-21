mod backends;
mod utils;

use crate::backends::{BackendRegistry, InputType};
use crate::utils::{
    build_input, ffi_guard, EmbeddingBatch, InputData, StringSlice,
    ERR_EMPTY_INPUT, ERR_INVALID_BACKEND, ERR_INVALID_POINTER, ERR_MODEL_NOT_ALLOWED, EXIT_SUCCESS, GENERIC_ERROR,
};
use anyhow::Result;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::slice;

/// Validates backend by name and returns its ID
/// Returns -1 if non-existent
#[unsafe(no_mangle)]
pub extern "C" fn validate_backend(name: *const c_char) -> c_int {
    ffi_guard(|| {
        if name.is_null() {
            return ERR_INVALID_POINTER;
        }

        let name_str = unsafe {
            match CStr::from_ptr(name).to_str() {
                Ok(s) => s,
                Err(_) => return ERR_INVALID_POINTER,
            }
        };

        if name_str.is_empty() {
            return ERR_EMPTY_INPUT;
        }

        BackendRegistry::lookup_backend_id(name_str).unwrap_or(GENERIC_ERROR)
    })
}

/// Validates model string for given backend and input type, returns model ID
/// Returns -1 if non-existent or doesn't support the input type
#[unsafe(no_mangle)]
pub extern "C" fn validate_model(
    backend_id: c_int,
    model_name: *const c_char,
    input_type: InputType,
) -> c_int {
    ffi_guard(|| {
        if model_name.is_null() {
            return ERR_INVALID_POINTER;
        }

        let model_str = unsafe {
            match CStr::from_ptr(model_name).to_str() {
                Ok(s) => s,
                Err(_) => return ERR_INVALID_POINTER,
            }
        };

        if model_str.is_empty() {
            return ERR_EMPTY_INPUT;
        }

        BackendRegistry::validate_model_and_input_type(backend_id, model_str, input_type)
            .unwrap_or(GENERIC_ERROR)
    })
}

/// Embeds the input_data using the backend and model specified by ID,
/// and stores the result in out_batch
#[unsafe(no_mangle)]
pub extern "C" fn generate_embeddings(
    backend_id: c_int,
    model_id: c_int,
    input_data: *const InputData,
    out_batch: *mut EmbeddingBatch,
) -> c_int {
    ffi_guard(|| {
        if input_data.is_null() || out_batch.is_null() {
            return ERR_INVALID_POINTER;
        }

        let input_data = unsafe { &*input_data };

        let backend = match BackendRegistry::lookup_backend(backend_id) {
            Some(e) => e,
            None => return ERR_INVALID_BACKEND,
        };

        if !backend.supports_input_for_model(model_id, input_data.input_type) {
            return ERR_MODEL_NOT_ALLOWED;
        }

        let input = match build_input(input_data) {
            Ok(value) => value,
            Err(value) => return value,
        };

        let result = backend.embed(model_id, input);

        let (mut flat, n_vectors, dim) = match result {
            Ok((flat, n_vectors, dim)) if n_vectors > 0 && !flat.is_empty() => {
                (flat, n_vectors, dim)
            }
            _ => return GENERIC_ERROR,
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
    })
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
