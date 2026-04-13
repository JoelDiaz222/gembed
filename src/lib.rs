mod backends;
mod telemetry;
mod utils;

use telemetry::tlog;

use crate::backends::{BackendRegistry, InputType};
use crate::utils::{
    build_input, ffi_guard, EmbeddingBatch, InputData,
    StringSlice, ERR_EMPTY_INPUT, ERR_INVALID_BACKEND, ERR_INVALID_POINTER, ERR_INVALID_UTF8, ERR_MODEL_NOT_ALLOWED,
    EXIT_SUCCESS, GENERIC_ERROR,
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

        tlog("rs_pre_embed", input_data.n_texts);

        let result = backend.embed(model_id, input);

        tlog("rs_post_embed", 0);

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
) -> Result<Vec<&'a str>, i32> {
    let slices = unsafe { slice::from_raw_parts(inputs, n_inputs) };
    let mut result = Vec::with_capacity(n_inputs);

    for s in slices {
        if s.ptr.is_null() || s.len == 0 {
            return Err(ERR_EMPTY_INPUT);
        }

        let bytes = unsafe { slice::from_raw_parts(s.ptr as *const u8, s.len) };
        let text = std::str::from_utf8(bytes).map_err(|_| ERR_INVALID_UTF8)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::BACKENDS;
    use std::ffi::CString;
    use std::ptr;

    // FFI tests

    #[test]
    fn validate_backend_success() {
        // Arrange
        let name = CString::new("fastembed").unwrap();

        // Act
        let id = validate_backend(name.as_ptr());

        // Assert
        assert_eq!(id, 0); // FASTEMBED_BACKEND_ID
    }

    #[test]
    fn validate_backend_null_ptr_returns_error() {
        // Act
        let id = validate_backend(ptr::null());

        // Assert
        assert_eq!(id, ERR_INVALID_POINTER);
    }

    #[test]
    fn validate_backend_empty_string_returns_empty_error() {
        // Arrange
        let name = CString::new("").unwrap();

        // Act
        let id = validate_backend(name.as_ptr());

        // Assert
        assert_eq!(id, ERR_EMPTY_INPUT);
    }

    #[test]
    fn validate_model_null_ptr_returns_error() {
        // Act
        let id = validate_model(0, ptr::null(), InputType::Text);

        // Assert
        assert_eq!(id, ERR_INVALID_POINTER);
    }

    #[test]
    fn validate_model_success() {
        // Arrange
        let name = CString::new("Xenova/bge-large-en-v1.5").unwrap();

        // Act
        let id = validate_model(0, name.as_ptr(), InputType::Text);

        // Assert
        assert!(id >= 0);
    }

    #[test]
    fn validate_model_unauthorized_input_type_returns_error() {
        // Arrange
        let name = CString::new("Xenova/bge-large-en-v1.5").unwrap();

        // Act
        let id = validate_model(0, name.as_ptr(), InputType::Image);

        // Assert
        assert!(id < 0);
    }

    #[test]
    fn generate_embeddings_null_inputs_returns_error() {
        // Arrange
        let mut batch = EmbeddingBatch {
            data: ptr::null_mut(),
            n_vectors: 0,
            dim: 0,
        };

        // Act
        let rc = generate_embeddings(0, 0, ptr::null(), &mut batch);

        // Assert
        assert_eq!(rc, ERR_INVALID_POINTER);
    }

    #[test]
    fn generate_embeddings_invalid_backend_returns_error() {
        // Arrange
        let input_data = InputData {
            input_type: InputType::Text,
            binary_data: ptr::null(),
            n_binaries: 0,
            text_data: ptr::null(),
            n_texts: 0,
        };
        let mut batch = EmbeddingBatch {
            data: ptr::null_mut(),
            n_vectors: 0,
            dim: 0,
        };

        // Act
        let rc = generate_embeddings(999, 0, &input_data, &mut batch);

        // Assert
        assert_eq!(rc, ERR_INVALID_BACKEND);
    }

    #[test]
    fn free_embedding_batch_handles_null() {
        // Act & Assert
        free_embedding_batch(ptr::null_mut());
    }

    #[test]
    fn free_embedding_batch_clears_struct() {
        // Arrange
        let mut data = vec![1.0f32, 2.0, 3.0];
        let ptr = data.as_mut_ptr();
        std::mem::forget(data);

        let mut batch = EmbeddingBatch {
            data: ptr,
            n_vectors: 1,
            dim: 3,
        };

        // Act
        free_embedding_batch(&mut batch);

        // Assert
        assert!(batch.data.is_null());
        assert_eq!(batch.n_vectors, 0);
        assert_eq!(batch.dim, 0);
    }

    // Integration tests

    #[test]
    fn all_backends_in_slice_are_unique_by_id() {
        // Act
        let mut ids = Vec::new();
        for b in BACKENDS.iter() {
            ids.push(b.id());
        }

        // Assert
        let unique: std::collections::HashSet<i32> = ids.iter().copied().collect();
        assert_eq!(
            ids.len(),
            unique.len(),
            "Duplicate backend IDs found in BACKENDS slice"
        );
    }

    #[test]
    fn all_backends_in_slice_are_unique_by_name() {
        // Act
        let mut names = Vec::new();
        for b in BACKENDS.iter() {
            names.push(b.name());
        }

        // Assert
        let unique: std::collections::HashSet<&str> = names.iter().copied().collect();
        assert_eq!(
            names.len(),
            unique.len(),
            "Duplicate backend names found in BACKENDS slice"
        );
    }
}
