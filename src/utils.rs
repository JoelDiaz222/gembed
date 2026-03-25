use crate::backends::{Input, InputType};
use crate::get_text_slices;
use anyhow::{anyhow, Result};
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
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(GENERIC_ERROR)
}

/// Build the input based on input_type
pub fn build_input(input_data: &'_ InputData) -> Result<Input<'_>, c_int> {
    let input = match input_data.input_type {
        InputType::Text => {
            if input_data.text_data.is_null() || input_data.n_texts == 0 {
                return Err(ERR_EMPTY_INPUT);
            }
            let texts = match unsafe { get_text_slices(input_data.text_data, input_data.n_texts) } {
                Ok(v) => v,
                Err(e) => return Err(e),
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
                    Err(e) => return Err(e),
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
                Err(e) => return Err(e),
            };
            Input::ImageDirectories(paths)
        }
    };

    Ok(input)
}

pub fn flatten_vectors(vectors: Vec<Vec<f32>>) -> Result<(Vec<f32>, usize, usize)> {
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
pub fn detect_image_format(bytes: &[u8]) -> Result<&'static str> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    // Utils tests

    #[test]
    fn flatten_single_vector() {
        // Arrange
        let input = vec![vec![1.0f32, 2.0, 3.0]];

        // Act
        let (flat, n, dim) = flatten_vectors(input).unwrap();

        // Assert
        assert_eq!(n, 1);
        assert_eq!(dim, 3);
        assert_eq!(flat, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn flatten_multiple_equal_length_vectors() {
        // Arrange
        let input = vec![vec![1.0f32, 2.0], vec![3.0f32, 4.0], vec![5.0f32, 6.0]];

        // Act
        let (flat, n, dim) = flatten_vectors(input).unwrap();

        // Assert
        assert_eq!(n, 3);
        assert_eq!(dim, 2);
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn flatten_empty_outer_vec_returns_zero_n_zero_dim() {
        // Arrange
        let input: Vec<Vec<f32>> = vec![];

        // Act
        let (flat, n, dim) = flatten_vectors(input).unwrap();

        // Assert
        assert_eq!(n, 0);
        assert_eq!(dim, 0);
        assert!(flat.is_empty());
    }

    #[test]
    fn flatten_single_empty_inner_vec() {
        // Arrange
        let input = vec![vec![]];

        // Act
        let (flat, n, dim) = flatten_vectors(input).unwrap();

        // Assert
        assert_eq!(n, 1);
        assert_eq!(dim, 0);
        assert!(flat.is_empty());
    }

    #[test]
    fn flatten_preserves_order() {
        // Arrange
        let a = vec![0.1f32, 0.2, 0.3];
        let b = vec![0.4f32, 0.5, 0.6];

        // Act
        let (flat, _, _) = flatten_vectors(vec![a.clone(), b.clone()]).unwrap();

        // Assert
        assert_eq!(&flat[..3], a.as_slice());
        assert_eq!(&flat[3..], b.as_slice());
    }

    #[test]
    fn flatten_large_batch_capacity_is_exact() {
        // Arrange
        let rows = 128usize;
        let dim = 768usize;
        let input: Vec<Vec<f32>> = (0..rows).map(|_| vec![0.0f32; dim]).collect();

        // Act
        let (flat, n, d) = flatten_vectors(input).unwrap();

        // Assert
        assert_eq!(n, rows);
        assert_eq!(d, dim);
        assert_eq!(flat.len(), rows * dim);
        assert!(flat.capacity() >= rows * dim);
    }

    #[test]
    fn flatten_single_float_values_survive_round_trip() {
        // Arrange
        let v = vec![
            f32::MIN,
            f32::MAX,
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];

        // Act
        let (flat, _, _) = flatten_vectors(vec![v.clone()]).unwrap();

        // Assert
        for (a, b) in v.iter().zip(flat.iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "bit pattern changed: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn error_constants_are_all_distinct() {
        // Arrange
        let codes = [
            EXIT_SUCCESS,
            GENERIC_ERROR,
            ERR_INVALID_POINTER,
            ERR_EMPTY_INPUT,
            ERR_INVALID_UTF8,
            ERR_INVALID_BACKEND,
            ERR_MODEL_NOT_ALLOWED,
        ];

        // Act
        let unique: std::collections::HashSet<i32> = codes.iter().copied().collect();

        // Assert
        assert_eq!(unique.len(), codes.len(), "duplicate error codes detected");
    }

    #[test]
    fn exit_success_is_zero() {
        // Assert
        assert_eq!(EXIT_SUCCESS, 0);
    }

    #[test]
    fn error_codes_are_negative() {
        // Assert
        for &code in &[
            GENERIC_ERROR,
            ERR_INVALID_POINTER,
            ERR_EMPTY_INPUT,
            ERR_INVALID_UTF8,
            ERR_INVALID_BACKEND,
            ERR_MODEL_NOT_ALLOWED,
        ] {
            assert!(code < 0, "error code {} should be negative", code);
        }
    }

    #[test]
    fn ffi_guard_returns_closure_value_on_success() {
        // Act
        let result = ffi_guard(|| EXIT_SUCCESS);

        // Assert
        assert_eq!(result, EXIT_SUCCESS);
    }

    #[test]
    fn ffi_guard_returns_generic_error_on_panic() {
        // Act
        let result = ffi_guard(|| panic!("deliberate panic"));

        // Assert
        assert_eq!(result, GENERIC_ERROR);
    }

    #[test]
    fn ffi_guard_returns_error_code_passed_by_closure() {
        // Act
        let result = ffi_guard(|| ERR_EMPTY_INPUT);

        // Assert
        assert_eq!(result, ERR_EMPTY_INPUT);
    }

    #[test]
    fn ffi_guard_nested_panics_propagate_correctly() {
        // Act
        let result = ffi_guard(|| {
            let _inner = ffi_guard(|| panic!("inner panic"));
            GENERIC_ERROR
        });

        // Assert
        assert_eq!(result, GENERIC_ERROR);
    }

    #[cfg(feature = "embed_anything")]
    mod detect_format {
        use super::detect_image_format;

        fn jpeg_magic() -> Vec<u8> {
            let mut b = vec![0xFF, 0xD8, 0xFF, 0xE0];
            b.resize(16, 0);
            b
        }

        fn png_magic() -> Vec<u8> {
            vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 1]
        }

        fn webp_magic() -> Vec<u8> {
            vec![
                0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50,
            ]
        }

        fn gif87_magic() -> Vec<u8> {
            vec![0x47, 0x49, 0x46, 0x38, b'7', 0x61, 0, 0, 0, 0, 0, 0]
        }

        fn gif89_magic() -> Vec<u8> {
            vec![0x47, 0x49, 0x46, 0x38, b'9', 0x61, 0, 0, 0, 0, 0, 0]
        }

        fn bmp_magic() -> Vec<u8> {
            let mut b = vec![0x42, 0x4D];
            b.resize(16, 0);
            b
        }

        #[test]
        fn detects_jpeg() {
            // Act
            let result = detect_image_format(&jpeg_magic()).unwrap();

            // Assert
            assert_eq!(result, "jpg");
        }

        #[test]
        fn detects_png() {
            // Act
            let result = detect_image_format(&png_magic()).unwrap();

            // Assert
            assert_eq!(result, "png");
        }

        #[test]
        fn detects_webp() {
            // Act
            let result = detect_image_format(&webp_magic()).unwrap();

            // Assert
            assert_eq!(result, "webp");
        }

        #[test]
        fn detects_gif87() {
            // Act
            let result = detect_image_format(&gif87_magic()).unwrap();

            // Assert
            assert_eq!(result, "gif");
        }

        #[test]
        fn detects_gif89() {
            // Act
            let result = detect_image_format(&gif89_magic()).unwrap();

            // Assert
            assert_eq!(result, "gif");
        }

        #[test]
        fn detects_bmp() {
            // Act
            let result = detect_image_format(&bmp_magic()).unwrap();

            // Assert
            assert_eq!(result, "bmp");
        }

        #[test]
        fn returns_error_on_unknown_format() {
            // Arrange
            let unknown = vec![0x00u8; 16];

            // Act
            let result = detect_image_format(&unknown);

            // Assert
            assert!(result.is_err());
        }

        #[test]
        fn returns_error_on_too_short_buffer() {
            // Arrange
            let short = vec![0xFF, 0xD8];

            // Act
            let result = detect_image_format(&short);

            // Assert
            assert!(result.is_err());
        }

        #[test]
        fn returns_error_on_empty_buffer() {
            // Act
            let result = detect_image_format(&[]);

            // Assert
            assert!(result.is_err());
        }
    }

    // build_input tests

    fn text_input_data(slices: &[StringSlice]) -> InputData {
        InputData {
            input_type: InputType::Text,
            binary_data: ptr::null(),
            n_binaries: 0,
            text_data: if slices.is_empty() {
                ptr::null()
            } else {
                slices.as_ptr()
            },
            n_texts: slices.len(),
        }
    }

    fn image_input_data(slices: &[ByteSlice]) -> InputData {
        InputData {
            input_type: InputType::Image,
            binary_data: if slices.is_empty() {
                ptr::null()
            } else {
                slices.as_ptr()
            },
            n_binaries: slices.len(),
            text_data: ptr::null(),
            n_texts: 0,
        }
    }

    #[test]
    fn build_input_text_produces_texts_variant() {
        // Arrange
        let text = b"hello world";
        let slice = StringSlice {
            ptr: text.as_ptr() as *const _,
            len: text.len(),
        };
        let input_data = text_input_data(slice::from_ref(&slice));

        // Act
        let input = build_input(&input_data).unwrap();

        // Assert
        match input {
            Input::Texts(texts) => assert_eq!(texts[0], "hello world"),
            _ => panic!("expected Texts variant"),
        }
    }

    #[test]
    fn build_input_text_with_null_ptr_returns_empty_error() {
        // Arrange
        let input_data = InputData {
            input_type: InputType::Text,
            binary_data: ptr::null(),
            n_binaries: 0,
            text_data: ptr::null(),
            n_texts: 0,
        };

        // Act
        let result = build_input(&input_data);

        // Assert
        assert_eq!(result, Err(ERR_EMPTY_INPUT));
    }

    #[test]
    fn build_input_text_with_n_texts_zero_returns_empty_error() {
        // Arrange
        let text = b"not empty";
        let slice = StringSlice {
            ptr: text.as_ptr() as *const _,
            len: text.len(),
        };
        let input_data = InputData {
            input_type: InputType::Text,
            binary_data: ptr::null(),
            n_binaries: 0,
            text_data: &slice as *const StringSlice,
            n_texts: 0,
        };

        // Act
        let result = build_input(&input_data);

        // Assert
        assert_eq!(result, Err(ERR_EMPTY_INPUT));
    }

    #[test]
    fn build_input_text_with_invalid_utf8_returns_utf8_error() {
        // Arrange
        let bad_utf8 = [0xFF, 0xFE, 0xFD];
        let slice = StringSlice {
            ptr: bad_utf8.as_ptr() as *const _,
            len: bad_utf8.len(),
        };
        let input_data = text_input_data(slice::from_ref(&slice));

        // Act
        let result = build_input(&input_data);

        // Assert
        assert_eq!(result, Err(ERR_INVALID_UTF8));
    }

    #[test]
    fn build_input_text_slice_with_zero_len_returns_empty_error() {
        // Arrange
        let text = b"x";
        let slice = StringSlice {
            ptr: text.as_ptr() as *const _,
            len: 0,
        };
        let input_data = text_input_data(slice::from_ref(&slice));

        // Act
        let result = build_input(&input_data);

        // Assert
        assert_eq!(result, Err(ERR_EMPTY_INPUT));
    }

    #[test]
    fn build_input_image_produces_images_variant() {
        // Arrange
        let bytes = vec![0xFF, 0xD8, 0xFF, 0xE0];
        let slice = ByteSlice {
            ptr: bytes.as_ptr(),
            len: bytes.len(),
        };
        let input_data = image_input_data(slice::from_ref(&slice));

        // Act
        let input = build_input(&input_data).unwrap();

        // Assert
        match input {
            Input::Images(images) => {
                assert_eq!(images.len(), 1);
                assert_eq!(images[0], bytes.as_slice());
            }
            _ => panic!("expected Images variant"),
        }
    }

    #[test]
    fn build_input_image_with_null_binary_data_returns_empty_error() {
        // Arrange
        let input_data = InputData {
            input_type: InputType::Image,
            binary_data: ptr::null(),
            n_binaries: 0,
            text_data: ptr::null(),
            n_texts: 0,
        };

        // Act
        let result = build_input(&input_data);

        // Assert
        assert_eq!(result, Err(ERR_EMPTY_INPUT));
    }

    #[test]
    fn build_input_multimodal_with_both_empty_returns_empty_error() {
        // Arrange
        let input_data = InputData {
            input_type: InputType::Multimodal,
            binary_data: ptr::null(),
            n_binaries: 0,
            text_data: ptr::null(),
            n_texts: 0,
        };

        // Act
        let result = build_input(&input_data);

        // Assert
        assert_eq!(result, Err(ERR_EMPTY_INPUT));
    }

    #[test]
    fn build_input_multimodal_text_only_succeeds() {
        // Arrange
        let text = b"caption";
        let slice = StringSlice {
            ptr: text.as_ptr() as *const _,
            len: text.len(),
        };
        let input_data = InputData {
            input_type: InputType::Multimodal,
            binary_data: ptr::null(),
            n_binaries: 0,
            text_data: &slice as *const StringSlice,
            n_texts: 1,
        };

        // Act
        let input = build_input(&input_data).unwrap();

        // Assert
        match input {
            Input::Multimodal { images, texts } => {
                assert!(images.is_empty());
                assert_eq!(texts.len(), 1);
                assert_eq!(texts[0], "caption");
            }
            _ => panic!("expected Multimodal variant"),
        }
    }

    #[test]
    fn build_input_multimodal_image_only_succeeds() {
        // Arrange
        let bytes = vec![0x89u8, 0x50, 0x4E, 0x47];
        let slice = ByteSlice {
            ptr: bytes.as_ptr(),
            len: bytes.len(),
        };
        let input_data = InputData {
            input_type: InputType::Multimodal,
            binary_data: &slice as *const ByteSlice,
            n_binaries: 1,
            text_data: ptr::null(),
            n_texts: 0,
        };

        // Act
        let input = build_input(&input_data).unwrap();

        // Assert
        match input {
            Input::Multimodal { images, texts } => {
                assert_eq!(images.len(), 1);
                assert!(texts.is_empty());
            }
            _ => panic!("expected Multimodal variant"),
        }
    }

    #[test]
    fn build_input_image_directory_produces_image_directories_variant() {
        // Arrange
        let path = b"/tmp/images";
        let slice = StringSlice {
            ptr: path.as_ptr() as *const _,
            len: path.len(),
        };
        let input_data = InputData {
            input_type: InputType::ImageDirectory,
            binary_data: ptr::null(),
            n_binaries: 0,
            text_data: &slice as *const StringSlice,
            n_texts: 1,
        };

        // Act
        let input = build_input(&input_data).unwrap();

        // Assert
        match input {
            Input::ImageDirectories(paths) => {
                assert_eq!(paths[0], "/tmp/images");
            }
            _ => panic!("expected ImageDirectories variant"),
        }
    }

    #[test]
    fn build_input_image_directory_null_paths_returns_empty_error() {
        // Arrange
        let input_data = InputData {
            input_type: InputType::ImageDirectory,
            binary_data: ptr::null(),
            n_binaries: 0,
            text_data: ptr::null(),
            n_texts: 0,
        };

        // Act
        let result = build_input(&input_data);

        // Assert
        assert_eq!(result, Err(ERR_EMPTY_INPUT));
    }

    #[test]
    fn build_input_multiple_texts_preserves_order() {
        // Arrange
        let a = b"alpha";
        let b_text = b"beta";
        let c = b"gamma";
        let slices = [
            StringSlice {
                ptr: a.as_ptr() as *const _,
                len: a.len(),
            },
            StringSlice {
                ptr: b_text.as_ptr() as *const _,
                len: b_text.len(),
            },
            StringSlice {
                ptr: c.as_ptr() as *const _,
                len: c.len(),
            },
        ];
        let input_data = text_input_data(&slices);

        // Act
        let input = build_input(&input_data).unwrap();

        // Assert
        match input {
            Input::Texts(texts) => {
                assert_eq!(texts, vec!["alpha", "beta", "gamma"]);
            }
            _ => panic!("expected Texts"),
        }
    }
}
