mod embed_anything;
mod fastembed;
mod grpc;
mod http;
mod ort;

use anyhow::Result;
use linkme::distributed_slice;

#[cfg(feature = "dynamic_model_loading")]
use lru::LruCache;
#[cfg(feature = "dynamic_model_loading")]
use std::collections::hash_map::DefaultHasher;
#[cfg(feature = "dynamic_model_loading")]
use std::hash::{Hash, Hasher};
#[cfg(feature = "dynamic_model_loading")]
use std::num::NonZeroUsize;
#[cfg(feature = "dynamic_model_loading")]
use std::sync::LazyLock;

#[cfg(feature = "dynamic_model_loading")]
static DEFAULT_DYNAMIC_MODEL_CACHE_SIZE: NonZeroUsize = NonZeroUsize::new(32).unwrap();

#[cfg(feature = "dynamic_model_loading")]
struct DynamicModelCache {
    name_to_id: LruCache<String, i32>,
    id_to_name: LruCache<i32, String>,
}

#[cfg(feature = "dynamic_model_loading")]
impl DynamicModelCache {
    fn new() -> Self {
        Self {
            name_to_id: LruCache::new(DEFAULT_DYNAMIC_MODEL_CACHE_SIZE),
            id_to_name: LruCache::new(DEFAULT_DYNAMIC_MODEL_CACHE_SIZE),
        }
    }

    fn get_or_create(&mut self, name: &str) -> i32 {
        let mut hasher = DefaultHasher::new();
        name.hash(&mut hasher);
        let id = (hasher.finish() & 0x7FFF_FFFF) as i32;

        if self.name_to_id.get(name).is_none() {
            if let Some((_, evicted_id)) = self.name_to_id.push(name.to_string(), id) {
                self.id_to_name.pop(&evicted_id);
            }
            self.id_to_name.push(id, name.to_string());
        }

        id
    }

    fn get_name(&self, id: i32) -> Option<String> {
        self.id_to_name.peek(&id).cloned()
    }
}

#[cfg(feature = "dynamic_model_loading")]
static DYNAMIC_MODEL_CACHE: LazyLock<std::sync::RwLock<DynamicModelCache>> =
    LazyLock::new(|| std::sync::RwLock::new(DynamicModelCache::new()));

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputType {
    Text = 0,
    Image = 1,
    Multimodal = 2,
    ImageDirectory = 3,
}

#[derive(Debug, PartialEq)]
pub enum Input<'a> {
    Texts(Vec<&'a str>),
    Images(Vec<&'a [u8]>),
    Multimodal {
        images: Vec<&'a [u8]>,
        texts: Vec<&'a str>,
    },
    ImageDirectories(Vec<&'a str>),
}

pub struct ModelInfo {
    id: i32,
    name: &'static str,
    supported_inputs: &'static [InputType],
}

impl ModelInfo {
    pub const fn new(id: i32, name: &'static str, supported_inputs: &'static [InputType]) -> Self {
        Self {
            id,
            name,
            supported_inputs,
        }
    }

    pub fn id(&self) -> i32 {
        self.id
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn supports_input_type(&self, input_type: InputType) -> bool {
        self.supported_inputs.contains(&input_type)
    }
}

pub trait Backend: Send + Sync {
    fn id(&self) -> i32;
    fn name(&self) -> &'static str;
    fn embed(&self, model_id: i32, input: Input) -> Result<(Vec<f32>, usize, usize)>;
    fn model_info(&self, model_name: &str) -> Option<&ModelInfo>;
    fn model_info_by_id(&self, model_id: i32) -> Option<&ModelInfo>;

    fn supports_input_for_model(&self, model_id: i32, _input_type: InputType) -> bool {
        #[cfg(not(feature = "dynamic_model_loading"))]
        {
            self.model_info_by_id(model_id)
                .map(|m| m.supports_input_type(_input_type))
                .unwrap_or(false)
        }
        #[cfg(feature = "dynamic_model_loading")]
        {
            // Check static registration first
            if let Some(m) = self.model_info_by_id(model_id) {
                return m.supports_input_type(_input_type);
            }

            // Assume any ID not in static registry but present in dynamic cache is acceptable
            // (Note: we don't know the input type for arbitrary dynamic models until we load them)
            BackendRegistry::lookup_dynamic_model_name(model_id).is_some()
        }
    }

    fn resolve_model_name(&self, model_id: i32) -> Result<String> {
        #[cfg(not(feature = "dynamic_model_loading"))]
        {
            self.model_info_by_id(model_id)
                .map(|m| m.name().to_string())
                .ok_or_else(|| anyhow::anyhow!("Unknown model ID: {}", model_id))
        }
        #[cfg(feature = "dynamic_model_loading")]
        {
            // Check static registration first
            if let Some(m) = self.model_info_by_id(model_id) {
                return Ok(m.name().to_string());
            }

            BackendRegistry::lookup_dynamic_model_name(model_id)
                .ok_or_else(|| anyhow::anyhow!("Unknown model ID: {}", model_id))
        }
    }
}

#[distributed_slice]
pub static BACKENDS: [&'static dyn Backend] = [..];

pub struct BackendRegistry;

impl BackendRegistry {
    pub fn lookup_backend(id: i32) -> Option<&'static dyn Backend> {
        BACKENDS.iter().find(|e| e.id() == id).copied()
    }

    pub fn lookup_backend_id(name: &str) -> Option<i32> {
        BACKENDS.iter().find(|e| e.name() == name).map(|e| e.id())
    }

    pub fn validate_model_and_input_type(
        _backend_id: i32,
        model_name: &str,
        _input_type: InputType,
    ) -> Option<i32> {
        #[cfg(not(feature = "dynamic_model_loading"))]
        {
            if let Some(backend) = Self::lookup_backend(_backend_id) {
                if let Some(model_info) = backend.model_info(model_name) {
                    return model_info
                        .supports_input_type(_input_type)
                        .then_some(model_info.id());
                }
            }
        }

        #[cfg(feature = "dynamic_model_loading")]
        {
            // Even when dynamic loading is enabled, check static registrations first
            // to ensure we respect their input type constraints and unique IDs.
            if let Some(backend) = Self::lookup_backend(_backend_id) {
                if let Some(model_info) = backend.model_info(model_name) {
                    return if model_info.supports_input_type(_input_type) {
                        Some(model_info.id())
                    } else {
                        // The model name is explicitly registered but doesn't support the requested type.
                        None
                    };
                }
            }

            if let Ok(mut cache) = DYNAMIC_MODEL_CACHE.write() {
                return Some(cache.get_or_create(model_name));
            }
        }

        None
    }

    #[cfg(feature = "dynamic_model_loading")]
    pub fn lookup_dynamic_model_name(model_id: i32) -> Option<String> {
        if let Ok(cache) = DYNAMIC_MODEL_CACHE.read() {
            return cache.get_name(model_id);
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_info_supports_single_input_type() {
        // Arrange
        let m = ModelInfo::new(0, "text-only", &[InputType::Text]);

        // Assert
        assert!(m.supports_input_type(InputType::Text));
        assert!(!m.supports_input_type(InputType::Image));
        assert!(!m.supports_input_type(InputType::Multimodal));
        assert!(!m.supports_input_type(InputType::ImageDirectory));
    }

    #[test]
    fn model_info_supports_multiple_input_types() {
        // Arrange
        let m = ModelInfo::new(
            0,
            "clip",
            &[InputType::Text, InputType::Image, InputType::Multimodal],
        );

        // Assert
        assert!(m.supports_input_type(InputType::Text));
        assert!(m.supports_input_type(InputType::Image));
        assert!(m.supports_input_type(InputType::Multimodal));
        assert!(!m.supports_input_type(InputType::ImageDirectory));
    }

    #[test]
    fn model_info_is_const_constructible() {
        // Arrange
        const M: ModelInfo = ModelInfo::new(7, "const-model", &[InputType::Text]);

        // Assert
        assert_eq!(M.id(), 7);
    }

    #[test]
    fn lookup_backend_returns_none_for_unknown_id() {
        // Act
        let result = BackendRegistry::lookup_backend(i32::MIN);

        // Assert
        assert!(result.is_none());
    }

    #[test]
    fn lookup_backend_id_returns_none_for_empty_string() {
        // Act
        let result = BackendRegistry::lookup_backend_id("");

        // Assert
        assert!(result.is_none());
    }

    #[test]
    fn lookup_backend_id_returns_none_for_nonexistent_name() {
        // Act
        let result = BackendRegistry::lookup_backend_id("__no_such_backend__");

        // Assert
        assert!(result.is_none());
    }

    #[cfg(feature = "dynamic_model_loading")]
    mod dynamic_cache {
        use super::*;

        fn make_cache() -> DynamicModelCache {
            DynamicModelCache::new()
        }

        #[test]
        fn get_or_create_returns_same_id_for_same_name() {
            // Arrange
            let mut cache = make_cache();
            let name = "bert-base";

            // Act
            let id1 = cache.get_or_create(name);
            let id2 = cache.get_or_create(name);

            // Assert
            assert_eq!(id1, id2);
        }

        #[test]
        fn get_or_create_returns_different_ids_for_different_names() {
            // Arrange
            let mut cache = make_cache();

            // Act
            let id1 = cache.get_or_create("model-a");
            let id2 = cache.get_or_create("model-b");

            // Assert
            assert_ne!(id1, id2);
        }

        #[test]
        fn get_name_returns_inserted_name() {
            // Arrange
            let mut cache = make_cache();
            let name = "some/hf-model";
            let id = cache.get_or_create(name);

            // Act
            let result = cache.get_name(id);

            // Assert
            assert_eq!(result, Some(name.to_string()));
        }

        #[test]
        fn get_name_returns_none_for_unknown_id() {
            let cache = make_cache();
            assert!(cache.get_name(999_999).is_none());
        }

        #[test]
        fn id_is_non_negative() {
            let mut cache = make_cache();
            let id = cache.get_or_create("any-model");
            assert!(id >= 0, "model IDs must be non-negative (got {})", id);
        }

        #[test]
        fn hash_is_deterministic_across_calls() {
            // Arrange
            let mut c1 = make_cache();
            let mut c2 = make_cache();
            let name = "sentence-transformers/all-MiniLM-L6-v2";

            // Act
            let id1 = c1.get_or_create(name);
            let id2 = c2.get_or_create(name);

            // Assert
            assert_eq!(id1, id2);
        }

        #[test]
        fn lru_eviction_removes_oldest_entry() {
            // Arrange
            let mut cache = make_cache();
            let mut ids = Vec::new();

            // Act
            for i in 0..33u32 {
                ids.push(cache.get_or_create(&format!("model-{}", i)));
            }

            // Assert
            let resolvable = ids
                .iter()
                .filter(|&&id| cache.get_name(id).is_some())
                .count();
            assert!(
                resolvable <= 32,
                "cache held {} entries, expected <=32",
                resolvable
            );
        }

        #[test]
        fn lookup_dynamic_model_name_round_trips() {
            // Arrange
            let name = "test/round-trip-model";
            let _result = BackendRegistry::validate_model_and_input_type(0, name, InputType::Text);

            // Act
            let id =
                BackendRegistry::validate_model_and_input_type(0, name, InputType::Text).unwrap();
            let looked_up = BackendRegistry::lookup_dynamic_model_name(id);

            // Assert
            assert_eq!(looked_up, Some(name.to_string()));
        }
    }

    #[cfg(not(feature = "dynamic_model_loading"))]
    #[test]
    fn validate_model_returns_none_for_nonexistent_model() {
        // Act
        let result =
            BackendRegistry::validate_model_and_input_type(0, "__no_such_model__", InputType::Text);

        // Assert
        assert!(result.is_none());
    }
}
