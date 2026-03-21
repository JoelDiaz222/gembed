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
struct DynamicModelCache {
    id_to_name: LruCache<i32, String>,
    name_to_id: LruCache<String, i32>,
}

#[cfg(feature = "dynamic_model_loading")]
impl DynamicModelCache {
    fn new() -> Self {
        Self {
            id_to_name: LruCache::new(NonZeroUsize::new(32).unwrap()),
            name_to_id: LruCache::new(NonZeroUsize::new(32).unwrap()),
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
    #[cfg(not(feature = "dynamic_model_loading"))]
    fn model_info(&self, model_name: &str) -> Option<&ModelInfo>;

    #[cfg(not(feature = "dynamic_model_loading"))]
    fn model_info_by_id(&self, model_id: i32) -> Option<&ModelInfo>;

    fn supports_input_for_model(&self, model_id: i32, input_type: InputType) -> bool {
        #[cfg(not(feature = "dynamic_model_loading"))]
        {
            self.model_info_by_id(model_id)
                .map(|m| m.supports_input_type(input_type))
                .unwrap_or(false)
        }
        #[cfg(feature = "dynamic_model_loading")]
        {
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
        backend_id: i32,
        model_name: &str,
        input_type: InputType,
    ) -> Option<i32> {
        #[cfg(not(feature = "dynamic_model_loading"))]
        {
            let backend = Self::lookup_backend(backend_id)?;

            if let Some(model_info) = backend.model_info(model_name) {
                return model_info
                    .supports_input_type(input_type)
                    .then_some(model_info.id());
            }
        }

        #[cfg(feature = "dynamic_model_loading")]
        {
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
