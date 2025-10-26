pub mod fastembed;
pub mod grpc;

use anyhow::Result;
use linkme::distributed_slice;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedMethod {
    #[cfg(feature = "fastembed")]
    FastEmbed = 0,
    #[cfg(feature = "grpc")]
    Grpc = 1,
}

pub trait Embedder: Send + Sync {
    fn method(&self) -> EmbedMethod;
    fn embed(&self, model_id: i32, text_slices: Vec<&str>) -> Result<(Vec<f32>, usize, usize)>;
    fn get_model_id(&self, model: &str) -> Option<i32>;
    fn supports_model_id(&self, model_id: i32) -> bool;
}

#[distributed_slice]
pub static EMBEDDERS: [&'static dyn Embedder] = [..];

pub struct EmbedderRegistry {}

impl EmbedderRegistry {
    pub fn get_embedder_by_method_id(method: i32) -> Option<&'static dyn Embedder> {
        EMBEDDERS
            .iter()
            .find(|e| e.method() as i32 == method)
            .copied()
    }

    pub fn validate_method(method: &str) -> Option<i32> {
        match method {
            #[cfg(feature = "fastembed")]
            "fastembed" => Some(EmbedMethod::FastEmbed as i32),
            #[cfg(feature = "grpc")]
            "remote" => Some(EmbedMethod::Grpc as i32),
            _ => None,
        }
    }

    pub fn validate_model(method_id: i32, model: &str) -> Option<i32> {
        let embedder = Self::get_embedder_by_method_id(method_id)?;
        embedder.get_model_id(model)
    }
}
