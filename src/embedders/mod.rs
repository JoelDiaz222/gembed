mod fastembed;
mod grpc;

use anyhow::Result;
use linkme::distributed_slice;

pub trait Embedder: Send + Sync {
    fn method_id(&self) -> i32;
    fn method_name(&self) -> &'static str;
    fn embed(&self, model_id: i32, text_slices: Vec<&str>) -> Result<(Vec<f32>, usize, usize)>;
    fn get_model_id(&self, model: &str) -> Option<i32>;
    fn supports_model_id(&self, model_id: i32) -> bool;
}

#[distributed_slice]
pub static EMBEDDERS: [&'static dyn Embedder] = [..];

pub struct EmbedderRegistry;

impl EmbedderRegistry {
    pub fn get_embedder_by_method_id(method: i32) -> Option<&'static dyn Embedder> {
        EMBEDDERS.iter().find(|e| e.method_id() == method).copied()
    }

    pub fn validate_method(method: &str) -> Option<i32> {
        EMBEDDERS
            .iter()
            .find(|e| e.method_name() == method)
            .map(|e| e.method_id())
    }

    pub fn validate_model(method_id: i32, model: &str) -> Option<i32> {
        Self::get_embedder_by_method_id(method_id)?.get_model_id(model)
    }
}
