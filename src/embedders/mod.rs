mod fastembed;
mod grpc;

use anyhow::Result;
use linkme::distributed_slice;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputType {
    Text = 0,
    Image = 1,
}

pub enum Input<'a> {
    Texts(Vec<&'a str>),
}

pub struct ModelInfo {
    pub id: i32,
    pub name: &'static str,
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

    pub fn supports_input_type(&self, input_type: InputType) -> bool {
        self.supported_inputs.contains(&input_type)
    }
}

pub trait Embedder: Send + Sync {
    fn method_id(&self) -> i32;
    fn method_name(&self) -> &'static str;
    fn embed(&self, model_id: i32, input: Input) -> Result<(Vec<f32>, usize, usize)>;
    fn get_model(&self, model_name: &str) -> Option<&ModelInfo>;
    fn supports_model_id(&self, model_id: i32, input_type: InputType) -> bool;
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

    pub fn validate_model(method_id: i32, model: &str, input_type: InputType) -> Option<i32> {
        let embedder = Self::get_embedder_by_method_id(method_id)?;
        let model_info = embedder.get_model(model)?;

        if model_info.supports_input_type(input_type) {
            Some(model_info.id)
        } else {
            None
        }
    }
}
