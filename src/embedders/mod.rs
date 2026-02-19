mod embed_anything;
mod fastembed;
mod grpc;

use anyhow::Result;
use linkme::distributed_slice;

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

pub trait Embedder: Send + Sync {
    fn id(&self) -> i32;
    fn name(&self) -> &'static str;
    fn embed(&self, model_id: i32, input: Input) -> Result<(Vec<f32>, usize, usize)>;
    fn model_info(&self, model_name: &str) -> Option<&ModelInfo>;
    fn supports_input_for_model(&self, model_id: i32, input_type: InputType) -> bool;
}

#[distributed_slice]
pub static EMBEDDERS: [&'static dyn Embedder] = [..];

pub struct EmbedderRegistry;

impl EmbedderRegistry {
    pub fn lookup_embedder(id: i32) -> Option<&'static dyn Embedder> {
        EMBEDDERS.iter().find(|e| e.id() == id).copied()
    }

    pub fn lookup_embedder_id(name: &str) -> Option<i32> {
        EMBEDDERS.iter().find(|e| e.name() == name).map(|e| e.id())
    }

    pub fn validate_model_and_input_type(
        embedder_id: i32,
        model_name: &str,
        input_type: InputType,
    ) -> Option<i32> {
        let embedder = Self::lookup_embedder(embedder_id)?;
        let model_info = embedder.model_info(model_name)?;
        model_info
            .supports_input_type(input_type)
            .then_some(model_info.id())
    }
}
