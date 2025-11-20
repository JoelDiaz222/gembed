#![cfg(feature = "grpc")]
use crate::embedders::grpc::tei::v1::EmbedBatchRequest;
use crate::embedders::grpc::tei::v1::embed_client::EmbedClient;
use crate::embedders::{EMBEDDERS, Embedder, Input, InputType, ModelInfo};
use anyhow::Result;
use std::os::raw::c_float;
use std::sync::LazyLock;
use std::time::Duration;
use tokio::runtime::Runtime;
use tonic::transport::{Channel, Endpoint};

pub mod tei {
    pub mod v1 {
        tonic::include_proto!("tei.v1");
    }
}

pub static EMBED_METHOD_GRPC_ID: i32 = 1;
pub static EMBED_METHOD_GRPC_NAME: &str = "grpc";

static RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("Failed to build Tokio runtime")
});

static ENDPOINT: LazyLock<Endpoint> = LazyLock::new(|| {
    Channel::from_static("http://127.0.0.1:50051")
        .http2_keep_alive_interval(Duration::from_secs(75))
        .keep_alive_timeout(Duration::from_secs(20))
        .connect_timeout(Duration::from_secs(5))
        .tcp_nodelay(true)
        .http2_adaptive_window(true)
});

thread_local! {
    static CLIENT: std::cell::RefCell<Option<EmbedClient<Channel>>> = std::cell::RefCell::new(None);
}

struct GrpcEmbedder;

impl GrpcEmbedder {
    const MODELS: &'static [ModelInfo] = &[
        ModelInfo::new(
            0,
            "sentence-transformers/all-MiniLM-L6-v2",
            &[InputType::Text],
        ),
        ModelInfo::new(
            1,
            "sentence-transformers/bge-large-en-v1.5",
            &[InputType::Text],
        ),
    ];

    fn get_grpc_client() -> Result<EmbedClient<Channel>> {
        CLIENT.with(|cell| {
            let mut client_opt = cell.borrow_mut();
            if client_opt.is_none() {
                let channel = RUNTIME.block_on(ENDPOINT.connect())?;
                *client_opt = Some(EmbedClient::new(channel));
            }
            Ok(client_opt.as_ref().unwrap().clone())
        })
    }
}

impl Embedder for GrpcEmbedder {
    fn method_id(&self) -> i32 {
        EMBED_METHOD_GRPC_ID
    }

    fn method_name(&self) -> &'static str {
        EMBED_METHOD_GRPC_NAME
    }

    fn embed(&self, model_id: i32, input: Input) -> Result<(Vec<f32>, usize, usize)> {
        let text_slices = match input {
            Input::Texts(texts) => texts,
        };

        let model = Self::MODELS
            .iter()
            .find(|m| m.id == model_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown model ID: {}", model_id))?;

        let mut client = GrpcEmbedder::get_grpc_client()?;

        let response = RUNTIME.block_on(async {
            let request = EmbedBatchRequest {
                inputs: text_slices.iter().map(|&s| s.to_string()).collect(),
                truncate: true,
                normalize: true,
                truncation_direction: 0,
                prompt_name: None,
                dimensions: None,
                model: model.name.to_string(),
            };
            client.embed_batch(tonic::Request::new(request)).await
        })?;

        let embeddings: Vec<Vec<f32>> = response
            .into_inner()
            .embeddings
            .into_iter()
            .map(|e| e.values)
            .collect();

        let n_vectors = embeddings.len();
        let dim = embeddings[0].len();
        let total = n_vectors * dim;
        let mut flat: Vec<c_float> = Vec::with_capacity(total);
        for e in embeddings {
            flat.extend_from_slice(&e);
        }

        Ok((flat, n_vectors, dim))
    }

    fn get_model(&self, model_name: &str) -> Option<&ModelInfo> {
        Self::MODELS.iter().find(|m| m.name == model_name)
    }

    fn supports_model_id(&self, model_id: i32, input_type: InputType) -> bool {
        Self::MODELS
            .iter()
            .find(|m| m.id == model_id)
            .map(|m| m.supports_input_type(input_type))
            .unwrap_or(false)
    }
}

#[linkme::distributed_slice(EMBEDDERS)]
static GRPC: &dyn Embedder = &GrpcEmbedder;
