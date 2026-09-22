pub mod asr;
pub mod conditioners;
pub mod conv;
pub mod lm;
pub mod mimi;
pub mod moshi;
pub mod quantization;
pub mod s2s;
pub mod sampling;
pub mod seanet;
pub mod transformer;
pub mod transformer_with_ca;

#[derive(Debug, Copy, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormType {
    /// RMS norm with eps = 1e-5, matching py-inference `rms_norm`.
    RmsNorm,
    /// RMS norm with eps = 1e-8, matching py-inference `rms_norm_f32` (aka
    /// `real_rms_norm_f32` in training configs).
    #[serde(alias = "real_rms_norm_f32", alias = "rms_norm_f32")]
    RmsNormF32,
    LayerNorm,
}

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<u32>;
}
