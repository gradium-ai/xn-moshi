use crate::transformer::{self, BatchedTransformerState, Config as TransformerConfig, Norm};
use xn::nn::{Embedding, Linear, var_builder::Path};
use xn::streaming::StreamMask;
use xn::{BackendQ, Result, Tensor};

// ============================================================================
// Config
// ============================================================================

#[derive(Debug, Clone)]
pub struct Config {
    pub transformer: TransformerConfig,
    pub text_in_vocab_size: usize,
    pub text_out_vocab_size: usize,
    pub audio_vocab_size: usize,
    pub audio_codebooks: usize,
    pub extra_heads: Option<ExtraHeadsConfig>,
}

#[derive(Debug, Clone)]
pub struct ExtraHeadsConfig {
    pub num_heads: usize,
    pub dim: usize,
    /// Layer(s) of the main transformer providing the input of the extra
    /// heads. With more than one layer, representations are mixed with a
    /// [`LayerMixer`]. If `None`, the heads read the final (post-norm)
    /// transformer output (legacy behavior).
    pub from_layer: Option<Vec<usize>>,
    /// Inner dimension of the extra heads' residual blocks (defaults to the
    /// model dimension). Requires `residual_blocks > 0`.
    pub hidden_dim: Option<usize>,
    /// If false, the mixer's layer-norms have no learnt affine parameters.
    pub mixer_affine: bool,
    /// If > 0, each extra head is this many pre-norm residual MLP blocks
    /// followed by a linear projection, instead of a single linear.
    pub residual_blocks: usize,
}

impl Config {
    pub fn asr_v0_1_1b() -> Self {
        let transformer = TransformerConfig {
            d_model: 2048,
            num_heads: 16,
            num_layers: 16,
            dim_feedforward: 2048 * 4,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 750,
            max_period: 100_000.0,
            use_conv_block: false,
            gating: Some(crate::seanet::Activation::Silu),
            norm: crate::NormType::RmsNormF32,
            positional_embedding: transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            kv_repeat: 1,
            head_dim: None,
            final_norm: None,
            proj_bias: false,
        };
        Self {
            transformer,
            audio_vocab_size: 2049,
            text_in_vocab_size: 48001,
            text_out_vocab_size: 48000,
            audio_codebooks: 8,
            extra_heads: None,
        }
    }

    pub fn stt_2_6b() -> Self {
        let transformer = TransformerConfig {
            d_model: 2048,
            num_heads: 32,
            num_layers: 48,
            dim_feedforward: 8448, // 2048 * 4.125
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 375,
            max_period: 100_000.0,
            use_conv_block: false,
            gating: Some(crate::seanet::Activation::Silu),
            norm: crate::NormType::RmsNormF32,
            positional_embedding: transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            kv_repeat: 1,
            head_dim: None,
            final_norm: None,
            proj_bias: false,
        };
        Self {
            transformer,
            audio_vocab_size: 2049,
            text_in_vocab_size: 4001,
            text_out_vocab_size: 4000,
            audio_codebooks: 32,
            extra_heads: None,
        }
    }

    pub fn asr_300m_202501() -> Self {
        let transformer = TransformerConfig {
            d_model: 1024,
            num_heads: 8,
            num_layers: 16,
            dim_feedforward: 1024 * 4,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: false,
            layer_scale: None,
            context: 750,
            max_period: 100_000.0,
            use_conv_block: false,
            gating: Some(crate::seanet::Activation::Silu),
            norm: crate::NormType::RmsNormF32,
            positional_embedding: transformer::PositionalEmbedding::Rope,
            conv_layout: false,
            kv_repeat: 1,
            head_dim: None,
            final_norm: None,
            proj_bias: false,
        };
        Self {
            transformer,
            audio_vocab_size: 2049,
            text_in_vocab_size: 48001,
            text_out_vocab_size: 48000,
            audio_codebooks: 32,
            extra_heads: None,
        }
    }
}

// ============================================================================
// Extra heads
// ============================================================================

/// Mixes representations coming from several layers with input-dependent (per
/// frame) weights: each input is layer-normalized, a scorer shared across
/// layers computes a logit for each layer's normed features, added to a learnt
/// per-layer bias, and the inputs are summed with the softmax of those logits.
struct LayerMixer<T: xn::WithDTypeF, B: xn::Backend> {
    norms: Vec<xn::nn::LayerNorm<T, B>>,
    weights: Tensor<T, B>, // (num_inputs,)
    scorer_in: Linear<T, B>,
    scorer_out: Linear<T, B>,
}

impl<T: xn::WithDTypeF, B: xn::Backend> LayerMixer<T, B> {
    fn load(vb: &Path<B>, num_inputs: usize, dim: usize, affine_norm: bool) -> Result<Self> {
        let mut norms = Vec::with_capacity(num_inputs);
        for i in 0..num_inputs {
            let norm = if affine_norm {
                xn::nn::LayerNorm::load(vb.pp("norms").pp(i), dim, 1e-5)?
            } else {
                let weight = Tensor::full(T::from_f32(1.0), (dim,), vb.device())?;
                let bias = Tensor::zeros((dim,), vb.device())?;
                xn::nn::LayerNorm::new(weight, bias, 1e-5)?
            };
            norms.push(norm);
        }
        let weights = vb.tensor("weights", (num_inputs,))?;
        let hidden = usize::max(dim / 32, 32);
        let scorer_in = Linear::load_b(vb.pp("scorer").pp(0), dim, hidden)?;
        let scorer_out = Linear::load(vb.pp("scorer").pp(2), hidden, 1)?;
        Ok(Self { norms, weights, scorer_in, scorer_out })
    }

    fn forward(&self, inputs: &[&Tensor<T, B>]) -> Result<Tensor<T, B>> {
        if inputs.len() != self.norms.len() {
            xn::bail!("layer-mixer input mismatch: {} != {}", inputs.len(), self.norms.len())
        }
        let mut normed = Vec::with_capacity(inputs.len());
        let mut logits = Vec::with_capacity(inputs.len());
        for (norm, xs) in self.norms.iter().zip(inputs.iter()) {
            let xs = norm.forward(xs)?;
            // (B, T, 1) logit for this layer.
            let logit = self.scorer_out.forward(&self.scorer_in.forward(&xs)?.tanh()?)?;
            normed.push(xs);
            logits.push(logit);
        }
        let logits = Tensor::cat(&logits.iter().collect::<Vec<_>>(), 2)?; // (B, T, N)
        let weights = logits.broadcast_add(&self.weights)?.softmax()?;
        // Weighted sum as a single batched matmul:
        // (B, T, 1, N) @ (B, T, N, C) -> (B, T, 1, C)
        let stacked = Tensor::stack(&normed.iter().collect::<Vec<_>>(), 2)?;
        let (b, t, n) = weights.dims3()?;
        let ys = weights.reshape((b, t, 1, n))?.matmul(&stacked)?;
        ys.reshape((b, t, stacked.dim(xn::D::Minus1)?))
    }
}

/// Pre-norm residual MLP block: `x + linear_out(GELU(linear_in(norm(x))))`.
struct ResidualMLPBlock<T: xn::WithDTypeF, B: xn::Backend> {
    norm: xn::nn::LayerNorm<T, B>,
    linear_in: Linear<T, B>,
    linear_out: Linear<T, B>,
}

impl<T: xn::WithDTypeF, B: xn::Backend> ResidualMLPBlock<T, B> {
    fn load(vb: &Path<B>, dim: usize, hidden: usize) -> Result<Self> {
        let norm = xn::nn::LayerNorm::load(vb.pp("norm"), dim, 1e-5)?;
        let linear_in = Linear::load(vb.pp("mlp").pp(0), dim, hidden)?;
        let linear_out = Linear::load(vb.pp("mlp").pp(2), hidden, dim)?;
        Ok(Self { norm, linear_in, linear_out })
    }

    fn forward(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        let ys = self.norm.forward(xs)?;
        let ys = self.linear_in.forward(&ys)?.gelu_erf()?;
        let ys = self.linear_out.forward(&ys)?;
        xs.add(&ys)
    }
}

/// An extra linear probe (e.g. for VAD), either a single linear or a stack of
/// pre-norm residual MLP blocks followed by a linear projection.
enum ExtraHead<T: xn::WithDTypeF, B: xn::Backend> {
    Linear(Linear<T, B>),
    Mlp { blocks: Vec<ResidualMLPBlock<T, B>>, linear: Linear<T, B> },
}

impl<T: xn::WithDTypeF, B: xn::Backend> ExtraHead<T, B> {
    fn load(vb: &Path<B>, d_model: usize, cfg: &ExtraHeadsConfig) -> Result<Self> {
        if cfg.residual_blocks > 0 {
            let hidden = cfg.hidden_dim.unwrap_or(d_model);
            let mut blocks = Vec::with_capacity(cfg.residual_blocks);
            for i in 0..cfg.residual_blocks {
                blocks.push(ResidualMLPBlock::load(&vb.pp(i), d_model, hidden)?);
            }
            let linear = Linear::load(vb.pp(cfg.residual_blocks), d_model, cfg.dim)?;
            Ok(Self::Mlp { blocks, linear })
        } else {
            if cfg.hidden_dim.is_some() {
                xn::bail!("extra-heads hidden_dim requires residual_blocks > 0")
            }
            Ok(Self::Linear(Linear::load(vb, d_model, cfg.dim)?))
        }
    }

    fn forward(&self, xs: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        match self {
            Self::Linear(linear) => linear.forward(xs),
            Self::Mlp { blocks, linear } => {
                let mut xs = xs.clone();
                for block in blocks.iter() {
                    xs = block.forward(&xs)?;
                }
                linear.forward(&xs)
            }
        }
    }
}

// ============================================================================
// State
// ============================================================================

pub struct LmState<Q: BackendQ> {
    pub model: std::sync::Arc<LmModel<Q>>,
    pub transformer: BatchedTransformerState<Q::T, Q::B>,
}

// ============================================================================
// LmModel
// ============================================================================

pub struct LmModel<Q: BackendQ> {
    transformer: transformer::BatchedTransformer<Q>,
    text_emb: Embedding<Q::T, Q::B>, // (text_in_vocab_size, d_model)
    audio_embs: Vec<Embedding<Q::T, Q::B>>, // each (audio_vocab_size, d_model)
    text_linear: Q::LinearQ,
    out_norm: Norm<Q::T, Q::B>,
    extra_heads: Vec<ExtraHead<Q::T, Q::B>>,
    extra_heads_mixer: Option<LayerMixer<Q::T, Q::B>>,
    extra_heads_from_layer: Option<Vec<usize>>,
    audio_vocab_size: usize,
    text_in_vocab_size: usize,
    text_out_vocab_size: usize,
}

impl<Q: BackendQ> LmModel<Q> {
    pub fn load(vb: &Path<Q::B>, cfg: &Config) -> Result<Self> {
        let d_model = cfg.transformer.d_model;

        let text_emb = Embedding::load(vb.pp("text_emb"), cfg.text_in_vocab_size, d_model)?;
        let out_norm = Norm::load(vb.pp("out_norm"), d_model, cfg.transformer.norm)?;
        let text_linear = Linear::load(vb.pp("text_linear"), d_model, cfg.text_out_vocab_size)?;
        let text_linear = Q::from_linear(text_linear)?;

        let transformer =
            transformer::BatchedTransformer::load(&vb.pp("transformer"), &cfg.transformer)?;

        let vb_e = vb.pp("emb");
        let mut audio_embs = Vec::with_capacity(cfg.audio_codebooks);
        for i in 0..cfg.audio_codebooks {
            let emb = Embedding::load(vb_e.pp(i), cfg.audio_vocab_size, d_model)?;
            audio_embs.push(emb);
        }

        let mut extra_heads = vec![];
        let mut extra_heads_mixer = None;
        let mut extra_heads_from_layer = None;
        if let Some(eh_cfg) = &cfg.extra_heads {
            for i in 0..eh_cfg.num_heads {
                extra_heads.push(ExtraHead::load(&vb.pp("extra_heads").pp(i), d_model, eh_cfg)?);
            }
            if let Some(layers) = &eh_cfg.from_layer {
                if layers.is_empty() {
                    xn::bail!("extra-heads from_layer cannot be empty")
                }
                if let Some(&l) = layers.iter().find(|&&l| l >= cfg.transformer.num_layers) {
                    xn::bail!(
                        "extra-heads from_layer {l} is out of range ({} layers)",
                        cfg.transformer.num_layers
                    )
                }
                if layers.len() > 1 {
                    extra_heads_mixer = Some(LayerMixer::load(
                        &vb.pp("extra_heads_mixer"),
                        layers.len(),
                        d_model,
                        eh_cfg.mixer_affine,
                    )?);
                }
                extra_heads_from_layer = Some(layers.clone());
            }
        }

        Ok(Self {
            transformer,
            text_emb,
            audio_embs,
            text_linear,
            out_norm,
            extra_heads,
            extra_heads_mixer,
            extra_heads_from_layer,
            audio_vocab_size: cfg.audio_vocab_size,
            text_in_vocab_size: cfg.text_in_vocab_size,
            text_out_vocab_size: cfg.text_out_vocab_size,
        })
    }

    pub fn init_state(self: &std::sync::Arc<Self>, batch_size: usize) -> Result<LmState<Q>> {
        Ok(LmState { model: self.clone(), transformer: self.transformer.init_state(batch_size)? })
    }

    pub fn audio_pad_token(&self) -> u32 {
        self.audio_vocab_size as u32 - 1
    }

    pub fn text_start_token(&self) -> u32 {
        self.text_in_vocab_size as u32 - 1
    }

    pub fn text_out_vocab_size(&self) -> usize {
        self.text_out_vocab_size
    }

    pub fn in_audio_codebooks(&self) -> usize {
        self.audio_embs.len()
    }

    pub fn device(&self) -> &Q::B {
        self.text_emb.device()
    }
}

impl<Q: BackendQ> LmState<Q> {
    /// Forward pass returning (text_logits, extra_head_input).
    ///
    /// The second value is the representation the extra heads should read:
    /// intermediate layer(s), possibly mixed, when `from_layer` is set, and
    /// the final (post-norm) transformer output otherwise.
    ///
    /// `text_ids`: token IDs per batch element (batch_size,), or None for zeros.
    /// `audio_ids`: per-codebook token IDs, each (batch_size,) or None to skip.
    #[allow(clippy::type_complexity)]
    pub fn forward(
        &mut self,
        text_ids: Option<&[u32]>,
        audio_ids: &[Option<&[u32]>],
        mask: &StreamMask,
        condition: Option<&Tensor<Q::T, Q::B>>,
    ) -> Result<(Tensor<Q::T, Q::B>, Tensor<Q::T, Q::B>)> {
        use xn::ModuleT;
        let model = &self.model;
        // Text embedding: forward gives (batch, d_model), unsqueeze to (batch, 1, d_model)
        let mut emb = match text_ids {
            Some(ids) => {
                let ids_t = Tensor::from_vec(
                    ids.iter().map(|&x| x as i64).collect(),
                    ids.len(),
                    model.device(),
                )?;
                model.text_emb.forward(&ids_t)?.unsqueeze(1)?
            }
            None => {
                let d_model = model.text_emb.hidden_size();
                let batch_size = self.transformer.batch_size();
                Tensor::zeros((batch_size, 1, d_model), model.device())?
            }
        };

        // Audio embeddings
        for (audio_emb, audio_ids) in model.audio_embs.iter().zip(audio_ids.iter()) {
            if let Some(ids) = audio_ids {
                let ids_t = Tensor::from_vec(
                    ids.iter().map(|&x| x as i64).collect(),
                    ids.len(),
                    model.device(),
                )?;
                let e = audio_emb.forward(&ids_t)?.unsqueeze(1)?;
                emb = emb.add(&e)?;
            }
        }

        // Conditioning
        if let Some(cond) = condition {
            emb = emb.add(cond)?;
        }

        // Transformer
        let (ys, extra_head_input) = match &model.extra_heads_from_layer {
            Some(layers) => {
                let (ys, intermediates) = model.transformer.forward_with_intermediates(
                    &emb,
                    &mut self.transformer,
                    mask,
                )?;
                let extra_head_input = match &model.extra_heads_mixer {
                    Some(mixer) => {
                        let inputs: Vec<&Tensor<Q::T, Q::B>> =
                            layers.iter().map(|&l| &intermediates[l]).collect();
                        mixer.forward(&inputs)?
                    }
                    None => intermediates[layers[0]].clone(),
                };
                (ys, Some(extra_head_input))
            }
            None => (model.transformer.forward(&emb, &mut self.transformer, mask)?, None),
        };
        let ys = model.out_norm.forward(&ys)?;
        let logits = model.text_linear.forward(&ys)?;
        // Legacy behavior: the extra heads read the post-norm transformer output.
        let extra_head_input = extra_head_input.unwrap_or(ys);
        Ok((logits, extra_head_input))
    }

    /// Compute extra head outputs from the extra-head input returned by
    /// `forward`.
    pub fn extra_heads(&self, ys: &Tensor<Q::T, Q::B>) -> Result<Vec<Tensor<Q::T, Q::B>>> {
        let mut results = Vec::with_capacity(self.model.extra_heads.len());
        for head in &self.model.extra_heads {
            results.push(head.forward(ys)?);
        }
        Ok(results)
    }

    pub fn reset_batch_idx(&mut self, batch_idx: usize) -> Result<()> {
        self.transformer.reset_batch_idx(batch_idx)
    }
}
