# xn-moshi

Rust implementation of the [Moshi](https://github.com/kyutai-labs/moshi) model family —
the Mimi neural audio codec, the streaming ASR model, and the speech-to-speech model —
built on top of the [xn](https://github.com/LaurentMazare/xn) inference framework.

This crate used to live in the `xn` repository as the `xn-moshi` workspace member.

## Usage

Add the dependency:

```toml
[dependencies]
xn-moshi = "0.2.4"
```

The library is backend-agnostic; pick one through the feature flags, which are
forwarded to `xn`:

| feature       | effect                                          |
| ------------- | ----------------------------------------------- |
| `cuda`        | CUDA backend                                    |
| `metal`       | Metal backend                                   |
| `vulkan`      | Vulkan backend                                  |
| `webgpu`      | WebGPU backend                                  |
| `accelerate`  | Accelerate BLAS on macOS                        |
| `audio`       | audio file IO via `kaudio`, needed for examples |

## Examples

The `moshi` example needs the `audio` feature. It downloads the weights from the
Hugging Face hub on first use.

```bash
# Encode an audio file to Mimi codes and decode it back.
cargo run --release --features audio --example moshi -- audio-to-audio input.wav -o output.wav

# Transcribe an audio file.
cargo run --release --features audio --example moshi -- asr input.wav

# Run the speech-to-speech model.
cargo run --release --features audio --example moshi -- s2s input.wav --voice voice.safetensors --config config.json
```

Add e.g. `--features cuda` to run on a GPU.

The `quantize` example converts an ASR safetensors checkpoint to a quantized GGUF file:

```bash
cargo run --release --example quantize -- asr model.safetensors model-q8.gguf --quant q8_0
```

## Development

Install the formatting pre-commit hook with:

```bash
./scripts/setup-git-hooks.sh
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
