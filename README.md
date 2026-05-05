# Model-Quantizer

This project downloads configured Hugging Face causal LMs, applies a set of manual quantization methods, evaluates raw and quantized variants on benchmark tasks, and cleans up quantized artifacts once benchmarking is complete.

The repo is intentionally focused on two workflows:

1. quantize configured models
2. benchmark them on Hugging Face dataset tasks

`transformers>=4.57` requires `torch>=2.4`, and NVIDIA Blackwell GPUs require a newer CUDA-enabled PyTorch build than the old `2.3.1 + CUDA 12.1` stack. If you are running on Run:ai or another Blackwell-backed environment, use a CUDA 12.8 PyTorch install or newer.

## Usage

List configured options:

```bash
python main.py --list-models
python main.py --list-quantizers
python main.py --list-benchmarks
```

Run quantization for a focused subset:
```bash
python main.py --models mistral-7b phi-3 --quantizers int8_per_tensor int4_grouped --device cuda:0
```
Run quantization for every enabled combination:
```bash
python main.py --all-models --all-quantizers --device auto
```

Remove local quantized artifacts that already have successful full benchmark summaries:
```bash
python main.py --cleanup-benchmarked-quantized
```

## Benchmark Behavior
Benchmark mode evaluates local raw snapshots and local quantized artifacts. The default config uses Hugging Face datasets for:
- `hellaswag` validation
- `mmlu` test with `dataset_config: all`

Scoring method:
- HellaSwag: normalized log-likelihood of each ending continuation
- MMLU: normalized log-likelihood of full answer continuations such as `A. <choice>`

Prompt rendering behavior:
- Benchmarks default to plain text prompt rendering, even when a tokenizer exposes a chat template.
- This avoids model-specific chat wrappers changing multiple-choice likelihood scoring across raw and quantized variants.
- MMLU scores full answer continuations instead of bare labels to reduce label-position bias on base models.
- Set `benchmarks.use_chat_template: true` only if you explicitly want chat-formatted benchmark prompts.

Outputs are written under:
- `artifacts/benchmarks/<model>/<variant>/`
- `artifacts/datasets/` for dataset cache


Each benchmark run writes:
- one summary JSON per `(model, variant, benchmark)`
- one JSONL file with per-example predictions

## Quantized Artifact Cleanup
The default config enables automatic cleanup after benchmarking.
Cleanup happens only when all of the following are true:
- the evaluated variant is quantized
- every enabled benchmark in the config was selected
- every selected benchmark finished successfully
- the run was not limited with `--benchmark-limit`

When cleanup runs, the repo deletes only `models/quantized/<model>/<quantizer>/`.
Raw snapshots under `models/raw/` are preserved.

## Quantization Notes
- [model_quantizer/quantization/int8.py](/mnt/c/Users/aljac/Desktop/Model-Quantizer/model_quantizer/quantization/int8.py:39) implements symmetric int8 weight-only quantization with per-tensor and per-channel scaling.
- [model_quantizer/quantization/int4.py](/mnt/c/Users/aljac/Desktop/Model-Quantizer/model_quantizer/quantization/int4.py:40) implements grouped symmetric int4 quantization with packed 4-bit storage.
- [model_quantizer/artifacts/loader.py](/mnt/c/Users/aljac/Desktop/Model-Quantizer/model_quantizer/artifacts/loader.py:107) reconstructs a dense Transformers model from the saved low-bit artifact for evaluation.

Because the loader dequantizes back into a standard dense model, benchmark accuracy comparisons are valid, but runtime is not representative of specialized low-bit serving stacks such as GPTQ or AWQ runtimes.

