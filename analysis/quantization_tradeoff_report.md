# Quantization Tradeoff Analysis

This report is generated from the latest benchmark summary for each `(benchmark, model, variant)` combination.

Interpretation note: these system metrics reflect the repository's dense reconstruction path for quantized artifacts, not a kernel-optimized low-bit inference stack.

## Compression Efficiency

Accuracy retention is computed relative to the raw baseline for the same model and benchmark. `acc_delta` is quantized minus raw, so positive values indicate an apparent gain.

### Hellaswag

| model | variant | retained_% | acc_delta | gb_saved | compression_ratio | retained_%_per_GB | loss_per_GB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemma-2-9b | int8_per_channel | 99.76 | -0.0017 | 23.25 | 3.08 | 4.29 | 0.0001 |
| gemma-2-9b | int8_per_tensor | 99.84 | -0.0011 | 23.26 | 3.08 | 4.29 | 0.0000 |
| gemma-2-9b | int4_grouped | 97.92 | -0.0145 | 26.89 | 4.57 | 3.64 | 0.0005 |
| llama-3-8b | int8_per_channel | 99.70 | -0.0021 | 6.98 | 1.88 | 14.28 | 0.0003 |
| llama-3-8b | int8_per_tensor | 97.34 | -0.0184 | 6.99 | 1.88 | 13.93 | 0.0026 |
| llama-3-8b | int4_grouped | 100.60 | +0.0042 | 10.27 | 3.19 | 9.80 | -0.0004 |
| mistral-7b | int8_per_channel | 100.03 | +0.0002 | 6.62 | 1.96 | 15.11 | -0.0000 |
| mistral-7b | int8_per_tensor | 100.03 | +0.0002 | 6.62 | 1.96 | 15.10 | -0.0000 |
| mistral-7b | int4_grouped | 98.33 | -0.0122 | 9.73 | 3.58 | 10.11 | 0.0013 |
| phi-3 | int8_per_channel | 101.72 | +0.0124 | 3.46 | 1.95 | 29.37 | -0.0036 |
| phi-3 | int8_per_tensor | 100.07 | +0.0005 | 3.47 | 1.95 | 28.87 | -0.0001 |
| phi-3 | int4_grouped | 99.75 | -0.0018 | 5.09 | 3.51 | 19.59 | 0.0004 |

- Highest retention: `phi-3 / int8_per_channel` at `101.72%` of raw accuracy.
- Largest storage reduction: `gemma-2-9b / int4_grouped` saving `26.89 GB` at `4.57x` compression.

### Mmlu

| model | variant | retained_% | acc_delta | gb_saved | compression_ratio | retained_%_per_GB | loss_per_GB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemma-2-9b | int8_per_channel | 106.28 | +0.0277 | 23.25 | 3.08 | 4.57 | -0.0012 |
| gemma-2-9b | int8_per_tensor | 112.49 | +0.0550 | 23.26 | 3.08 | 4.84 | -0.0024 |
| gemma-2-9b | int4_grouped | 62.96 | -0.1633 | 26.89 | 4.57 | 2.34 | 0.0061 |
| llama-3-8b | int8_per_channel | 102.48 | +0.0074 | 6.98 | 1.88 | 14.67 | -0.0011 |
| llama-3-8b | int8_per_tensor | 91.50 | -0.0254 | 6.99 | 1.88 | 13.09 | 0.0036 |
| llama-3-8b | int4_grouped | 94.43 | -0.0166 | 10.27 | 3.19 | 9.20 | 0.0016 |
| mistral-7b | int8_per_channel | 97.28 | -0.0090 | 6.62 | 1.96 | 14.70 | 0.0014 |
| mistral-7b | int8_per_tensor | 102.07 | +0.0068 | 6.62 | 1.96 | 15.41 | -0.0010 |
| mistral-7b | int4_grouped | 110.43 | +0.0344 | 9.73 | 3.58 | 11.35 | -0.0035 |
| phi-3 | int8_per_channel | 100.91 | +0.0062 | 3.46 | 1.95 | 29.14 | -0.0018 |
| phi-3 | int8_per_tensor | 96.99 | -0.0205 | 3.47 | 1.95 | 27.98 | 0.0059 |
| phi-3 | int4_grouped | 94.84 | -0.0353 | 5.09 | 3.51 | 18.63 | 0.0069 |

- Highest retention: `gemma-2-9b / int8_per_tensor` at `112.49%` of raw accuracy.
- Largest storage reduction: `gemma-2-9b / int4_grouped` saving `26.89 GB` at `4.57x` compression.

## Systems Tradeoffs

All ratios below are quantized divided by raw for the same model and benchmark.

### Hellaswag

| variant | avg_normalized_load_x | avg_load_x | avg_eval_x | avg_throughput_x | avg_cuda_alloc_x |
| --- | --- | --- | --- | --- | --- |
| int8_per_channel | 20.32 | 20.32 | 2.05 | 0.80 | 1.25 |
| int8_per_tensor | 21.58 | 21.58 | 2.08 | 0.78 | 1.25 |
| int4_grouped | 56.96 | 56.96 | 2.05 | 0.80 | 1.25 |

- Largest normalized reconstruction penalty: `llama-3-8b / int4_grouped` at `152.25x` the raw reconstruction path.
- Largest load-time penalty: `llama-3-8b / int4_grouped` at `152.25x` raw load time.
- Largest CUDA allocation increase: `gemma-2-9b / int8_per_channel` at `2.00x` raw peak allocation.

### Mmlu

| variant | avg_normalized_load_x | avg_load_x | avg_eval_x | avg_throughput_x | avg_cuda_alloc_x |
| --- | --- | --- | --- | --- | --- |
| int8_per_channel | 25.00 | 25.00 | 1.95 | 1.23 | 1.25 |
| int8_per_tensor | 24.87 | 24.87 | 1.95 | 1.23 | 1.25 |
| int4_grouped | 69.06 | 69.06 | 1.95 | 1.22 | 1.25 |

- Largest normalized reconstruction penalty: `llama-3-8b / int4_grouped` at `200.92x` the raw reconstruction path.
- Largest load-time penalty: `llama-3-8b / int4_grouped` at `200.92x` raw load time.
- Largest CUDA allocation increase: `gemma-2-9b / int8_per_channel` at `2.00x` raw peak allocation.

## Int8 Scheme Robustness

Positive deltas mean `int8_per_channel` outperformed `int8_per_tensor`.

| model | hellaswag_delta | mmlu_delta |
| --- | --- | --- |
| gemma-2-9b | -0.0006 | -0.0273 |
| llama-3-8b | +0.0163 | +0.0328 |
| mistral-7b | +0.0000 | -0.0158 |
| phi-3 | +0.0120 | +0.0267 |

- Mean channel minus tensor delta on HellaSwag: `+0.0069`.
- Mean channel minus tensor delta on MMLU: `+0.0041`.
