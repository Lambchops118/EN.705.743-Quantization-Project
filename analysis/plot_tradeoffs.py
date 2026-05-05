from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


BENCHMARKS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "benchmarks"
OUTPUT_DIR = Path(__file__).resolve().parent
BENCHMARK_ORDER = ["hellaswag", "mmlu"]
VARIANT_ORDER = ["int8_per_channel", "int8_per_tensor", "int4_grouped"]
MODEL_ORDER = ["gemma-2-9b", "llama-3-8b", "mistral-7b", "phi-3"]
MODEL_COLORS = {
    "gemma-2-9b": "#2a5b8a",
    "llama-3-8b": "#d17c2f",
    "mistral-7b": "#2e7d5a",
    "phi-3": "#8b3d88",
}
VARIANT_LABELS = {
    "int8_per_channel": "i8-channel",
    "int8_per_tensor": "i8-tensor",
    "int4_grouped": "i4-grouped",
}


@dataclass(frozen=True)
class TradeoffRecord:
    benchmark: str
    model: str
    variant: str
    raw_accuracy: float
    quantized_accuracy: float
    accuracy_delta: float
    accuracy_retained_pct: float
    original_size_gb: float
    quantized_size_gb: float
    gb_saved: float
    compression_ratio: float
    accuracy_points_lost_per_gb_saved: float
    retained_pct_per_gb_saved: float
    retained_pct_per_compression_ratio: float
    load_seconds: float
    normalized_load_seconds: float
    raw_load_seconds: float
    raw_normalized_load_seconds: float
    load_ratio: float
    normalized_load_ratio: float
    evaluation_seconds: float
    raw_evaluation_seconds: float
    evaluation_ratio: float
    examples_per_second: float
    raw_examples_per_second: float
    throughput_ratio: float
    cuda_alloc_gb: float | None
    raw_cuda_alloc_gb: float | None
    cuda_alloc_ratio: float | None
    cuda_reserved_gb: float | None
    raw_cuda_reserved_gb: float | None
    cuda_reserved_ratio: float | None


def load_latest_summaries() -> dict[str, dict[str, dict[str, dict[str, object]]]]:
    latest_paths: dict[tuple[str, str, str], Path] = {}
    summaries: dict[str, dict[str, dict[str, dict[str, object]]]] = {
        benchmark: {} for benchmark in BENCHMARK_ORDER
    }

    for path in BENCHMARKS_DIR.glob("*/*/*.summary*.json"):
        with path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        benchmark = str(summary.get("benchmark_name"))
        model = str(summary.get("model_name"))
        variant = str(summary.get("variant_label"))
        if benchmark not in summaries or not model or not variant:
            continue

        key = (benchmark, model, variant)
        current = latest_paths.get(key)
        if current is not None and current.stat().st_mtime >= path.stat().st_mtime:
            continue

        latest_paths[key] = path
        summaries[benchmark].setdefault(model, {})[variant] = summary

    return summaries


def build_tradeoff_records(
    summaries: dict[str, dict[str, dict[str, dict[str, object]]]],
) -> list[TradeoffRecord]:
    records: list[TradeoffRecord] = []

    for benchmark in BENCHMARK_ORDER:
        for model in MODEL_ORDER:
            model_variants = summaries.get(benchmark, {}).get(model)
            if not model_variants or "raw" not in model_variants:
                continue

            raw = model_variants["raw"]
            raw_accuracy = float(raw["accuracy"])
            raw_load = float(raw["load_seconds"])
            raw_normalized_load = _summary_normalized_load_seconds(raw)
            raw_eval = float(raw["evaluation_seconds"])
            raw_eps = float(raw["examples_per_second"])
            raw_alloc = _optional_float(raw.get("max_cuda_memory_allocated_gb"))
            raw_reserved = _optional_float(raw.get("max_cuda_memory_reserved_gb"))

            for variant in VARIANT_ORDER:
                summary = model_variants.get(variant)
                if summary is None:
                    continue
                metrics = summary["source_metrics"]
                original_size_gb = float(metrics["artifact_original_size_bytes"]) / (1024 ** 3)
                quantized_size_gb = float(metrics["artifact_quantized_size_bytes"]) / (1024 ** 3)
                gb_saved = original_size_gb - quantized_size_gb
                quantized_accuracy = float(summary["accuracy"])
                accuracy_delta = quantized_accuracy - raw_accuracy
                accuracy_retained_pct = (quantized_accuracy / raw_accuracy) * 100.0
                load_seconds = float(summary["load_seconds"])
                normalized_load_seconds = _summary_normalized_load_seconds(summary)
                evaluation_seconds = float(summary["evaluation_seconds"])
                examples_per_second = float(summary["examples_per_second"])
                cuda_alloc = _optional_float(summary.get("max_cuda_memory_allocated_gb"))
                cuda_reserved = _optional_float(summary.get("max_cuda_memory_reserved_gb"))

                records.append(
                    TradeoffRecord(
                        benchmark=benchmark,
                        model=model,
                        variant=variant,
                        raw_accuracy=raw_accuracy,
                        quantized_accuracy=quantized_accuracy,
                        accuracy_delta=accuracy_delta,
                        accuracy_retained_pct=accuracy_retained_pct,
                        original_size_gb=original_size_gb,
                        quantized_size_gb=quantized_size_gb,
                        gb_saved=gb_saved,
                        compression_ratio=float(metrics["compression_ratio"]),
                        accuracy_points_lost_per_gb_saved=(-accuracy_delta / gb_saved),
                        retained_pct_per_gb_saved=accuracy_retained_pct / gb_saved,
                        retained_pct_per_compression_ratio=(
                            accuracy_retained_pct / float(metrics["compression_ratio"])
                        ),
                        load_seconds=load_seconds,
                        normalized_load_seconds=normalized_load_seconds,
                        raw_load_seconds=raw_load,
                        raw_normalized_load_seconds=raw_normalized_load,
                        load_ratio=load_seconds / raw_load,
                        normalized_load_ratio=normalized_load_seconds / raw_normalized_load,
                        evaluation_seconds=evaluation_seconds,
                        raw_evaluation_seconds=raw_eval,
                        evaluation_ratio=evaluation_seconds / raw_eval,
                        examples_per_second=examples_per_second,
                        raw_examples_per_second=raw_eps,
                        throughput_ratio=examples_per_second / raw_eps,
                        cuda_alloc_gb=cuda_alloc,
                        raw_cuda_alloc_gb=raw_alloc,
                        cuda_alloc_ratio=_safe_ratio(cuda_alloc, raw_alloc),
                        cuda_reserved_gb=cuda_reserved,
                        raw_cuda_reserved_gb=raw_reserved,
                        cuda_reserved_ratio=_safe_ratio(cuda_reserved, raw_reserved),
                    )
                )

    return records


def write_metrics_csv(records: list[TradeoffRecord]) -> None:
    output_path = OUTPUT_DIR / "quantization_tradeoff_metrics.csv"
    fieldnames = list(TradeoffRecord.__dataclass_fields__.keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def write_report(records: list[TradeoffRecord]) -> None:
    output_path = OUTPUT_DIR / "quantization_tradeoff_report.md"
    lines: list[str] = [
        "# Quantization Tradeoff Analysis",
        "",
        "This report is generated from the latest benchmark summary for each `(benchmark, model, variant)` combination.",
        "",
        "Interpretation note: these system metrics reflect the repository's dense reconstruction path for quantized artifacts, not a kernel-optimized low-bit inference stack.",
        "",
        "## Compression Efficiency",
        "",
        "Accuracy retention is computed relative to the raw baseline for the same model and benchmark. `acc_delta` is quantized minus raw, so positive values indicate an apparent gain.",
        "",
    ]

    for benchmark in BENCHMARK_ORDER:
        lines.append(f"### {benchmark.title()}")
        lines.append("")
        lines.extend(
            _markdown_table(
                ["model", "variant", "retained_%", "acc_delta", "gb_saved", "compression_ratio", "retained_%_per_GB", "loss_per_GB"],
                [
                    [
                        record.model,
                        record.variant,
                        f"{record.accuracy_retained_pct:.2f}",
                        f"{record.accuracy_delta:+.4f}",
                        f"{record.gb_saved:.2f}",
                        f"{record.compression_ratio:.2f}",
                        f"{record.retained_pct_per_gb_saved:.2f}",
                        f"{record.accuracy_points_lost_per_gb_saved:.4f}",
                    ]
                    for record in _records_for_benchmark(records, benchmark)
                ],
            )
        )
        lines.append("")

        top_retained = max(
            _records_for_benchmark(records, benchmark),
            key=lambda item: item.accuracy_retained_pct,
        )
        top_saved = max(
            _records_for_benchmark(records, benchmark),
            key=lambda item: item.gb_saved,
        )
        lines.append(
            f"- Highest retention: `{top_retained.model} / {top_retained.variant}` at `{top_retained.accuracy_retained_pct:.2f}%` of raw accuracy."
        )
        lines.append(
            f"- Largest storage reduction: `{top_saved.model} / {top_saved.variant}` saving `{top_saved.gb_saved:.2f} GB` at `{top_saved.compression_ratio:.2f}x` compression."
        )
        lines.append("")

    lines.extend(
        [
            "## Systems Tradeoffs",
            "",
            "All ratios below are quantized divided by raw for the same model and benchmark.",
            "",
        ]
    )

    for benchmark in BENCHMARK_ORDER:
        lines.append(f"### {benchmark.title()}")
        lines.append("")
        lines.extend(
            _markdown_table(
                ["variant", "avg_normalized_load_x", "avg_load_x", "avg_eval_x", "avg_throughput_x", "avg_cuda_alloc_x"],
                [
                    [
                        variant,
                        f"{_mean_for(records, benchmark, variant, 'normalized_load_ratio'):.2f}",
                        f"{_mean_for(records, benchmark, variant, 'load_ratio'):.2f}",
                        f"{_mean_for(records, benchmark, variant, 'evaluation_ratio'):.2f}",
                        f"{_mean_for(records, benchmark, variant, 'throughput_ratio'):.2f}",
                        f"{_mean_for(records, benchmark, variant, 'cuda_alloc_ratio'):.2f}",
                    ]
                    for variant in VARIANT_ORDER
                ],
            )
        )
        lines.append("")
        worst_normalized_load = max(
            _records_for_benchmark(records, benchmark),
            key=lambda item: item.normalized_load_ratio,
        )
        worst_load = max(_records_for_benchmark(records, benchmark), key=lambda item: item.load_ratio)
        worst_mem = max(
            _records_for_benchmark(records, benchmark),
            key=lambda item: item.cuda_alloc_ratio or 0.0,
        )
        lines.append(
            f"- Largest normalized reconstruction penalty: `{worst_normalized_load.model} / {worst_normalized_load.variant}` at `{worst_normalized_load.normalized_load_ratio:.2f}x` the raw reconstruction path."
        )
        lines.append(
            f"- Largest load-time penalty: `{worst_load.model} / {worst_load.variant}` at `{worst_load.load_ratio:.2f}x` raw load time."
        )
        lines.append(
            f"- Largest CUDA allocation increase: `{worst_mem.model} / {worst_mem.variant}` at `{(worst_mem.cuda_alloc_ratio or 0.0):.2f}x` raw peak allocation."
        )
        lines.append("")

    lines.extend(
        [
            "## Int8 Scheme Robustness",
            "",
            "Positive deltas mean `int8_per_channel` outperformed `int8_per_tensor`.",
            "",
        ]
    )

    lines.extend(
        _markdown_table(
            ["model", "hellaswag_delta", "mmlu_delta"],
            [
                [
                    model,
                    f"{_int8_delta(records, 'hellaswag', model):+.4f}",
                    f"{_int8_delta(records, 'mmlu', model):+.4f}",
                ]
                for model in MODEL_ORDER
            ],
        )
    )
    lines.append("")
    lines.append(
        f"- Mean channel minus tensor delta on HellaSwag: `{mean(_int8_delta(records, 'hellaswag', model) for model in MODEL_ORDER):+.4f}`."
    )
    lines.append(
        f"- Mean channel minus tensor delta on MMLU: `{mean(_int8_delta(records, 'mmlu', model) for model in MODEL_ORDER):+.4f}`."
    )
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_compression_efficiency(records: list[TradeoffRecord], benchmark: str) -> None:
    points = _records_for_benchmark(records, benchmark)
    width = 1100
    height = 720
    left = 90
    right = 260
    top = 70
    bottom = 600
    right_edge = width - right

    x_values = [record.gb_saved for record in points]
    y_values = [record.accuracy_retained_pct for record in points]
    min_x = min(x_values) - 0.5
    max_x = max(x_values) + 0.5
    min_y = min(min(y_values), 60.0) - 2.0
    max_y = max(max(y_values), 110.0) + 2.0
    if min_y < 0.0:
        min_y = 0.0

    svg: list[str] = [
        _svg_open(width, height),
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="24" font-family="Arial">{" ".join([benchmark.upper(), "compression efficiency"])}</text>',
        f'<text x="{width / 2}" y="56" text-anchor="middle" font-size="14" font-family="Arial" fill="#555">x = GB saved, y = accuracy retained vs raw baseline</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right_edge}" y2="{bottom}" stroke="black" stroke-width="2" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="black" stroke-width="2" />',
        f'<text x="{(left + right_edge) / 2}" y="{height - 26}" text-anchor="middle" font-size="16" font-family="Arial">Storage saved (GB)</text>',
        f'<text x="{left - 62}" y="{top - 18}" font-size="16" font-family="Arial">Retained (%)</text>',
    ]

    for tick in _linear_ticks(min_y, max_y, 6):
        y = _value_to_y(tick, top, bottom, min_y, max_y)
        svg.append(
            f'<line x1="{left}" y1="{y}" x2="{right_edge}" y2="{y}" stroke="#dfdfdf" stroke-width="1" />'
        )
        svg.append(
            f'<text x="{left - 12}" y="{y + 5}" text-anchor="end" font-size="13" font-family="Arial">{tick:.0f}</text>'
        )

    for tick in _linear_ticks(min_x, max_x, 6):
        x = _value_to_x(tick, left, right_edge, min_x, max_x)
        svg.append(
            f'<line x1="{x}" y1="{top}" x2="{x}" y2="{bottom}" stroke="#ececec" stroke-width="1" />'
        )
        svg.append(
            f'<text x="{x}" y="{bottom + 24}" text-anchor="middle" font-size="13" font-family="Arial">{tick:.1f}</text>'
        )

    baseline_y = _value_to_y(100.0, top, bottom, min_y, max_y)
    svg.append(
        f'<line x1="{left}" y1="{baseline_y}" x2="{right_edge}" y2="{baseline_y}" stroke="#999" stroke-width="1.5" stroke-dasharray="6 4" />'
    )
    svg.append(
        f'<text x="{right_edge - 8}" y="{baseline_y - 8}" text-anchor="end" font-size="12" font-family="Arial" fill="#666">raw baseline = 100%</text>'
    )

    for record in points:
        x = _value_to_x(record.gb_saved, left, right_edge, min_x, max_x)
        y = _value_to_y(record.accuracy_retained_pct, top, bottom, min_y, max_y)
        color = MODEL_COLORS[record.model]
        marker = _variant_marker(record.variant)
        svg.extend(_draw_marker(x, y, color, marker))
        svg.append(
            f'<text x="{x + 10}" y="{y - 10}" font-size="11" font-family="Arial" fill="{color}">{html.escape(record.model)} | {html.escape(_short_variant(record.variant))}</text>'
        )

    legend_x = right_edge + 24
    legend_y = 96
    svg.append(
        f'<text x="{legend_x}" y="{legend_y - 20}" font-size="16" font-family="Arial">Models</text>'
    )
    for index, model in enumerate(MODEL_ORDER):
        item_y = legend_y + index * 28
        color = MODEL_COLORS[model]
        svg.append(f'<circle cx="{legend_x + 8}" cy="{item_y - 4}" r="6" fill="{color}" />')
        svg.append(
            f'<text x="{legend_x + 22}" y="{item_y}" font-size="13" font-family="Arial">{html.escape(model)}</text>'
        )

    variant_legend_y = legend_y + len(MODEL_ORDER) * 28 + 40
    svg.append(
        f'<text x="{legend_x}" y="{variant_legend_y - 20}" font-size="16" font-family="Arial">Quantizers</text>'
    )
    for index, variant in enumerate(VARIANT_ORDER):
        item_y = variant_legend_y + index * 32
        svg.extend(_draw_marker(legend_x + 8, item_y - 5, "#444", _variant_marker(variant)))
        svg.append(
            f'<text x="{legend_x + 22}" y="{item_y}" font-size="13" font-family="Arial">{html.escape(_short_variant(variant))}</text>'
        )

    svg.append("</svg>")
    (OUTPUT_DIR / f"compression_efficiency_{benchmark}.svg").write_text("\n".join(svg), encoding="utf-8")


def plot_systems_tradeoffs(records: list[TradeoffRecord], benchmark: str) -> None:
    rows = _records_for_benchmark(records, benchmark)
    width = 1240
    height = 120 + len(rows) * 34
    left = 40
    top = 60
    row_height = 34
    name_width = 250
    col_width = 165
    metrics = [
        ("normalized_load_ratio", "norm load x", "lower"),
        ("load_ratio", "load x", "lower"),
        ("evaluation_ratio", "eval x", "lower"),
        ("throughput_ratio", "ex/s x", "higher"),
        ("cuda_alloc_ratio", "cuda alloc x", "lower"),
    ]

    svg: list[str] = [
        _svg_open(width, height),
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="24" font-family="Arial">{benchmark.upper()} systems tradeoffs vs raw</text>',
        f'<text x="{width / 2}" y="50" text-anchor="middle" font-size="14" font-family="Arial" fill="#555">Green is better than raw, red is worse than raw.</text>',
    ]

    header_y = top
    svg.append(
        f'<rect x="{left}" y="{header_y}" width="{name_width}" height="{row_height}" fill="#efefef" stroke="#cfcfcf" />'
    )
    svg.append(
        f'<text x="{left + 12}" y="{header_y + 22}" font-size="14" font-family="Arial">model / variant</text>'
    )

    for index, (_, label, _) in enumerate(metrics):
        cell_x = left + name_width + index * col_width
        svg.append(
            f'<rect x="{cell_x}" y="{header_y}" width="{col_width}" height="{row_height}" fill="#efefef" stroke="#cfcfcf" />'
        )
        svg.append(
            f'<text x="{cell_x + col_width / 2}" y="{header_y + 22}" text-anchor="middle" font-size="14" font-family="Arial">{html.escape(label)}</text>'
        )

    for row_index, record in enumerate(rows, start=1):
        y = top + row_index * row_height
        fill = "#ffffff" if row_index % 2 else "#fafafa"
        svg.append(
            f'<rect x="{left}" y="{y}" width="{name_width}" height="{row_height}" fill="{fill}" stroke="#d8d8d8" />'
        )
        svg.append(
            f'<text x="{left + 12}" y="{y + 22}" font-size="13" font-family="Arial" fill="{MODEL_COLORS[record.model]}">{html.escape(record.model)} | {html.escape(_short_variant(record.variant))}</text>'
        )

        for col_index, (field_name, _, direction) in enumerate(metrics):
            value = getattr(record, field_name)
            cell_x = left + name_width + col_index * col_width
            cell_fill = _heatmap_fill(value, direction)
            svg.append(
                f'<rect x="{cell_x}" y="{y}" width="{col_width}" height="{row_height}" fill="{cell_fill}" stroke="#d8d8d8" />'
            )
            display = "n/a" if value is None else f"{value:.2f}x"
            svg.append(
                f'<text x="{cell_x + col_width / 2}" y="{y + 22}" text-anchor="middle" font-size="13" font-family="Arial">{display}</text>'
            )

    svg.append("</svg>")
    (OUTPUT_DIR / f"systems_tradeoffs_{benchmark}.svg").write_text("\n".join(svg), encoding="utf-8")


def plot_int8_channel_vs_tensor(records: list[TradeoffRecord]) -> None:
    width = 1080
    height = 640
    left = 90
    right = 140
    top = 70
    bottom = 540
    chart_width = width - left - right
    group_step = chart_width / len(MODEL_ORDER)
    bar_width = 48

    deltas = {
        benchmark: [_int8_delta(records, benchmark, model) for model in MODEL_ORDER]
        for benchmark in BENCHMARK_ORDER
    }
    all_values = [value for values in deltas.values() for value in values]
    min_y = min(all_values + [-0.04]) - 0.003
    max_y = max(all_values + [0.04]) + 0.003

    colors = {"hellaswag": "#2a5b8a", "mmlu": "#c5671c"}

    svg: list[str] = [
        _svg_open(width, height),
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="24" font-family="Arial">int8 per-channel minus per-tensor accuracy delta</text>',
        f'<text x="{width / 2}" y="54" text-anchor="middle" font-size="14" font-family="Arial" fill="#555">Positive values mean channel-wise scaling helped.</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{width - right}" y2="{bottom}" stroke="black" stroke-width="2" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="black" stroke-width="2" />',
        f'<text x="{left - 62}" y="{top - 18}" font-size="16" font-family="Arial">Accuracy delta</text>',
    ]

    for tick in _linear_ticks(min_y, max_y, 7):
        y = _value_to_y(tick, top, bottom, min_y, max_y)
        stroke = "#999999" if abs(tick) < 1e-9 else "#e1e1e1"
        stroke_width = 1.5 if abs(tick) < 1e-9 else 1
        svg.append(
            f'<line x1="{left}" y1="{y}" x2="{width - right}" y2="{y}" stroke="{stroke}" stroke-width="{stroke_width}" />'
        )
        svg.append(
            f'<text x="{left - 12}" y="{y + 5}" text-anchor="end" font-size="13" font-family="Arial">{tick:+.02f}</text>'
        )

    for index, model in enumerate(MODEL_ORDER):
        center_x = left + group_step * index + group_step / 2
        positions = {
            "hellaswag": center_x - bar_width * 0.6,
            "mmlu": center_x + bar_width * 0.6,
        }
        for benchmark in BENCHMARK_ORDER:
            delta = _int8_delta(records, benchmark, model)
            x = positions[benchmark] - bar_width / 2
            bar_top = _value_to_y(max(delta, 0.0), top, bottom, min_y, max_y)
            bar_bottom = _value_to_y(min(delta, 0.0), top, bottom, min_y, max_y)
            height_px = max(1.0, abs(bar_bottom - bar_top))
            rect_y = min(bar_top, bar_bottom)
            svg.append(
                f'<rect x="{x}" y="{rect_y}" width="{bar_width}" height="{height_px}" fill="{colors[benchmark]}" opacity="0.88" />'
            )
            label_y = rect_y - 8 if delta >= 0 else rect_y + height_px + 16
            svg.append(
                f'<text x="{x + bar_width / 2}" y="{label_y}" text-anchor="middle" font-size="11" font-family="Arial" fill="{colors[benchmark]}">{delta:+.4f}</text>'
            )

        svg.append(
            f'<text x="{center_x}" y="{bottom + 26}" text-anchor="middle" font-size="13" font-family="Arial">{html.escape(model)}</text>'
        )

    legend_x = width - right + 10
    legend_y = 120
    for index, benchmark in enumerate(BENCHMARK_ORDER):
        item_y = legend_y + index * 28
        svg.append(
            f'<rect x="{legend_x}" y="{item_y - 12}" width="16" height="16" fill="{colors[benchmark]}" opacity="0.88" />'
        )
        svg.append(
            f'<text x="{legend_x + 26}" y="{item_y + 1}" font-size="13" font-family="Arial">{benchmark}</text>'
        )

    svg.append("</svg>")
    (OUTPUT_DIR / "int8_channel_vs_tensor.svg").write_text("\n".join(svg), encoding="utf-8")


def _records_for_benchmark(records: list[TradeoffRecord], benchmark: str) -> list[TradeoffRecord]:
    return sorted(
        (record for record in records if record.benchmark == benchmark),
        key=lambda item: (MODEL_ORDER.index(item.model), VARIANT_ORDER.index(item.variant)),
    )


def _mean_for(
    records: list[TradeoffRecord],
    benchmark: str,
    variant: str,
    field_name: str,
) -> float:
    values = [
        getattr(record, field_name)
        for record in records
        if record.benchmark == benchmark and record.variant == variant and getattr(record, field_name) is not None
    ]
    return mean(values)


def _int8_delta(records: list[TradeoffRecord], benchmark: str, model: str) -> float:
    channel = next(
        record.quantized_accuracy
        for record in records
        if record.benchmark == benchmark and record.model == model and record.variant == "int8_per_channel"
    )
    tensor = next(
        record.quantized_accuracy
        for record in records
        if record.benchmark == benchmark and record.model == model and record.variant == "int8_per_tensor"
    )
    return channel - tensor


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _svg_open(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )


def _linear_ticks(min_value: float, max_value: float, count: int) -> list[float]:
    if count <= 1:
        return [min_value]
    return [min_value + ((max_value - min_value) * index / (count - 1)) for index in range(count)]


def _value_to_x(value: float, left: int, right_edge: int, min_value: float, max_value: float) -> float:
    if max_value == min_value:
        return float(left)
    scaled = (value - min_value) / (max_value - min_value)
    return left + scaled * (right_edge - left)


def _value_to_y(value: float, top: int, bottom: int, min_value: float, max_value: float) -> float:
    if max_value == min_value:
        return float(bottom)
    scaled = (value - min_value) / (max_value - min_value)
    return bottom - scaled * (bottom - top)


def _variant_marker(variant: str) -> str:
    return {
        "int8_per_channel": "circle",
        "int8_per_tensor": "square",
        "int4_grouped": "diamond",
    }[variant]


def _draw_marker(x: float, y: float, color: str, marker: str) -> list[str]:
    if marker == "circle":
        return [f'<circle cx="{x}" cy="{y}" r="6" fill="{color}" />']
    if marker == "square":
        return [f'<rect x="{x - 6}" y="{y - 6}" width="12" height="12" fill="{color}" />']
    return [f'<polygon points="{x},{y - 7} {x + 7},{y} {x},{y + 7} {x - 7},{y}" fill="{color}" />']


def _short_variant(variant: str) -> str:
    return {
        "int8_per_channel": "i8-channel",
        "int8_per_tensor": "i8-tensor",
        "int4_grouped": "i4-grouped",
    }[variant]


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _summary_normalized_load_seconds(summary: dict[str, object]) -> float:
    value = summary.get("normalized_load_seconds")
    if value is not None:
        return float(value)

    breakdown = summary.get("load_breakdown")
    if isinstance(breakdown, dict):
        total = 0.0
        for key in (
            "manifest_load_seconds",
            "weight_read_seconds",
            "weight_decode_seconds",
            "model_init_seconds",
            "state_dict_apply_seconds",
            "dtype_cast_seconds",
            "device_transfer_seconds",
        ):
            phase = breakdown.get(key)
            if phase is None:
                continue
            total += float(phase)
        if total > 0.0:
            return total

    return float(summary["load_seconds"])


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


def _heatmap_fill(value: float | None, direction: str) -> str:
    if value is None:
        return "#f0f0f0"

    if direction == "higher":
        signed = value - 1.0
    else:
        signed = 1.0 - value

    magnitude = min(abs(signed), 3.0) / 3.0
    alpha = 0.12 + magnitude * 0.38
    if signed >= 0:
        base = (66, 135, 96)
    else:
        base = (184, 82, 82)

    r = int(255 - (255 - base[0]) * alpha)
    g = int(255 - (255 - base[1]) * alpha)
    b = int(255 - (255 - base[2]) * alpha)
    return f"#{r:02x}{g:02x}{b:02x}"


def main() -> None:
    summaries = load_latest_summaries()
    records = build_tradeoff_records(summaries)
    write_metrics_csv(records)
    write_report(records)
    for benchmark in BENCHMARK_ORDER:
        plot_compression_efficiency(records, benchmark)
        plot_systems_tradeoffs(records, benchmark)
    plot_int8_channel_vs_tensor(records)


if __name__ == "__main__":
    main()
