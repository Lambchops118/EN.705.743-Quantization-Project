from __future__ import annotations

import html
import json
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "benchmarks"
OUTPUT_DIR = Path(__file__).resolve().parent
METHOD_ORDER = ["raw", "int8_per_channel", "int8_per_tensor", "int4_grouped"]
METHOD_LABELS = {
    "raw": "raw",
    "int8_per_channel": "int8_per_channel",
    "int8_per_tensor": "int8_per_tensor",
    "int4_grouped": "int4_grouped",
}
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
QUANTIZER_NAMES = ["int8_per_channel", "int8_per_tensor", "int4_grouped"]


def load_accuracy_data() -> dict[str, dict[str, dict[str, float]]]:
    data: dict[str, dict[str, dict[str, float]]] = {"hellaswag": {}, "mmlu": {}}
    latest_files: dict[tuple[str, str, str], Path] = {}

    for path in BENCHMARKS_DIR.glob("*/*/*.summary*.json"):
        parts = path.relative_to(BENCHMARKS_DIR).parts
        model_name, method_name = parts[0], parts[1]
        method_label = METHOD_LABELS.get(method_name)
        if method_label is None:
            continue

        with path.open(encoding="utf-8") as file:
            summary = json.load(file)

        benchmark_name = summary.get("benchmark_name")
        accuracy = summary.get("accuracy")
        if benchmark_name not in data or accuracy is None:
            continue

        key = (benchmark_name, model_name, method_label)
        current = latest_files.get(key)
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            latest_files[key] = path
            data[benchmark_name].setdefault(model_name, {})[method_label] = accuracy

    return data


def load_size_accuracy_data() -> dict[str, list[dict[str, float | str]]]:
    data: dict[str, list[dict[str, float | str]]] = {"hellaswag": [], "mmlu": []}
    latest_files: dict[tuple[str, str, str], Path] = {}

    for path in BENCHMARKS_DIR.glob("*/*/*.summary*.json"):
        with path.open(encoding="utf-8") as file:
            summary = json.load(file)

        benchmark_name = summary.get("benchmark_name")
        model_name = summary.get("model_name")
        quantizer_name = summary.get("quantizer_name")
        accuracy = summary.get("accuracy")
        source_metrics = summary.get("source_metrics") or {}
        size_bytes = source_metrics.get("artifact_quantized_size_bytes")

        if (
            benchmark_name not in data
            or not model_name
            or not quantizer_name
            or quantizer_name not in QUANTIZER_NAMES
            or accuracy is None
            or size_bytes is None
        ):
            continue

        key = (benchmark_name, model_name, quantizer_name)
        current = latest_files.get(key)
        if current is not None and current.stat().st_mtime >= path.stat().st_mtime:
            continue

        latest_files[key] = path

    for (benchmark_name, model_name, quantizer_name), path in latest_files.items():
        with path.open(encoding="utf-8") as file:
            summary = json.load(file)
        data[benchmark_name].append(
            {
                "model_name": model_name,
                "quantizer_name": quantizer_name,
                "accuracy": float(summary["accuracy"]),
                "size_gb": float(summary["source_metrics"]["artifact_quantized_size_bytes"]) / (1024 ** 3),
            }
        )

    return data


def accuracy_to_y(
    accuracy: float,
    top: int,
    bottom: int,
    min_accuracy: float,
    max_accuracy: float,
) -> float:
    scaled = (accuracy - min_accuracy) / (max_accuracy - min_accuracy)
    return bottom - (scaled * (bottom - top))


def value_to_x(
    value: float,
    left: int,
    right_edge: int,
    min_value: float,
    max_value: float,
) -> float:
    scaled = (value - min_value) / (max_value - min_value)
    return left + (scaled * (right_edge - left))


def plot_benchmark(benchmark_name: str, benchmark_data: dict[str, dict[str, float]]) -> None:
    width = 1000
    height = 600
    left = 90
    right = 220
    top = 60
    bottom = 520
    chart_width = width - left - right
    x_step = chart_width / (len(METHOD_ORDER) - 1)
    x_positions = {
        method: left + (index * x_step) for index, method in enumerate(METHOD_ORDER)
    }
    tick_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    min_accuracy = tick_values[0]
    max_accuracy = tick_values[-1]

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="22" font-family="Arial">{" ".join([benchmark_name.upper(), "accuracy by quantization method"])}</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{width - right}" y2="{bottom}" stroke="black" stroke-width="2" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="black" stroke-width="2" />',
        f'<text x="{left - 55}" y="{top - 15}" font-size="16" font-family="Arial">Accuracy</text>',
    ]

    for accuracy in tick_values:
        y = accuracy_to_y(accuracy, top, bottom, min_accuracy, max_accuracy)
        svg.append(
            f'<line x1="{left}" y1="{y}" x2="{width - right}" y2="{y}" stroke="#d0d0d0" stroke-width="1" />'
        )
        svg.append(
            f'<text x="{left - 12}" y="{y + 5}" text-anchor="end" font-size="14" font-family="Arial">{accuracy:.1f}</text>'
        )

    for method, x in x_positions.items():
        svg.append(
            f'<text x="{x}" y="{bottom + 30}" text-anchor="middle" font-size="14" font-family="Arial">{html.escape(method)}</text>'
        )

    legend_x = width - right + 20
    legend_y = 90
    label_entries: list[dict[str, float | str]] = []

    for index, model_name in enumerate(sorted(benchmark_data)):
        scores = benchmark_data[model_name]
        color = COLORS[index % len(COLORS)]
        points = []
        methods = []
        for method in METHOD_ORDER:
            if method in scores:
                methods.append(method)
                points.append(
                    (
                        x_positions[method],
                        accuracy_to_y(
                            scores[method],
                            top,
                            bottom,
                            min_accuracy,
                            max_accuracy,
                        ),
                    )
                )

        if len(points) >= 2:
            point_string = " ".join(f"{x},{y}" for x, y in points)
            svg.append(
                f'<polyline points="{point_string}" fill="none" stroke="{color}" stroke-width="3" />'
            )

        for x, y in points:
            svg.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{color}" />')
        for (x, y), method in zip(points, methods):
            label_entries.append(
                {
                    "x": x,
                    "y": y,
                    "label_y": y - 10,
                    "text": f"{scores[method]:.6f}",
                    "color": color,
                }
            )

        legend_item_y = legend_y + (index * 28)
        svg.append(
            f'<line x1="{legend_x}" y1="{legend_item_y}" x2="{legend_x + 24}" y2="{legend_item_y}" stroke="{color}" stroke-width="3" />'
        )
        svg.append(
            f'<circle cx="{legend_x + 12}" cy="{legend_item_y}" r="5" fill="{color}" />'
        )
        svg.append(
            f'<text x="{legend_x + 35}" y="{legend_item_y + 5}" font-size="14" font-family="Arial">{html.escape(model_name)}</text>'
        )

    min_label_gap = 14
    labels_by_x: dict[float, list[dict[str, float | str]]] = {}
    for entry in label_entries:
        labels_by_x.setdefault(float(entry["x"]), []).append(entry)

    for x in sorted(labels_by_x):
        entries = sorted(labels_by_x[x], key=lambda entry: float(entry["label_y"]))
        previous_label_y = top
        for entry in entries:
            label_y = max(float(entry["label_y"]), previous_label_y)
            entry["label_y"] = label_y
            previous_label_y = label_y + min_label_gap

    for entry in label_entries:
        svg.append(
            f'<text x="{entry["x"]}" y="{entry["label_y"]}" text-anchor="middle" font-size="12" font-family="Arial" fill="{entry["color"]}">{entry["text"]}</text>'
        )

    svg.append("</svg>")
    (OUTPUT_DIR / f"{benchmark_name}_accuracy.svg").write_text(
        "\n".join(svg),
        encoding="utf-8",
    )


def plot_size_vs_accuracy(
    benchmark_name: str,
    points: list[dict[str, float | str]],
) -> None:
    width = 1100
    height = 650
    left = 90
    right = 260
    top = 60
    bottom = 560
    right_edge = width - right
    tick_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    size_values = [float(point["size_gb"]) for point in points]
    min_size = min(size_values)
    max_size = max(size_values)
    padding = max((max_size - min_size) * 0.08, 0.25)
    min_size -= padding
    max_size += padding

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="22" font-family="Arial">{benchmark_name.upper()} quantized size vs accuracy</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right_edge}" y2="{bottom}" stroke="black" stroke-width="2" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="black" stroke-width="2" />',
        f'<text x="{left - 55}" y="{top - 15}" font-size="16" font-family="Arial">Accuracy</text>',
        f'<text x="{(left + right_edge) / 2}" y="{height - 20}" text-anchor="middle" font-size="16" font-family="Arial">Quantized model size (GB)</text>',
    ]

    for accuracy in tick_values:
        y = accuracy_to_y(accuracy, top, bottom, 0.0, 1.0)
        svg.append(
            f'<line x1="{left}" y1="{y}" x2="{right_edge}" y2="{y}" stroke="#d0d0d0" stroke-width="1" />'
        )
        svg.append(
            f'<text x="{left - 12}" y="{y + 5}" text-anchor="end" font-size="14" font-family="Arial">{accuracy:.1f}</text>'
        )

    for index in range(6):
        size_gb = min_size + ((max_size - min_size) * index / 5)
        x = value_to_x(size_gb, left, right_edge, min_size, max_size)
        svg.append(
            f'<line x1="{x}" y1="{top}" x2="{x}" y2="{bottom}" stroke="#e0e0e0" stroke-width="1" />'
        )
        svg.append(
            f'<text x="{x}" y="{bottom + 25}" text-anchor="middle" font-size="14" font-family="Arial">{size_gb:.1f}</text>'
        )

    legend_x = right_edge + 20
    legend_y = 90
    models = sorted({str(point["model_name"]) for point in points})
    color_by_model = {
        model_name: COLORS[index % len(COLORS)] for index, model_name in enumerate(models)
    }

    for index, model_name in enumerate(models):
        item_y = legend_y + (index * 28)
        color = color_by_model[model_name]
        svg.append(f'<circle cx="{legend_x + 12}" cy="{item_y}" r="6" fill="{color}" />')
        svg.append(
            f'<text x="{legend_x + 30}" y="{item_y + 5}" font-size="14" font-family="Arial">{html.escape(model_name)}</text>'
        )

    for point in points:
        model_name = str(point["model_name"])
        quantizer_name = str(point["quantizer_name"])
        x = value_to_x(float(point["size_gb"]), left, right_edge, min_size, max_size)
        y = accuracy_to_y(float(point["accuracy"]), top, bottom, 0.0, 1.0)
        color = color_by_model[model_name]
        svg.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{color}" />')
        svg.append(
            f'<text x="{x + 8}" y="{y - 8}" font-size="11" font-family="Arial" fill="{color}">{html.escape(quantizer_name)}</text>'
        )

    svg.append("</svg>")
    (OUTPUT_DIR / f"{benchmark_name}_size_vs_accuracy.svg").write_text(
        "\n".join(svg),
        encoding="utf-8",
    )


def main() -> None:
    accuracy_data = load_accuracy_data()
    for benchmark_name, benchmark_data in accuracy_data.items():
        plot_benchmark(benchmark_name, benchmark_data)
    size_accuracy_data = load_size_accuracy_data()
    for benchmark_name, points in size_accuracy_data.items():
        if points:
            plot_size_vs_accuracy(benchmark_name, points)


if __name__ == "__main__":
    main()
