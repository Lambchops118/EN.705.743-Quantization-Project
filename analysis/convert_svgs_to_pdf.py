from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def find_inkscape() -> str | None:
    inkscape = shutil.which("inkscape")
    if inkscape is not None:
        return inkscape

    if sys.platform != "win32":
        return None

    candidates = (
        Path(r"C:\Program Files\Inkscape\bin\inkscape.exe"),
        Path(r"C:\Program Files\Inkscape\inkscape.exe"),
        Path(r"C:\Program Files (x86)\Inkscape\bin\inkscape.exe"),
        Path(r"C:\Program Files (x86)\Inkscape\inkscape.exe"),
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SVG plots in the analysis directory to PDF files for LaTeX."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing SVG files. Defaults to the analysis directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reconvert files even if the PDF is newer than the SVG.",
    )
    return parser.parse_args()


def convert_with_inkscape(svg_path: Path, pdf_path: Path) -> bool:
    inkscape = find_inkscape()
    if inkscape is None:
        return False
    command = [
        inkscape,
        str(svg_path),
        "--export-type=pdf",
        f"--export-filename={pdf_path}",
    ]
    subprocess.run(command, check=True)
    return True


def convert_with_rsvg(svg_path: Path, pdf_path: Path) -> bool:
    if shutil.which("rsvg-convert") is None:
        return False
    command = [
        "rsvg-convert",
        "-f",
        "pdf",
        "-o",
        str(pdf_path),
        str(svg_path),
    ]
    subprocess.run(command, check=True)
    return True


def convert_with_cairosvg(svg_path: Path, pdf_path: Path) -> bool:
    try:
        import cairosvg
    except (ImportError, OSError):
        return False
    try:
        cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
    except OSError:
        return False
    return True


def convert_svg(svg_path: Path, pdf_path: Path) -> str:
    if convert_with_inkscape(svg_path, pdf_path):
        return "inkscape"
    if convert_with_rsvg(svg_path, pdf_path):
        return "rsvg-convert"
    if convert_with_cairosvg(svg_path, pdf_path):
        return "cairosvg"
    raise RuntimeError(
        "No usable SVG-to-PDF converter found. Install Inkscape, librsvg "
        "(rsvg-convert), or CairoSVG with the native Cairo library available."
    )


def should_convert(svg_path: Path, pdf_path: Path, force: bool) -> bool:
    if force or not pdf_path.exists():
        return True
    return svg_path.stat().st_mtime > pdf_path.stat().st_mtime


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    svg_paths = sorted(input_dir.glob("*.svg"))

    if not svg_paths:
        print(f"No SVG files found in {input_dir}")
        return 0

    converted = 0
    skipped = 0
    converter_name: str | None = None

    for svg_path in svg_paths:
        pdf_path = svg_path.with_suffix(".pdf")
        if not should_convert(svg_path, pdf_path, args.force):
            skipped += 1
            print(f"skip {svg_path.name} -> {pdf_path.name}")
            continue

        try:
            used = convert_svg(svg_path, pdf_path)
        except RuntimeError as exc:
            print(f"error {svg_path.name}: {exc}", file=sys.stderr)
            return 1
        converter_name = converter_name or used
        converted += 1
        print(f"ok   {svg_path.name} -> {pdf_path.name} ({used})")

    print(
        f"Finished: {converted} converted, {skipped} skipped"
        + (f", converter={converter_name}" if converter_name else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
