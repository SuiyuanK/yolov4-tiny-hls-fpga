#!/usr/bin/env python3
"""Convert raw float32 .bin files to ap_fixed<16,7,AP_RND,AP_SAT> int16 binary files.

Assumptions:
- Input .bin stores little-endian float32 values.
- Output .bin stores little-endian signed int16 values.
- ap_fixed<16,7> means 9 fractional bits, so the scale factor is 2^9 = 512.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path
from typing import Iterable

TOTAL_BITS = 16
INTEGER_BITS = 7
FRACTION_BITS = TOTAL_BITS - INTEGER_BITS
SCALE = 1 << FRACTION_BITS
INT16_MIN = -(1 << (TOTAL_BITS - 1))
INT16_MAX = (1 << (TOTAL_BITS - 1)) - 1


def round_half_away_from_zero(value: float) -> int:
    """Round to nearest integer, with .5 ties away from zero.

    This is a practical match for the common "四舍五入" expectation when converting
    floating-point weights to fixed-point binary data.
    """

    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def quantize_float32_to_int16(values: Iterable[float]) -> bytes:
    out = bytearray()
    pack = struct.pack

    for value in values:
        if math.isnan(value):
            quantized = 0
        elif math.isinf(value):
            quantized = INT16_MAX if value > 0 else INT16_MIN
        else:
            scaled = value * SCALE
            quantized = round_half_away_from_zero(scaled)
            if quantized > INT16_MAX:
                quantized = INT16_MAX
            elif quantized < INT16_MIN:
                quantized = INT16_MIN

        out.extend(pack("<h", quantized))

    return bytes(out)


def convert_file(input_path: Path, output_path: Path) -> None:
    raw = input_path.read_bytes()
    if len(raw) % 4 != 0:
        raise ValueError(f"{input_path} size {len(raw)} is not a multiple of 4 bytes; input may not be float32.")

    count = len(raw) // 4
    floats = struct.iter_unpack("<f", raw)
    output_path.write_bytes(quantize_float32_to_int16(value[0] for value in floats))
    print(f"{input_path} -> {output_path} | {count} float32 values -> {count} int16 values")


def iter_input_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    pattern = "**/*.bin" if recursive else "*.bin"
    return sorted(p for p in path.glob(pattern) if p.is_file())


def build_output_path(input_path: Path, source_root: Path | None, output_dir: Path | None, suffix: str) -> Path:
    if output_dir is None:
        return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")

    if source_root is None:
        return output_dir / f"{input_path.stem}{suffix}{input_path.suffix}"

    relative_path = input_path.relative_to(source_root)
    return output_dir / source_root.name / relative_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def default_source_roots(script_dir: Path) -> list[Path]:
    return [script_dir / "folded_weight", script_dir / "yolo_test"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert float32 .bin files to ap_fixed<16,7,AP_RND,AP_SAT> int16 .bin files.")
    parser.add_argument("inputs", type=Path, nargs="*", help="Input .bin file(s) or directory(ies). Defaults to folded_weight and yolo_test under the script directory.")
    parser.add_argument("-o", "--output-dir", type=Path, default=None, help="Directory for converted files")
    parser.add_argument("--suffix", default="", help="Suffix inserted before .bin in output file names")
    parser.add_argument("--recursive", action="store_true", help="Recursively convert all .bin files under a directory")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    input_paths = args.inputs or default_source_roots(script_dir)
    recursive_dirs = args.recursive or not args.inputs
    if args.output_dir is None and not args.inputs:
        args.output_dir = script_dir / "fixed16_output"

    source_items: list[tuple[Path, Path | None]] = []
    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")

        if input_path.is_dir():
            files = iter_input_files(input_path, recursive_dirs)
            if not files:
                print(f"No .bin files found under {input_path}")
                continue
            source_items.extend((file_path, input_path) for file_path in files)
        else:
            source_items.append((input_path, None))

    if not source_items:
        print("No .bin files found to convert")
        return 0

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for file_path, source_root in source_items:
        output_path = build_output_path(file_path, source_root, args.output_dir, args.suffix)
        if args.output_dir is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        convert_file(file_path, output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
