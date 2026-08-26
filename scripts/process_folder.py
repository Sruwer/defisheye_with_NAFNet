#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import FisheyePanoramaPipeline
from src.utils import load_config, save_json


DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _extensions(values: list[str]) -> set[str]:
    return {value.lower() if value.startswith(".") else f".{value.lower()}" for value in values}


def _images_in(
    input_dir: Path,
    output_dir: Path,
    recursive: bool,
    extensions: set[str],
) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    output_resolved = output_dir.resolve()
    images = []
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        # Do not consume results again when a recursive output directory lives
        # below the input directory.
        if path.resolve().is_relative_to(output_resolved):
            continue
        images.append(path)
    return sorted(images, key=lambda path: str(path.relative_to(input_dir)))


def _output_path(input_path: Path, input_dir: Path, output_dir: Path) -> Path:
    return (output_dir / input_path.relative_to(input_dir)).with_suffix(".png")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Process an image folder while keeping geometry LUTs in RAM"
    )
    parser.add_argument("--input-dir", "--input", required=True, type=Path)
    parser.add_argument("--output-dir", "--output", required=True, type=Path)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--restoration", choices=("none", "nafnet"))
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--save-raw-panorama", action="store_true")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        metavar="EXT",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")
    if input_dir == output_dir:
        parser.error("Input and output directories must be different")

    image_paths = _images_in(
        input_dir, output_dir, args.recursive, _extensions(args.extensions)
    )
    if not image_paths:
        parser.error(f"No supported images found in: {input_dir}")

    try:
        import cv2
    except ImportError:
        raise SystemExit("OpenCV is required: pip install -e .")

    config = load_config(args.config)
    output_dir.mkdir(parents=True, exist_ok=True)

    # This is deliberately outside the frame loop. Both the projection LUTs
    # and the panorama LUT are loaded/computed exactly once and then retained
    # by these two long-lived objects for the entire dataset.
    pipeline = FisheyePanoramaPipeline(config, args.restoration)
    cache_report = pipeline.prepare_geometry_cache()
    print(f"Geometry cache prepared once: {cache_report}")
    print(f"Found {len(image_paths)} image(s)")

    started = perf_counter()
    processed = 0
    skipped = 0
    failures: list[dict[str, str]] = []
    timing_sums: dict[str, float] = {}

    for index, input_path in enumerate(image_paths, start=1):
        destination = _output_path(input_path, input_dir, output_dir)
        if destination.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(image_paths)}] skip {input_path.name}")
            continue

        try:
            image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("OpenCV could not decode the image")
            result = pipeline.process(
                image, include_raw_panorama=args.save_raw_panorama
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination), result.panorama):
                raise OSError(f"Could not write {destination}")

            if args.save_raw_panorama and pipeline.restoration_mode != "none":
                assert result.panorama_raw is not None
                raw_destination = output_dir / "raw_without_restoration" / input_path.relative_to(input_dir)
                raw_destination = raw_destination.with_suffix(".png")
                raw_destination.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(raw_destination), result.panorama_raw):
                    raise OSError(f"Could not write {raw_destination}")

            for name, value in result.timings.items():
                if isinstance(value, (int, float)):
                    timing_sums[name] = timing_sums.get(name, 0.0) + float(value)
            processed += 1
            print(
                f"[{index}/{len(image_paths)}] {input_path.name} -> "
                f"{destination.relative_to(output_dir)} ({result.timings['total']:.3f}s)"
            )
        except Exception as exc:
            failures.append({"input": str(input_path), "error": str(exc)})
            print(f"[{index}/{len(image_paths)}] ERROR {input_path}: {exc}", file=sys.stderr)
            if args.fail_fast:
                break

    elapsed = perf_counter() - started
    summary = {
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "discovered": len(image_paths),
        "processed": processed,
        "skipped": skipped,
        "failed": len(failures),
        "elapsed_seconds": elapsed,
        "frames_per_second": processed / elapsed if elapsed > 0 else 0.0,
        "average_timings": {
            name: total / processed for name, total in timing_sums.items()
        } if processed else {},
        "geometry_cache": cache_report,
        "failures": failures,
    }
    save_json(output_dir / "batch_summary.json", summary)
    print(
        f"Done: processed={processed}, skipped={skipped}, failed={len(failures)}, "
        f"elapsed={elapsed:.2f}s"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
