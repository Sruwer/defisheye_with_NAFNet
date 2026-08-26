#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fisheye import DirectFisheyePanoramaProjector, RadialFisheyeModel, View
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
        if path.resolve().is_relative_to(output_resolved):
            continue
        images.append(path)
    return sorted(images, key=lambda path: str(path.relative_to(input_dir)))


def _output_path(input_path: Path, input_dir: Path, output_dir: Path) -> Path:
    return (output_dir / input_path.relative_to(input_dir)).with_suffix(".png")


def _projector(config: dict, interpolation: str) -> DirectFisheyePanoramaProjector:
    camera_config = config["camera"]
    projection_config = config["projection"]
    stitching_config = config["stitching"]
    cache_config = config.get("geometry_cache", {})
    cache_dir = (
        cache_config.get("directory", ".cache/geometry")
        if cache_config.get("enabled", True)
        else None
    )
    camera = RadialFisheyeModel(
        camera_config["cx"],
        camera_config["cy"],
        camera_config["radius_x"],
        camera_config["radius_y"],
        camera_config["fisheye_fov_deg"],
        camera_config["model"],
        tuple(camera_config.get("distortion", [0.0] * 4)),
    )
    views = [View(**view) for view in projection_config["views"]]
    vfov_deg = projection_config.get("vfov_deg")
    if vfov_deg is None:
        import numpy as np

        vfov_deg = np.rad2deg(
            2
            * np.arctan(
                np.tan(np.deg2rad(projection_config["hfov_deg"]) / 2)
                * projection_config["height"]
                / projection_config["width"]
            )
        )
    return DirectFisheyePanoramaProjector(
        camera,
        views,
        projection_config["hfov_deg"],
        vfov_deg,
        stitching_config["panorama_width"],
        stitching_config["panorama_height"],
        interpolation=interpolation,
        cache_dir=cache_dir,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fast folder processing via one direct fisheye-to-panorama LUT kept in RAM"
        )
    )
    parser.add_argument("--input-dir", "--input", required=True, type=Path)
    parser.add_argument("--output-dir", "--output", required=True, type=Path)
    parser.add_argument("--config", default="configs/default.yaml")
    # Retain compatibility with the first batch-script interface while making
    # it impossible to accidentally initialize NAFNet on this optimized path.
    parser.add_argument("--restoration", choices=("none",), default="none", help=argparse.SUPPRESS)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--interpolation",
        choices=("linear", "cubic", "lanczos"),
        default="linear",
        help="Remap interpolation; linear is fastest and is the default",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        choices=range(10),
        default=0,
        help="PNG compression 0-9; lower is faster (default: 0)",
    )
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

    cv2.setUseOptimized(True)
    config = load_config(args.config)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Construct and prepare once. All following calls share the same fixed-point
    # map arrays; no .npz file is touched inside the frame loop.
    projector = _projector(config, args.interpolation)
    cache_report = projector.prepare()
    print(f"Direct LUT prepared once: {cache_report}")
    print(f"Found {len(image_paths)} image(s)")

    started = perf_counter()
    processed = 0
    skipped = 0
    remap_seconds = 0.0
    failures: list[dict[str, str]] = []
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression]

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

            frame_started = perf_counter()
            panorama = projector.project(image)
            remap_seconds += perf_counter() - frame_started

            destination.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination), panorama, png_params):
                raise OSError(f"Could not write {destination}")
            processed += 1
            print(
                f"[{index}/{len(image_paths)}] {input_path.name} -> "
                f"{destination.relative_to(output_dir)} "
                f"({perf_counter() - frame_started:.3f}s)"
            )
        except Exception as exc:
            failures.append({"input": str(input_path), "error": str(exc)})
            print(f"[{index}/{len(image_paths)}] ERROR {input_path}: {exc}", file=sys.stderr)
            if args.fail_fast:
                break

    elapsed = perf_counter() - started
    summary = {
        "mode": "direct_fisheye_to_panorama_no_nafnet",
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "discovered": len(image_paths),
        "processed": processed,
        "skipped": skipped,
        "failed": len(failures),
        "elapsed_seconds": elapsed,
        "remap_seconds": remap_seconds,
        "frames_per_second": processed / elapsed if elapsed > 0 else 0.0,
        "average_remap_seconds": remap_seconds / processed if processed else 0.0,
        "interpolation": args.interpolation,
        "png_compression": args.png_compression,
        "geometry_cache": projector.cache_report(),
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
