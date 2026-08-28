#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fisheye import PanoramaFisheyeProjector, RadialFisheyeModel, View
from src.utils import load_config, save_json


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _find_images(input_dir: Path, output_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    output_resolved = output_dir.resolve()
    return sorted(
        (
            path
            for path in iterator
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and not path.resolve().is_relative_to(output_resolved)
        ),
        key=lambda path: str(path.relative_to(input_dir)),
    )


def _make_projector(
    config: dict,
    fisheye_width: int,
    fisheye_height: int,
    interpolation: str,
) -> PanoramaFisheyeProjector:
    import numpy as np

    camera_config = config["camera"]
    projection_config = config["projection"]
    stitching_config = config["stitching"]
    cache_config = config.get("geometry_cache", {})
    cache_dir = (
        cache_config.get("directory", ".cache/geometry")
        if cache_config.get("enabled", True)
        else None
    )
    vfov_deg = projection_config.get("vfov_deg")
    if vfov_deg is None:
        vfov_deg = np.rad2deg(
            2
            * np.arctan(
                np.tan(np.deg2rad(projection_config["hfov_deg"]) / 2)
                * projection_config["height"]
                / projection_config["width"]
            )
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
    return PanoramaFisheyeProjector(
        camera,
        [View(**view) for view in projection_config["views"]],
        projection_config["hfov_deg"],
        vfov_deg,
        stitching_config["panorama_width"],
        stitching_config["panorama_height"],
        fisheye_width,
        fisheye_height,
        interpolation=interpolation,
        cache_dir=cache_dir,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a folder of generated panoramas back to fisheye images"
    )
    parser.add_argument("--input-dir", "--panorama-dir", required=True, type=Path)
    parser.add_argument("--output-dir", "--fisheye-dir", required=True, type=Path)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--fisheye-width", type=int)
    parser.add_argument("--fisheye-height", type=int)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--interpolation",
        choices=("linear", "cubic", "lanczos"),
        default="linear",
    )
    parser.add_argument("--png-compression", type=int, choices=range(10), default=0)
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")
    if input_dir == output_dir:
        parser.error("Input and output directories must be different")
    panoramas = _find_images(input_dir, output_dir, args.recursive)
    if not panoramas:
        parser.error(f"No panorama images found in: {input_dir}")

    try:
        import cv2
    except ImportError:
        raise SystemExit("OpenCV is required: pip install -e .")

    config = load_config(args.config)
    input_config = config.get("input", {})
    fisheye_width = args.fisheye_width or input_config.get("width")
    fisheye_height = args.fisheye_height or input_config.get("height")
    if not fisheye_width or not fisheye_height:
        parser.error(
            "Set input.width/input.height in the config or pass "
            "--fisheye-width and --fisheye-height"
        )

    cv2.setUseOptimized(True)
    output_dir.mkdir(parents=True, exist_ok=True)
    projector = _make_projector(
        config, fisheye_width, fisheye_height, args.interpolation
    )
    cache_report = projector.prepare()
    print(f"Inverse LUT prepared once: {cache_report}")
    print(f"Found {len(panoramas)} panorama(s)")

    started = perf_counter()
    processed = 0
    skipped = 0
    failures: list[dict[str, str]] = []
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression]
    for index, panorama_path in enumerate(panoramas, start=1):
        destination = (
            output_dir / panorama_path.relative_to(input_dir)
        ).with_suffix(".png")
        if destination.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(panoramas)}] skip {panorama_path.name}")
            continue
        try:
            panorama = cv2.imread(str(panorama_path), cv2.IMREAD_COLOR)
            if panorama is None:
                raise ValueError("OpenCV could not decode the panorama")
            frame_started = perf_counter()
            fisheye = projector.project(panorama)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination), fisheye, png_params):
                raise OSError(f"Could not write {destination}")
            processed += 1
            print(
                f"[{index}/{len(panoramas)}] {panorama_path.name} -> "
                f"{destination.relative_to(output_dir)} "
                f"({perf_counter() - frame_started:.3f}s)"
            )
        except Exception as exc:
            failures.append({"input": str(panorama_path), "error": str(exc)})
            print(f"[{index}/{len(panoramas)}] ERROR {panorama_path}: {exc}", file=sys.stderr)
            if args.fail_fast:
                break

    elapsed = perf_counter() - started
    save_json(
        output_dir / "batch_summary.json",
        {
            "mode": "panorama_to_fisheye",
            "input_directory": str(input_dir),
            "output_directory": str(output_dir),
            "processed": processed,
            "skipped": skipped,
            "failed": len(failures),
            "elapsed_seconds": elapsed,
            "frames_per_second": processed / elapsed if elapsed > 0 else 0.0,
            "geometry_cache": projector.cache_report(),
            "failures": failures,
        },
    )
    print(
        f"Done: processed={processed}, skipped={skipped}, "
        f"failed={len(failures)}, elapsed={elapsed:.2f}s"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
