# Fisheye panorama prototype

Modular pipeline: equal-area/equidistant fisheye projection → three perspective panels → optional NAFNet restoration → deterministic known-geometry spherical panorama.

The stitcher inverse-warps every panel onto one equirectangular canvas using its known yaw, pitch, roll, and FOV. Panorama pixels are assigned to the view nearest its optical axis; GraphCut may refine the seam inside a narrow angular corridor, and only a small band around that seam is blended. The large source overlap is therefore available for seam selection without producing broad ghosted averages.

The defaults use the supplied calibration, `960×720` panels at yaw `-50/0/+50°`, a `1996×720` panorama canvas, a 4-degree GraphCut corridor, and 20-pixel seam feathering.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest
python scripts/test_projection.py /path/to/000036.jpg
python scripts/build_geometry_cache.py --config configs/default.yaml
python scripts/process_image.py --input /path/to/000036.jpg --output outputs/frame001 --restoration none
```

The first run stores content-addressed projection and panorama LUT files under `.cache/geometry`. Later images with identical camera, view, FOV, and panorama settings load those arrays from disk. Changing any geometry parameter automatically selects a new cache key. GraphCut itself remains per-image because its seam depends on image content.

Process a complete folder in one Python process so the LUTs (and NAFNet, when
enabled) are loaded only once and remain in RAM for every frame:

```bash
python scripts/process_folder.py \
  --input-dir /path/to/images \
  --output-dir outputs/dataset \
  --config configs/default.yaml \
  --restoration none
```

Add `--recursive` to scan subdirectories, `--overwrite` to replace existing
panoramas, or `--save-raw-panorama` to additionally save the no-restoration
result when NAFNet is enabled. Output keeps the input directory structure and
uses lossless PNG. A run report is written to `batch_summary.json`.

Debug output includes `ownership.png` and one blend-weight mask per view. For a deterministic midpoint seam, set `stitching.seam.method: nearest_axis`; this is also a useful diagnostic if GraphCut follows an undesirable object boundary.

For NAFNet, clone the official repository under `third_party/NAFNet`, place `NAFNet-REDS-width64.pth` under `weights/`, and run with `--restoration nafnet`. Raw and restored views stay separate, so an ANAFNet wrapper can later replace `NAFNetRestorer` without changing projection or stitching.
