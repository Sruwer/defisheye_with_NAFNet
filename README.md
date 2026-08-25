# Fisheye panorama prototype

Modular pipeline: equal-area/equidistant fisheye projection → three perspective panels → optional NAFNet restoration → deterministic known-geometry overlap blend.

The defaults reproduce the supplied calibration: center `(960, 583)`, radii `(814, 810)`, fisheye FOV `190°`, `960×720` panels at yaw `-45/0/+45°`, and 96 px overlap, yielding `2688×720`.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest
python scripts/test_projection.py /path/to/000036.jpg
python scripts/process_image.py --input /path/to/000036.jpg --output outputs/frame001 --restoration none
```

For NAFNet, clone the official repository under `third_party/NAFNet`, place `NAFNet-REDS-width64.pth` under `weights/`, and run with `--restoration nafnet`. Raw and restored views stay separate, so an ANAFNet wrapper can later replace `NAFNetRestorer` without changing projection or stitching.
