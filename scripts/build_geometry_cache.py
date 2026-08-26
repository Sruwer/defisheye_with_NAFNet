#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import FisheyePanoramaPipeline
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Precompute fisheye-to-view and view-to-panorama lookup tables"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = FisheyePanoramaPipeline(config, restoration_override="none")
    report = pipeline.prepare_geometry_cache()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
