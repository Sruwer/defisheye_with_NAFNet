#!/usr/bin/env python3
from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.pipeline import FisheyePanoramaPipeline
from src.utils import load_config

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("input"); ap.add_argument("--config",default="configs/default.yaml"); ap.add_argument("--output",default="outputs/projection_debug"); args=ap.parse_args()
    import cv2
    cfg=load_config(args.config); image=cv2.imread(args.input)
    if image is None: raise SystemExit(f"Cannot read {args.input}")
    out=Path(args.output); (out/"perspective_raw").mkdir(parents=True,exist_ok=True)
    c=cfg["camera"]; debug=image.copy(); center=(round(c["cx"]),round(c["cy"]))
    cv2.ellipse(debug,center,(round(c["radius_x"]),round(c["radius_y"])),0,0,360,(0,255,0),2); cv2.drawMarker(debug,center,(0,0,255),cv2.MARKER_CROSS,30,2)
    cv2.imwrite(str(out/"fisheye_center_debug.png"),debug)
    pipe=FisheyePanoramaPipeline(cfg,"none")
    for i,v in enumerate(pipe.views):
        panel,_=pipe.projector.project(image,v); cv2.imwrite(str(out/"perspective_raw"/f"view_{i:02d}.png"),panel)
if __name__=="__main__": main()

