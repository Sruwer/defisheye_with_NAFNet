#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.pipeline import FisheyePanoramaPipeline
from src.utils import load_config, save_json

def main():
    ap=argparse.ArgumentParser(description="Fisheye → views → optional NAFNet → panorama")
    ap.add_argument("--input",required=True); ap.add_argument("--config",default="configs/default.yaml")
    ap.add_argument("--output",required=True); ap.add_argument("--restoration",choices=["none","nafnet"])
    args=ap.parse_args()
    try: import cv2
    except ImportError: raise SystemExit("OpenCV is required: pip install -e .")
    image=cv2.imread(args.input)
    if image is None: raise SystemExit(f"Cannot read input image: {args.input}")
    cfg=load_config(args.config); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    result=FisheyePanoramaPipeline(cfg,args.restoration).process(image)
    cv2.imwrite(str(out/"input.png"),image)
    for folder,items in (("raw",result.raw_views),("restored",result.restored_views)):
        d=out/folder; d.mkdir(exist_ok=True)
        for i,(item,view) in enumerate(zip(items,cfg["projection"]["views"])): cv2.imwrite(str(d/f"view_{i:02d}_yaw_{view['yaw']:g}.png"),item)
    cv2.imwrite(str(out/"panorama_without_nafnet.png"),result.panorama_raw)
    cv2.imwrite(str(out/"panorama_with_restoration.png"),result.panorama)
    cv2.imwrite(str(out/"valid_mask.png"),result.debug["valid_mask"])
    save_json(out/"timings.json",result.timings)
    print(f"Saved output to {out}; result shape={result.panorama.shape}; total={result.timings['total']:.3f}s")
if __name__=="__main__": main()

