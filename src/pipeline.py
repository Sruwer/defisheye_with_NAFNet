from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from .fisheye import RadialFisheyeModel, PerspectiveProjector, View
from .restoration import IdentityRestorer, NAFNetRestorer
from .stitching.known_geometry import KnownGeometryStitcher, crop_to_valid_region

@dataclass
class PipelineResult:
    panorama: np.ndarray
    panorama_raw: np.ndarray
    raw_views: list[np.ndarray]
    restored_views: list[np.ndarray]
    masks: list[np.ndarray]
    timings: dict
    debug: dict

class FisheyePanoramaPipeline:
    def __init__(self, config: dict, restoration_override: str | None = None):
        self.config=config; c=config["camera"]; p=config["projection"]
        camera=RadialFisheyeModel(c["cx"],c["cy"],c["radius_x"],c["radius_y"],c["fisheye_fov_deg"],c["model"],tuple(c.get("distortion",[0]*4)))
        self.projector=PerspectiveProjector(camera,p["width"],p["height"],p["hfov_deg"],p.get("vfov_deg"),p.get("interpolation","lanczos"))
        self.views=[View(**v) for v in p["views"]]
        rc=config["restoration"]; mode=restoration_override or (rc["type"] if rc.get("enabled") else "none")
        self.restorer=IdentityRestorer() if mode=="none" else NAFNetRestorer(rc)
        sc=config["stitching"]
        if sc.get("backend") != "known_geometry": raise ValueError("This prototype currently supports stitching.backend=known_geometry")
        self.stitcher=KnownGeometryStitcher(sc.get("overlap_fraction",.1),sc.get("blend_strength",5))

    def process(self, image: np.ndarray) -> PipelineResult:
        started=perf_counter(); raw=[]; masks=[]
        t=perf_counter()
        for view in self.views:
            panel,mask=self.projector.project(image,view); raw.append(panel); masks.append(mask)
        projection=perf_counter()-t
        restored=[]; per_view=[]
        for panel in raw:
            t=perf_counter(); restored.append(self.restorer.restore(panel)); per_view.append(perf_counter()-t)
        t=perf_counter(); pano_raw,raw_mask=self.stitcher.stitch(raw,masks); raw_stitch=perf_counter()-t
        t=perf_counter(); panorama,mask=self.stitcher.stitch(restored,masks); restore_stitch=perf_counter()-t
        panorama,mask=crop_to_valid_region(panorama,mask); pano_raw,_=crop_to_valid_region(pano_raw,raw_mask)
        timings={"projection":projection,"restoration_per_view":per_view,"restoration":sum(per_view),
                 "stitching_raw":raw_stitch,"stitching_restored":restore_stitch,"total":perf_counter()-started}
        return PipelineResult(panorama,pano_raw,raw,restored,masks,timings,{"valid_mask":mask})

