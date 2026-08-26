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
    panorama_raw: np.ndarray | None
    raw_views: list[np.ndarray]
    restored_views: list[np.ndarray]
    masks: list[np.ndarray]
    timings: dict
    debug: dict

class FisheyePanoramaPipeline:
    def __init__(self, config: dict, restoration_override: str | None = None):
        self.config=config; c=config["camera"]; p=config["projection"]
        cache_config=config.get("geometry_cache", {})
        cache_dir=cache_config.get("directory", ".cache/geometry") if cache_config.get("enabled", True) else None
        camera=RadialFisheyeModel(c["cx"],c["cy"],c["radius_x"],c["radius_y"],c["fisheye_fov_deg"],c["model"],tuple(c.get("distortion",[0]*4)))
        self.projector=PerspectiveProjector(camera,p["width"],p["height"],p["hfov_deg"],p.get("vfov_deg"),p.get("interpolation","lanczos"),cache_dir)
        self.views=[View(**v) for v in p["views"]]
        rc=config["restoration"]; mode=restoration_override or (rc["type"] if rc.get("enabled") else "none")
        self.restoration_mode=mode
        self.restorer=IdentityRestorer() if mode=="none" else NAFNetRestorer(rc)
        sc=config["stitching"]
        if sc.get("backend") != "known_geometry": raise ValueError("This prototype currently supports stitching.backend=known_geometry")
        seam = sc.get("seam", {})
        self.stitcher=KnownGeometryStitcher(
            views=self.views,
            hfov_deg=p["hfov_deg"],
            vfov_deg=self.projector.vfov_deg,
            panorama_width=sc["panorama_width"],
            panorama_height=sc["panorama_height"],
            projection=sc.get("projection", "equirectangular"),
            ownership=sc.get("ownership", "nearest_axis"),
            seam_method=seam.get("method", "graphcut"),
            seam_search_band_deg=seam.get("search_band_deg", 4.0),
            feather_px=seam.get("feather_px", 20),
            border_margin_px=sc.get("border_margin_px", 32),
            blend=sc.get("blend", "multiband"),
            blend_levels=sc.get("blend_levels", sc.get("blend_strength", 4)),
            cache_dir=cache_dir,
        )

    def prepare_geometry_cache(self) -> dict:
        return {
            "projection": self.projector.prepare(self.views),
            "stitching": self.stitcher.prepare(self.projector.height, self.projector.width),
        }

    def process(self, image: np.ndarray, include_raw_panorama: bool = True) -> PipelineResult:
        started=perf_counter(); raw=[]; masks=[]
        t=perf_counter()
        for view in self.views:
            panel,mask=self.projector.project(image,view); raw.append(panel); masks.append(mask)
        projection=perf_counter()-t
        restored=[]; per_view=[]
        if self.restoration_mode == "none":
            restored=raw
            per_view=[0.0]*len(raw)
        else:
            for panel in raw:
                t=perf_counter(); restored.append(self.restorer.restore(panel)); per_view.append(perf_counter()-t)
        if self.restoration_mode == "none" or include_raw_panorama:
            t=perf_counter(); pano_raw,raw_mask=self.stitcher.stitch(raw,masks); raw_stitch=perf_counter()-t
        else:
            pano_raw=None; raw_mask=None; raw_stitch=0.0
        if self.restoration_mode == "none":
            assert pano_raw is not None and raw_mask is not None
            panorama=pano_raw.copy(); mask=raw_mask.copy(); restore_stitch=0.0
        else:
            t=perf_counter(); panorama,mask=self.stitcher.stitch(restored,masks); restore_stitch=perf_counter()-t
        stitching_debug = self.stitcher.last_debug
        ys, xs = np.nonzero(mask)
        if len(xs):
            y0, y1 = int(ys.min()), int(ys.max() + 1)
            x0, x1 = int(xs.min()), int(xs.max() + 1)
            ownership = stitching_debug["ownership"][y0:y1, x0:x1]
            blend_weights = stitching_debug["blend_weights"][:, y0:y1, x0:x1]
        else:
            ownership = stitching_debug["ownership"][:0, :0]
            blend_weights = stitching_debug["blend_weights"][:, :0, :0]
        panorama,mask=crop_to_valid_region(panorama,mask)
        if pano_raw is not None and raw_mask is not None:
            pano_raw,_=crop_to_valid_region(pano_raw,raw_mask)
        timings={"projection":projection,"restoration_per_view":per_view,"restoration":sum(per_view),
                 "stitching_raw":raw_stitch,"stitching_restored":restore_stitch,"total":perf_counter()-started}
        debug={
            "valid_mask": mask,
            "ownership": ownership,
            "blend_weights": blend_weights,
            "longitude_bounds_deg": stitching_debug["longitude_bounds_deg"],
            "latitude_bounds_deg": stitching_debug["latitude_bounds_deg"],
            "geometry_cache": {
                "projection": self.projector.cache_report(),
                "stitching": self.stitcher.cache_report(),
            },
        }
        return PipelineResult(panorama,pano_raw,raw,restored,masks,timings,debug)
