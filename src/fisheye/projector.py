from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ..geometry_cache import cache_path, load_arrays, save_arrays
from .camera_models import FisheyeCameraModel


@dataclass(frozen=True)
class View:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


def _rotation(view: View) -> np.ndarray:
    y, p, r = np.deg2rad([view.yaw, view.pitch, view.roll])
    ry = np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    rx = np.array([[1,0,0],[0,np.cos(p),-np.sin(p)],[0,np.sin(p),np.cos(p)]])
    rz = np.array([[np.cos(r),-np.sin(r),0],[np.sin(r),np.cos(r),0],[0,0,1]])
    return ry @ rx @ rz


class PerspectiveProjector:
    def __init__(self, camera: FisheyeCameraModel, width: int, height: int,
                 hfov_deg: float, vfov_deg: float | None = None, interpolation: str = "lanczos",
                 cache_dir: str | Path | None = None):
        self.camera, self.width, self.height = camera, width, height
        self.hfov_deg = hfov_deg
        self.vfov_deg = vfov_deg or np.rad2deg(2*np.arctan(np.tan(np.deg2rad(hfov_deg)/2)*height/width))
        self.interpolation = interpolation
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._memory_maps: dict[View, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._cache_events: dict[View, dict[str, object]] = {}

    def _cache_path(self, view: View) -> Path | None:
        return cache_path(
            self.cache_dir,
            "fisheye-perspective-v1",
            {
                "camera_class": type(self.camera).__qualname__,
                "camera": vars(self.camera),
                "width": self.width,
                "height": self.height,
                "hfov_deg": self.hfov_deg,
                "vfov_deg": self.vfov_deg,
                "view": {"yaw": view.yaw, "pitch": view.pitch, "roll": view.roll},
            },
        )

    def maps(self, view: View) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if view in self._memory_maps:
            return self._memory_maps[view]

        path = self._cache_path(view)
        cached = load_arrays(path, {"map_x", "map_y", "valid"})
        expected_shape = (self.height, self.width)
        if cached is not None and all(cached[name].shape == expected_shape for name in cached):
            result = (
                cached["map_x"].astype(np.float32, copy=False),
                cached["map_y"].astype(np.float32, copy=False),
                cached["valid"].astype(bool, copy=False),
            )
            self._memory_maps[view] = result
            self._cache_events[view] = {"source": "disk", "path": str(path)}
            return result

        x = ((np.arange(self.width)+0.5)/self.width*2-1)*np.tan(np.deg2rad(self.hfov_deg)/2)
        # The camera model uses image coordinates (x right, y down).  Keeping
        # perspective Y in the same convention prevents a vertical flip.
        y = ((np.arange(self.height)+0.5)/self.height*2-1)*np.tan(np.deg2rad(self.vfov_deg)/2)
        xx, yy = np.meshgrid(x, y)
        rays = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
        # Explicit multiply avoids a known Accelerate/NumPy 2 warning emitted by
        # batched matmul on some macOS installations.
        rot = _rotation(view)
        rays = np.stack(tuple(sum(rays[..., j] * rot[i, j] for j in range(3))
                              for i in range(3)), axis=-1)
        pixels, valid = self.camera.project_rays(rays)
        result = (
            pixels[..., 0].astype(np.float32, copy=False),
            pixels[..., 1].astype(np.float32, copy=False),
            valid.astype(bool, copy=False),
        )
        save_arrays(path, {"map_x": result[0], "map_y": result[1], "valid": result[2]})
        self._memory_maps[view] = result
        self._cache_events[view] = {
            "source": "computed" if path is not None else "disabled",
            "path": str(path) if path is not None else None,
        }
        return result

    def prepare(self, views: list[View]) -> dict:
        for view in views:
            self.maps(view)
        return self.cache_report()

    def cache_report(self) -> dict:
        entries = []
        for view, event in self._cache_events.items():
            entries.append({
                "view": {"yaw": view.yaw, "pitch": view.pitch, "roll": view.roll},
                **event,
            })
        return {"enabled": self.cache_dir is not None, "entries": entries}

    def project(self, image: np.ndarray, view: View) -> tuple[np.ndarray, np.ndarray]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required; install with: pip install -e .") from exc
        flags = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR,
                 "cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}
        mx, my, valid = self.maps(view)
        source_height, source_width = image.shape[:2]
        # A cropped fisheye circle can extend beyond the stored image. Those
        # rays must not become valid black samples during panorama blending.
        valid = (valid & (mx >= 0) & (mx <= source_width - 1)
                 & (my >= 0) & (my <= source_height - 1))
        out = cv2.remap(image, mx, my, flags[self.interpolation], borderMode=cv2.BORDER_CONSTANT)
        out[~valid] = 0
        return out, valid.astype(np.uint8)*255
