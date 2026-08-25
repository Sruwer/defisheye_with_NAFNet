from __future__ import annotations
from dataclasses import dataclass
import numpy as np
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
                 hfov_deg: float, vfov_deg: float | None = None, interpolation: str = "lanczos"):
        self.camera, self.width, self.height = camera, width, height
        self.hfov_deg = hfov_deg
        self.vfov_deg = vfov_deg or np.rad2deg(2*np.arctan(np.tan(np.deg2rad(hfov_deg)/2)*height/width))
        self.interpolation = interpolation

    def maps(self, view: View) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = ((np.arange(self.width)+0.5)/self.width*2-1)*np.tan(np.deg2rad(self.hfov_deg)/2)
        y = -((np.arange(self.height)+0.5)/self.height*2-1)*np.tan(np.deg2rad(self.vfov_deg)/2)
        xx, yy = np.meshgrid(x, y)
        rays = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
        # Explicit multiply avoids a known Accelerate/NumPy 2 warning emitted by
        # batched matmul on some macOS installations.
        rot = _rotation(view)
        rays = np.stack(tuple(sum(rays[..., j] * rot[i, j] for j in range(3))
                              for i in range(3)), axis=-1)
        pixels, valid = self.camera.project_rays(rays)
        return pixels[..., 0], pixels[..., 1], valid

    def project(self, image: np.ndarray, view: View) -> tuple[np.ndarray, np.ndarray]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required; install with: pip install -e .") from exc
        flags = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR,
                 "cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}
        mx, my, valid = self.maps(view)
        out = cv2.remap(image, mx, my, flags[self.interpolation], borderMode=cv2.BORDER_CONSTANT)
        out[~valid] = 0
        return out, valid.astype(np.uint8)*255
