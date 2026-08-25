from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


class FisheyeCameraModel(ABC):
    @abstractmethod
    def project_rays(self, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return source pixels and a validity mask for (..., 3) unit rays."""

    @abstractmethod
    def unproject_pixels(self, pixels: np.ndarray) -> np.ndarray:
        """Return unit rays for (..., 2) source pixels."""


@dataclass(frozen=True)
class RadialFisheyeModel(FisheyeCameraModel):
    cx: float
    cy: float
    radius_x: float
    radius_y: float
    fisheye_fov_deg: float
    model: str = "equalarea"
    distortion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def _rho(self, theta: np.ndarray) -> np.ndarray:
        half = np.deg2rad(self.fisheye_fov_deg) / 2
        if self.model == "equalarea":
            return np.sin(theta / 2) / np.sin(half / 2)
        if self.model == "equidistant":
            return theta / half
        raise ValueError(f"Unsupported fisheye model: {self.model}")

    def _theta(self, rho: np.ndarray) -> np.ndarray:
        half = np.deg2rad(self.fisheye_fov_deg) / 2
        if self.model == "equalarea":
            return 2 * np.arcsin(np.clip(rho * np.sin(half / 2), -1, 1))
        if self.model == "equidistant":
            return rho * half
        raise ValueError(f"Unsupported fisheye model: {self.model}")

    def project_rays(self, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rays = np.asarray(rays, dtype=np.float64)
        rays = rays / np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-12)
        theta = np.arccos(np.clip(rays[..., 2], -1, 1))
        phi = np.arctan2(rays[..., 0], -rays[..., 1])
        rho = self._rho(theta)
        r2 = rho * rho
        k1, k2, k3, k4 = self.distortion
        rho = rho * (1 + k1*r2 + k2*r2**2 + k3*r2**3 + k4*r2**4)
        pixels = np.stack((self.cx + self.radius_x*rho*np.sin(phi),
                           self.cy - self.radius_y*rho*np.cos(phi)), axis=-1)
        return pixels.astype(np.float32), (rho <= 1.0)

    def unproject_pixels(self, pixels: np.ndarray) -> np.ndarray:
        p = np.asarray(pixels, dtype=np.float64)
        nx, ny = (p[..., 0]-self.cx)/self.radius_x, (p[..., 1]-self.cy)/self.radius_y
        rho = np.hypot(nx, ny)
        theta = self._theta(rho)
        phi = np.arctan2(nx, -ny)
        st = np.sin(theta)
        return np.stack((st*np.sin(phi), -st*np.cos(phi), np.cos(theta)), axis=-1)

