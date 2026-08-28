from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..geometry_cache import cache_path, load_arrays, save_arrays
from .camera_models import FisheyeCameraModel
from .projector import View, _rotation


def _local_to_world(rays: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return np.stack(
        tuple(
            sum(rays[..., j] * rotation[i, j] for j in range(3))
            for i in range(3)
        ),
        axis=-1,
    )


def _world_to_local(rays: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return np.stack(
        tuple(
            sum(rays[..., j] * rotation[j, i] for j in range(3))
            for i in range(3)
        ),
        axis=-1,
    )


@dataclass(frozen=True)
class _DirectLut:
    map1: np.ndarray
    map2: np.ndarray
    valid: np.ndarray
    longitude_bounds_deg: tuple[float, float]
    latitude_bounds_deg: tuple[float, float]


@dataclass(frozen=True)
class _SourceLut:
    map1: np.ndarray
    map2: np.ndarray
    valid: np.ndarray
    crop_box: tuple[int, int, int, int]


class DirectFisheyePanoramaProjector:
    """Fast fisheye-to-panorama path for datasets that do not use restoration.

    With no NAFNet, perspective panels contain no information that is absent
    from the original fisheye frame. The complete projection/stitching chain is
    therefore reduced to one fixed-point OpenCV remap table.
    """

    def __init__(
        self,
        camera: FisheyeCameraModel,
        views: list[View],
        hfov_deg: float,
        vfov_deg: float,
        panorama_width: int,
        panorama_height: int,
        *,
        interpolation: str = "linear",
        cache_dir: str | Path | None = None,
    ):
        if not views:
            raise ValueError("At least one view is required")
        if not 0 < hfov_deg < 180 or not 0 < vfov_deg < 180:
            raise ValueError("Perspective FOV must be between 0 and 180 degrees")
        if panorama_width <= 0 or panorama_height <= 0:
            raise ValueError("Panorama dimensions must be positive")
        if interpolation not in {"linear", "cubic", "lanczos"}:
            raise ValueError("interpolation must be linear, cubic, or lanczos")

        self.camera = camera
        self.views = list(views)
        self.hfov_deg = float(hfov_deg)
        self.vfov_deg = float(vfov_deg)
        self.panorama_width = int(panorama_width)
        self.panorama_height = int(panorama_height)
        self.interpolation = interpolation
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

        self._lut: _DirectLut | None = None
        self._source_luts: dict[tuple[int, int], _SourceLut] = {}
        self._cache_source = "not_prepared"
        self._cache_file: Path | None = None

    @staticmethod
    def _cv2():
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required; install with: pip install -e .") from exc
        return cv2

    def _cache_path(self) -> Path | None:
        return cache_path(
            self.cache_dir,
            "fisheye-panorama-direct-v1",
            {
                "camera_class": type(self.camera).__qualname__,
                "camera": vars(self.camera),
                "views": [
                    {"yaw": view.yaw, "pitch": view.pitch, "roll": view.roll}
                    for view in self.views
                ],
                "hfov_deg": self.hfov_deg,
                "vfov_deg": self.vfov_deg,
                "panorama_width": self.panorama_width,
                "panorama_height": self.panorama_height,
            },
        )

    def _frustum_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        tan_h = np.tan(np.deg2rad(self.hfov_deg) / 2)
        tan_v = np.tan(np.deg2rad(self.vfov_deg) / 2)
        t = np.linspace(-1.0, 1.0, 257)
        borders = np.concatenate(
            (
                np.stack((t * tan_h, np.full_like(t, -tan_v), np.ones_like(t)), axis=-1),
                np.stack((t * tan_h, np.full_like(t, tan_v), np.ones_like(t)), axis=-1),
                np.stack((np.full_like(t, -tan_h), t * tan_v, np.ones_like(t)), axis=-1),
                np.stack((np.full_like(t, tan_h), t * tan_v, np.ones_like(t)), axis=-1),
            )
        )
        reference = float(np.median([view.yaw for view in self.views]))
        longitudes: list[np.ndarray] = []
        latitudes: list[np.ndarray] = []
        for view in self.views:
            world = _local_to_world(borders, _rotation(view))
            world /= np.maximum(np.linalg.norm(world, axis=-1, keepdims=True), 1e-12)
            longitude = np.rad2deg(np.arctan2(world[:, 0], world[:, 2]))
            longitude = reference + (longitude - reference + 180.0) % 360.0 - 180.0
            latitude = np.rad2deg(np.arcsin(np.clip(-world[:, 1], -1.0, 1.0)))
            longitudes.append(longitude)
            latitudes.append(latitude)
        all_longitudes = np.concatenate(longitudes)
        all_latitudes = np.concatenate(latitudes)
        return (
            (float(all_longitudes.min()), float(all_longitudes.max())),
            (float(all_latitudes.min()), float(all_latitudes.max())),
        )

    def _panorama_rays(
        self,
        longitude_bounds_deg: tuple[float, float],
        latitude_bounds_deg: tuple[float, float],
    ) -> np.ndarray:
        lon_step = (longitude_bounds_deg[1] - longitude_bounds_deg[0]) / self.panorama_width
        lat_step = (latitude_bounds_deg[1] - latitude_bounds_deg[0]) / self.panorama_height
        longitude = np.deg2rad(
            longitude_bounds_deg[0] + (np.arange(self.panorama_width) + 0.5) * lon_step
        )
        latitude = np.deg2rad(
            latitude_bounds_deg[1] - (np.arange(self.panorama_height) + 0.5) * lat_step
        )
        longitude, latitude = np.meshgrid(longitude, latitude)
        cos_latitude = np.cos(latitude)
        return np.stack(
            (
                cos_latitude * np.sin(longitude),
                -np.sin(latitude),
                cos_latitude * np.cos(longitude),
            ),
            axis=-1,
        )

    def _view_coverage(self, rays: np.ndarray) -> np.ndarray:
        tan_h = np.tan(np.deg2rad(self.hfov_deg) / 2)
        tan_v = np.tan(np.deg2rad(self.vfov_deg) / 2)
        coverage = np.zeros(rays.shape[:2], dtype=bool)
        for view in self.views:
            local = _world_to_local(rays, _rotation(view))
            z = local[..., 2]
            safe_z = np.where(np.abs(z) > 1e-12, z, 1.0)
            plane_x = local[..., 0] / safe_z
            plane_y = local[..., 1] / safe_z
            coverage |= (
                (z > 0)
                & (np.abs(plane_x) <= tan_h)
                & (np.abs(plane_y) <= tan_v)
            )
        return coverage

    def _build_lut(self) -> _DirectLut:
        cv2 = self._cv2()
        longitude_bounds, latitude_bounds = self._frustum_bounds()
        rays = self._panorama_rays(longitude_bounds, latitude_bounds)
        pixels, fisheye_valid = self.camera.project_rays(rays)
        valid = fisheye_valid & self._view_coverage(rays)
        pixels = pixels.astype(np.float32, copy=False)
        pixels[~valid] = -1.0
        map1, map2 = cv2.convertMaps(
            pixels[..., 0], pixels[..., 1], cv2.CV_16SC2, nninterpolation=False
        )
        return _DirectLut(
            map1,
            map2,
            valid,
            longitude_bounds,
            latitude_bounds,
        )

    def _load_lut(self, path: Path | None) -> _DirectLut | None:
        names = {
            "map1",
            "map2",
            "valid",
            "longitude_bounds_deg",
            "latitude_bounds_deg",
        }
        cached = load_arrays(path, names)
        if cached is None:
            return None
        image_shape = (self.panorama_height, self.panorama_width)
        if (
            cached["map1"].shape != image_shape + (2,)
            or cached["map2"].shape != image_shape
            or cached["valid"].shape != image_shape
            or cached["longitude_bounds_deg"].shape != (2,)
            or cached["latitude_bounds_deg"].shape != (2,)
        ):
            return None
        return _DirectLut(
            cached["map1"].astype(np.int16, copy=False),
            cached["map2"].astype(np.uint16, copy=False),
            cached["valid"].astype(bool, copy=False),
            tuple(float(x) for x in cached["longitude_bounds_deg"]),
            tuple(float(x) for x in cached["latitude_bounds_deg"]),
        )

    @staticmethod
    def _save_lut(path: Path | None, lut: _DirectLut) -> None:
        save_arrays(
            path,
            {
                "map1": lut.map1,
                "map2": lut.map2,
                "valid": lut.valid,
                "longitude_bounds_deg": np.asarray(lut.longitude_bounds_deg),
                "latitude_bounds_deg": np.asarray(lut.latitude_bounds_deg),
            },
        )

    def prepare(self) -> dict:
        if self._lut is None:
            path = self._cache_path()
            self._lut = self._load_lut(path)
            if self._lut is None:
                self._lut = self._build_lut()
                self._save_lut(path, self._lut)
                self._cache_source = "computed" if path is not None else "disabled"
            else:
                self._cache_source = "disk"
            self._cache_file = path
        return self.cache_report()

    def _source_lut(self, source_height: int, source_width: int) -> _SourceLut:
        key = (source_height, source_width)
        if key in self._source_luts:
            return self._source_luts[key]
        self.prepare()
        assert self._lut is not None

        integer_x = self._lut.map1[..., 0]
        integer_y = self._lut.map1[..., 1]
        valid = (
            self._lut.valid
            & (integer_x >= 0)
            & (integer_x < source_width)
            & (integer_y >= 0)
            & (integer_y < source_height)
        )
        ys, xs = np.nonzero(valid)
        if not len(xs):
            source_lut = _SourceLut(
                self._lut.map1[:0, :0],
                self._lut.map2[:0, :0],
                valid[:0, :0],
                (0, 0, 0, 0),
            )
        else:
            y0, y1 = int(ys.min()), int(ys.max() + 1)
            x0, x1 = int(xs.min()), int(xs.max() + 1)
            # Contiguous cropped maps avoid an implicit NumPy-to-OpenCV copy on
            # every frame. The cost is paid only for the first source shape.
            source_lut = _SourceLut(
                np.ascontiguousarray(self._lut.map1[y0:y1, x0:x1]),
                np.ascontiguousarray(self._lut.map2[y0:y1, x0:x1]),
                np.ascontiguousarray(valid[y0:y1, x0:x1]),
                (x0, y0, x1, y1),
            )
        self._source_luts[key] = source_lut
        return source_lut

    def project(self, image: np.ndarray) -> np.ndarray:
        cv2 = self._cv2()
        source_height, source_width = image.shape[:2]
        lut = self._source_lut(source_height, source_width)
        if not lut.valid.size:
            return np.zeros((0, 0, image.shape[2]), dtype=image.dtype)
        flags = {
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }
        result = cv2.remap(
            image,
            lut.map1,
            lut.map2,
            flags[self.interpolation],
            borderMode=cv2.BORDER_CONSTANT,
        )
        result[~lut.valid] = 0
        return result

    def cache_report(self) -> dict:
        size_bytes = self._cache_file.stat().st_size if self._cache_file and self._cache_file.is_file() else None
        return {
            "enabled": self.cache_dir is not None,
            "source": self._cache_source,
            "path": str(self._cache_file) if self._cache_file is not None else None,
            "size_bytes": size_bytes,
            "resident_source_shapes": [list(shape) for shape in self._source_luts],
        }


class PanoramaFisheyeProjector:
    """Reproject cropped panoramas made by the direct path back to fisheye."""

    def __init__(
        self,
        camera: FisheyeCameraModel,
        views: list[View],
        hfov_deg: float,
        vfov_deg: float,
        panorama_width: int,
        panorama_height: int,
        fisheye_width: int,
        fisheye_height: int,
        *,
        interpolation: str = "linear",
        cache_dir: str | Path | None = None,
    ):
        if fisheye_width <= 0 or fisheye_height <= 0:
            raise ValueError("Fisheye dimensions must be positive")
        if interpolation not in {"linear", "cubic", "lanczos"}:
            raise ValueError("interpolation must be linear, cubic, or lanczos")
        self.camera = camera
        self.views = list(views)
        self.hfov_deg = float(hfov_deg)
        self.vfov_deg = float(vfov_deg)
        self.panorama_width = int(panorama_width)
        self.panorama_height = int(panorama_height)
        self.fisheye_width = int(fisheye_width)
        self.fisheye_height = int(fisheye_height)
        self.interpolation = interpolation
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

        self._forward = DirectFisheyePanoramaProjector(
            camera,
            views,
            hfov_deg,
            vfov_deg,
            panorama_width,
            panorama_height,
            interpolation="linear",
            cache_dir=cache_dir,
        )
        self._map1: np.ndarray | None = None
        self._map2: np.ndarray | None = None
        self._valid: np.ndarray | None = None
        self._panorama_shape: tuple[int, int] | None = None
        self._crop_box: tuple[int, int, int, int] | None = None
        self._cache_source = "not_prepared"
        self._cache_file: Path | None = None

    @staticmethod
    def _cv2():
        return DirectFisheyePanoramaProjector._cv2()

    def _cache_path(
        self, crop_box: tuple[int, int, int, int]
    ) -> Path | None:
        return cache_path(
            self.cache_dir,
            "panorama-fisheye-direct-v1",
            {
                "camera_class": type(self.camera).__qualname__,
                "camera": vars(self.camera),
                "views": [
                    {"yaw": view.yaw, "pitch": view.pitch, "roll": view.roll}
                    for view in self.views
                ],
                "hfov_deg": self.hfov_deg,
                "vfov_deg": self.vfov_deg,
                "panorama_width": self.panorama_width,
                "panorama_height": self.panorama_height,
                "fisheye_width": self.fisheye_width,
                "fisheye_height": self.fisheye_height,
                "crop_box": crop_box,
            },
        )

    def _build_maps(
        self,
        crop_box: tuple[int, int, int, int],
        longitude_bounds: tuple[float, float],
        latitude_bounds: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cv2 = self._cv2()
        pixel_x, pixel_y = np.meshgrid(
            np.arange(self.fisheye_width, dtype=np.float64),
            np.arange(self.fisheye_height, dtype=np.float64),
        )
        fisheye_pixels = np.stack((pixel_x, pixel_y), axis=-1)
        rays = self.camera.unproject_pixels(fisheye_pixels)
        _, fisheye_valid = self.camera.project_rays(rays)
        coverage = self._forward._view_coverage(rays)

        longitude = np.rad2deg(np.arctan2(rays[..., 0], rays[..., 2]))
        reference = (longitude_bounds[0] + longitude_bounds[1]) / 2.0
        longitude = reference + (longitude - reference + 180.0) % 360.0 - 180.0
        latitude = np.rad2deg(
            np.arcsin(np.clip(-rays[..., 1], -1.0, 1.0))
        )
        longitude_step = (longitude_bounds[1] - longitude_bounds[0]) / self.panorama_width
        latitude_step = (latitude_bounds[1] - latitude_bounds[0]) / self.panorama_height
        crop_x0, crop_y0, crop_x1, crop_y1 = crop_box
        map_x = (
            (longitude - longitude_bounds[0]) / longitude_step
            - 0.5
            - crop_x0
        )
        map_y = (
            (latitude_bounds[1] - latitude) / latitude_step
            - 0.5
            - crop_y0
        )
        cropped_width = crop_x1 - crop_x0
        cropped_height = crop_y1 - crop_y0
        valid = (
            fisheye_valid
            & coverage
            & (map_x >= 0)
            & (map_x <= cropped_width - 1)
            & (map_y >= 0)
            & (map_y <= cropped_height - 1)
        )
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        map_x[~valid] = -1.0
        map_y[~valid] = -1.0
        map1, map2 = cv2.convertMaps(
            map_x, map_y, cv2.CV_16SC2, nninterpolation=False
        )
        return map1, map2, valid

    def _load_maps(self, path: Path | None) -> dict[str, np.ndarray] | None:
        cached = load_arrays(path, {"map1", "map2", "valid"})
        expected = (self.fisheye_height, self.fisheye_width)
        if cached is None or (
            cached["map1"].shape != expected + (2,)
            or cached["map2"].shape != expected
            or cached["valid"].shape != expected
        ):
            return None
        return cached

    def prepare(self) -> dict:
        if self._map1 is not None:
            return self.cache_report()

        self._forward.prepare()
        source_lut = self._forward._source_lut(
            self.fisheye_height, self.fisheye_width
        )
        forward_lut = self._forward._lut
        assert forward_lut is not None
        crop_box = source_lut.crop_box
        path = self._cache_path(crop_box)
        cached = self._load_maps(path)
        if cached is None:
            self._map1, self._map2, self._valid = self._build_maps(
                crop_box,
                forward_lut.longitude_bounds_deg,
                forward_lut.latitude_bounds_deg,
            )
            save_arrays(
                path,
                {"map1": self._map1, "map2": self._map2, "valid": self._valid},
            )
            self._cache_source = "computed" if path is not None else "disabled"
        else:
            self._map1 = cached["map1"].astype(np.int16, copy=False)
            self._map2 = cached["map2"].astype(np.uint16, copy=False)
            self._valid = cached["valid"].astype(bool, copy=False)
            self._cache_source = "disk"

        self._crop_box = crop_box
        self._panorama_shape = (
            crop_box[3] - crop_box[1],
            crop_box[2] - crop_box[0],
        )
        self._cache_file = path
        # The inverse fixed-point maps are now self-contained. Releasing the
        # forward LUT avoids retaining an extra copy for a reverse-only batch.
        self._forward._source_luts.clear()
        self._forward._lut = None
        return self.cache_report()

    def project(self, panorama: np.ndarray) -> np.ndarray:
        self.prepare()
        assert self._map1 is not None and self._map2 is not None
        assert self._valid is not None and self._panorama_shape is not None
        if panorama.shape[:2] != self._panorama_shape:
            raise ValueError(
                f"Expected panorama shape {self._panorama_shape}, got {panorama.shape[:2]}"
            )
        cv2 = self._cv2()
        flags = {
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }
        result = cv2.remap(
            panorama,
            self._map1,
            self._map2,
            flags[self.interpolation],
            borderMode=cv2.BORDER_CONSTANT,
        )
        result[~self._valid] = 0
        return result

    @property
    def valid_mask(self) -> np.ndarray:
        self.prepare()
        assert self._valid is not None
        return self._valid

    def cache_report(self) -> dict:
        size_bytes = (
            self._cache_file.stat().st_size
            if self._cache_file and self._cache_file.is_file()
            else None
        )
        return {
            "enabled": self.cache_dir is not None,
            "source": self._cache_source,
            "path": str(self._cache_file) if self._cache_file is not None else None,
            "size_bytes": size_bytes,
            "expected_panorama_shape": list(self._panorama_shape)
            if self._panorama_shape is not None
            else None,
            "fisheye_shape": [self.fisheye_height, self.fisheye_width],
            "crop_box": list(self._crop_box) if self._crop_box is not None else None,
        }
