from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..fisheye.projector import View, _rotation
from ..geometry_cache import cache_path, load_arrays, save_arrays


def _local_to_world(rays: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Apply rotation without macOS Accelerate's noisy batched matmul path."""
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
class _WarpGeometry:
    map_x: list[np.ndarray]
    map_y: list[np.ndarray]
    geometric_valid: list[np.ndarray]
    selection_valid: list[np.ndarray]
    angles: np.ndarray
    longitude_bounds_deg: tuple[float, float]
    latitude_bounds_deg: tuple[float, float]


class KnownGeometryStitcher:
    """Compose perspective views on one equirectangular canvas.

    Every input panel was rendered with a known view rotation and FOV, so its
    pixels are inverse-warped to the panorama instead of being approximated by
    a horizontal translation. A single view owns most panorama pixels. The
    large geometric overlap is used to choose a seam; blending is restricted to
    a narrow band around that seam to avoid ghosts between restored panels.
    """

    def __init__(
        self,
        views: list[View],
        hfov_deg: float,
        vfov_deg: float,
        panorama_width: int,
        panorama_height: int,
        *,
        projection: str = "equirectangular",
        ownership: str = "nearest_axis",
        seam_method: str = "graphcut",
        seam_search_band_deg: float = 4.0,
        feather_px: int = 20,
        border_margin_px: int = 32,
        blend: str = "multiband",
        blend_levels: int = 4,
        cache_dir: str | Path | None = None,
    ):
        if not views:
            raise ValueError("At least one view is required")
        if not 0 < hfov_deg < 180 or not 0 < vfov_deg < 180:
            raise ValueError("Perspective FOV must be between 0 and 180 degrees")
        if panorama_width <= 0 or panorama_height <= 0:
            raise ValueError("Panorama dimensions must be positive")
        if projection != "equirectangular":
            raise ValueError("Only projection=equirectangular is currently supported")
        if ownership != "nearest_axis":
            raise ValueError("Only ownership=nearest_axis is currently supported")
        if seam_method not in {"nearest_axis", "graphcut"}:
            raise ValueError("seam_method must be nearest_axis or graphcut")
        if blend not in {"feather", "multiband"}:
            raise ValueError("blend must be feather or multiband")
        if seam_search_band_deg < 0 or feather_px < 0 or border_margin_px < 0:
            raise ValueError("Seam and border sizes cannot be negative")
        if blend_levels < 1:
            raise ValueError("blend_levels must be at least 1")

        self.views = list(views)
        self.hfov_deg = float(hfov_deg)
        self.vfov_deg = float(vfov_deg)
        self.panorama_width = int(panorama_width)
        self.panorama_height = int(panorama_height)
        self.projection = projection
        self.ownership = ownership
        self.seam_method = seam_method
        self.seam_search_band_deg = float(seam_search_band_deg)
        self.feather_px = int(feather_px)
        self.border_margin_px = int(border_margin_px)
        self.blend = blend
        self.blend_levels = int(blend_levels)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

        self._geometry_key: tuple[int, int] | None = None
        self._geometry: _WarpGeometry | None = None
        self._geometry_cache_source = "not_prepared"
        self._geometry_cache_path: Path | None = None
        self.last_debug: dict[str, object] = {}

    @staticmethod
    def _cv2():
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required; install with: pip install -e .") from exc
        return cv2

    def _frustum_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return unwrapped longitude and latitude bounds of all view borders."""
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
            lon = np.rad2deg(np.arctan2(world[:, 0], world[:, 2]))
            lon = reference + (lon - reference + 180.0) % 360.0 - 180.0
            lat = np.rad2deg(np.arcsin(np.clip(-world[:, 1], -1.0, 1.0)))
            longitudes.append(lon)
            latitudes.append(lat)

        all_lon = np.concatenate(longitudes)
        all_lat = np.concatenate(latitudes)
        lon_bounds = (float(all_lon.min()), float(all_lon.max()))
        lat_bounds = (float(all_lat.min()), float(all_lat.max()))
        return lon_bounds, lat_bounds

    def _build_geometry(self, panel_height: int, panel_width: int) -> _WarpGeometry:
        lon_bounds, lat_bounds = self._frustum_bounds()
        lon_step = (lon_bounds[1] - lon_bounds[0]) / self.panorama_width
        lat_step = (lat_bounds[1] - lat_bounds[0]) / self.panorama_height
        lon = np.deg2rad(
            lon_bounds[0] + (np.arange(self.panorama_width) + 0.5) * lon_step
        )
        lat = np.deg2rad(
            lat_bounds[1] - (np.arange(self.panorama_height) + 0.5) * lat_step
        )
        longitude, latitude = np.meshgrid(lon, lat)
        cos_lat = np.cos(latitude)
        rays = np.stack(
            (
                cos_lat * np.sin(longitude),
                -np.sin(latitude),
                cos_lat * np.cos(longitude),
            ),
            axis=-1,
        )

        tan_h = np.tan(np.deg2rad(self.hfov_deg) / 2)
        tan_v = np.tan(np.deg2rad(self.vfov_deg) / 2)
        map_x: list[np.ndarray] = []
        map_y: list[np.ndarray] = []
        geometric_valid: list[np.ndarray] = []
        interior_valid: list[np.ndarray] = []
        angle_maps: list[np.ndarray] = []

        for view in self.views:
            rotation = _rotation(view)
            local = _world_to_local(rays, rotation)
            z = local[..., 2]
            safe_z = np.where(np.abs(z) > 1e-12, z, 1.0)
            plane_x = local[..., 0] / safe_z
            plane_y = local[..., 1] / safe_z
            u = ((plane_x / tan_h + 1.0) * panel_width / 2.0) - 0.5
            v = ((plane_y / tan_v + 1.0) * panel_height / 2.0) - 0.5
            valid = (
                (z > 0)
                & (u >= 0)
                & (u <= panel_width - 1)
                & (v >= 0)
                & (v <= panel_height - 1)
            )
            edge_distance = np.minimum.reduce(
                (u, panel_width - 1 - u, v, panel_height - 1 - v)
            )
            interior = valid & (edge_distance >= self.border_margin_px)
            center = rotation[:, 2]
            cosine = np.clip(np.sum(rays * center, axis=-1), -1.0, 1.0)

            map_x.append(u.astype(np.float32))
            map_y.append(v.astype(np.float32))
            geometric_valid.append(valid)
            interior_valid.append(interior)
            angle_maps.append(np.arccos(cosine).astype(np.float32))

        valid_stack = np.stack(geometric_valid)
        interior_stack = np.stack(interior_valid)
        # Avoid a panel border only where another panel has a safe sample for
        # the same ray. Outer panorama boundaries therefore remain covered.
        has_interior = np.any(interior_stack, axis=0, keepdims=True)
        selection_valid = np.where(has_interior, interior_stack, valid_stack)
        angles = np.stack(angle_maps)
        return _WarpGeometry(
            map_x,
            map_y,
            [x for x in valid_stack],
            [x for x in selection_valid],
            angles,
            lon_bounds,
            lat_bounds,
        )

    def _cache_path(self, panel_height: int, panel_width: int) -> Path | None:
        return cache_path(
            self.cache_dir,
            "perspective-panorama-v1",
            {
                "views": [
                    {"yaw": view.yaw, "pitch": view.pitch, "roll": view.roll}
                    for view in self.views
                ],
                "hfov_deg": self.hfov_deg,
                "vfov_deg": self.vfov_deg,
                "panel_width": panel_width,
                "panel_height": panel_height,
                "panorama_width": self.panorama_width,
                "panorama_height": self.panorama_height,
                "projection": self.projection,
                "border_margin_px": self.border_margin_px,
            },
        )

    def _load_geometry(self, path: Path | None) -> _WarpGeometry | None:
        names = {
            "map_x",
            "map_y",
            "geometric_valid",
            "selection_valid",
            "angles",
            "longitude_bounds_deg",
            "latitude_bounds_deg",
        }
        cached = load_arrays(path, names)
        if cached is None:
            return None
        expected = (len(self.views), self.panorama_height, self.panorama_width)
        if any(cached[name].shape != expected for name in names - {
            "longitude_bounds_deg", "latitude_bounds_deg"
        }):
            return None
        if cached["longitude_bounds_deg"].shape != (2,) or cached["latitude_bounds_deg"].shape != (2,):
            return None
        return _WarpGeometry(
            [x.astype(np.float32, copy=False) for x in cached["map_x"]],
            [x.astype(np.float32, copy=False) for x in cached["map_y"]],
            [x.astype(bool, copy=False) for x in cached["geometric_valid"]],
            [x.astype(bool, copy=False) for x in cached["selection_valid"]],
            cached["angles"].astype(np.float32, copy=False),
            tuple(float(x) for x in cached["longitude_bounds_deg"]),
            tuple(float(x) for x in cached["latitude_bounds_deg"]),
        )

    @staticmethod
    def _save_geometry(path: Path | None, geometry: _WarpGeometry) -> None:
        save_arrays(
            path,
            {
                "map_x": np.stack(geometry.map_x),
                "map_y": np.stack(geometry.map_y),
                "geometric_valid": np.stack(geometry.geometric_valid),
                "selection_valid": np.stack(geometry.selection_valid),
                "angles": geometry.angles,
                "longitude_bounds_deg": np.asarray(geometry.longitude_bounds_deg),
                "latitude_bounds_deg": np.asarray(geometry.latitude_bounds_deg),
            },
        )

    def _get_geometry(self, panel_height: int, panel_width: int) -> _WarpGeometry:
        key = (panel_height, panel_width)
        if self._geometry is None or self._geometry_key != key:
            path = self._cache_path(panel_height, panel_width)
            self._geometry = self._load_geometry(path)
            if self._geometry is None:
                self._geometry = self._build_geometry(panel_height, panel_width)
                self._save_geometry(path, self._geometry)
                self._geometry_cache_source = "computed" if path is not None else "disabled"
            else:
                self._geometry_cache_source = "disk"
            self._geometry_cache_path = path
            self._geometry_key = key
        return self._geometry

    def prepare(self, panel_height: int, panel_width: int) -> dict:
        self._get_geometry(panel_height, panel_width)
        return self.cache_report()

    def cache_report(self) -> dict:
        return {
            "enabled": self.cache_dir is not None,
            "source": self._geometry_cache_source,
            "path": str(self._geometry_cache_path) if self._geometry_cache_path is not None else None,
        }

    def _warp_inputs(
        self,
        images: list[np.ndarray],
        masks: list[np.ndarray],
        geometry: _WarpGeometry,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        cv2 = self._cv2()
        warped_images: list[np.ndarray] = []
        warped_valid: list[np.ndarray] = []
        for image, mask, mx, my, geometric_valid in zip(
            images, masks, geometry.map_x, geometry.map_y, geometry.geometric_valid
        ):
            warped = cv2.remap(
                image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
            )
            warped_mask = cv2.remap(
                mask, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
            )
            valid = geometric_valid & (warped_mask > 0)
            warped[~valid] = 0
            warped_images.append(warped)
            warped_valid.append(valid)
        return warped_images, np.stack(warped_valid)

    @staticmethod
    def _nearest_owner(angles: np.ndarray, valid: np.ndarray) -> np.ndarray:
        costs = np.where(valid, angles, np.inf)
        owner = np.argmin(costs, axis=0).astype(np.int16)
        owner[~np.any(valid, axis=0)] = -1
        return owner

    def _graphcut_owner(
        self,
        warped_images: list[np.ndarray],
        angles: np.ndarray,
        valid: np.ndarray,
        nearest_owner: np.ndarray,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        cv2 = self._cv2()
        if not hasattr(cv2, "detail_GraphCutSeamFinder"):
            raise RuntimeError(
                "This OpenCV build has no GraphCut seam finder; use seam.method=nearest_axis"
            )

        best_angle = np.min(np.where(valid, angles, np.inf), axis=0)
        # The angular cost difference grows at about twice the distance from a
        # midpoint between equal-yaw views. Doubling keeps GraphCut inside the
        # requested seam-search corridor.
        max_delta = np.deg2rad(2.0 * self.seam_search_band_deg)
        view_indices = np.arange(len(self.views))[:, None, None]
        candidates = valid & (
            (angles <= best_angle[None, ...] + max_delta)
            | (view_indices == nearest_owner[None, ...])
        )
        seam_masks = [cv2.UMat(x.astype(np.uint8) * 255) for x in candidates]
        float_images = [x.astype(np.float32) for x in warped_images]
        corners = [(0, 0)] * len(float_images)
        finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
        finder.find(float_images, corners, seam_masks)
        cut_masks = [x.get() > 0 for x in seam_masks]
        claims = np.stack(cut_masks) & valid

        # Resolve rare overlaps with angular quality and retain the nearest
        # valid owner in any gaps returned by GraphCut.
        claim_costs = np.where(claims, angles, np.inf)
        owner = np.argmin(claim_costs, axis=0).astype(np.int16)
        no_claim = ~np.any(claims, axis=0)
        owner[no_claim] = nearest_owner[no_claim]
        owner[~np.any(valid, axis=0)] = -1
        return owner, [x.astype(np.uint8) * 255 for x in cut_masks]

    def _feather_weights(self, owner: np.ndarray, valid: np.ndarray) -> np.ndarray:
        cv2 = self._cv2()
        weights: list[np.ndarray] = []
        for index in range(len(self.views)):
            owned = owner == index
            if self.feather_px == 0 or not np.any(owned):
                weight = owned.astype(np.float32)
            else:
                distance_outside = cv2.distanceTransform(
                    (~owned).astype(np.uint8), cv2.DIST_L2, 3
                )
                weight = np.clip(
                    1.0 - distance_outside / float(self.feather_px), 0.0, 1.0
                )
            weights.append(weight * valid[index])
        result = np.stack(weights)
        total = result.sum(axis=0, keepdims=True)
        return np.divide(result, total, out=np.zeros_like(result), where=total > 1e-6)

    @staticmethod
    def _feather_blend(images: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
        accum = np.zeros_like(images[0], dtype=np.float32)
        for image, weight in zip(images, weights):
            accum += image.astype(np.float32) * weight[..., None]
        return np.clip(accum, 0, 255).astype(np.uint8)

    def _multiband_blend(
        self, images: list[np.ndarray], weights: np.ndarray
    ) -> np.ndarray:
        cv2 = self._cv2()
        blender = cv2.detail_MultiBandBlender()
        blender.setNumBands(self.blend_levels)
        blender.prepare((0, 0, self.panorama_width, self.panorama_height))
        for image, weight in zip(images, weights):
            mask = np.clip(np.rint(weight * 255.0), 0, 255).astype(np.uint8)
            blender.feed(image.astype(np.int16), mask, (0, 0))
        result, _ = blender.blend(None, None)
        return np.clip(result, 0, 255).astype(np.uint8)

    def stitch(
        self, images: list[np.ndarray], masks: list[np.ndarray] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if not images:
            raise ValueError("At least one image is required")
        if len(images) != len(self.views):
            raise ValueError("The number of images must match the configured views")
        panel_height, panel_width = images[0].shape[:2]
        if any(image.shape[:2] != (panel_height, panel_width) for image in images):
            raise ValueError("All panels must have equal size")
        if any(image.ndim != 3 or image.shape[2] != 3 for image in images):
            raise ValueError("All panels must be three-channel images")

        if masks is None:
            masks = [np.full((panel_height, panel_width), 255, np.uint8) for _ in images]
        if len(masks) != len(images):
            raise ValueError("The number of masks must match the images")
        if any(mask.shape != (panel_height, panel_width) for mask in masks):
            raise ValueError("Every mask must match its panel size")

        geometry = self._get_geometry(panel_height, panel_width)
        warped_images, source_valid = self._warp_inputs(images, masks, geometry)
        configured_selection = np.stack(geometry.selection_valid)
        selection_valid = source_valid & configured_selection
        # A supplied mask can remove all preferred samples. Fall back to any
        # geometrically valid source instead of creating a hole.
        has_selected = np.any(selection_valid, axis=0, keepdims=True)
        selection_valid = np.where(has_selected, selection_valid, source_valid)
        nearest_owner = self._nearest_owner(geometry.angles, selection_valid)

        graphcut_masks: list[np.ndarray] | None = None
        if self.seam_method == "graphcut" and len(images) > 1:
            owner, graphcut_masks = self._graphcut_owner(
                warped_images,
                geometry.angles,
                selection_valid,
                nearest_owner,
            )
        else:
            owner = nearest_owner

        weights = self._feather_weights(owner, source_valid)
        if self.blend == "multiband" and len(images) > 1:
            result = self._multiband_blend(warped_images, weights)
        else:
            result = self._feather_blend(warped_images, weights)

        valid = np.any(source_valid, axis=0)
        result[~valid] = 0
        self.last_debug = {
            "ownership": owner.copy(),
            "blend_weights": weights.copy(),
            "warped_masks": [x.astype(np.uint8) * 255 for x in source_valid],
            "graphcut_masks": graphcut_masks,
            "longitude_bounds_deg": geometry.longitude_bounds_deg,
            "latitude_bounds_deg": geometry.latitude_bounds_deg,
        }
        return result, valid.astype(np.uint8) * 255


def crop_to_valid_region(image: np.ndarray, mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return image[:0, :0], mask[:0, :0]
    box = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]
