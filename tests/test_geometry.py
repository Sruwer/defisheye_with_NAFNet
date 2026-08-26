from pathlib import Path

import numpy as np
from src.fisheye.camera_models import RadialFisheyeModel
from src.fisheye.direct_panorama import DirectFisheyePanoramaProjector
from src.fisheye.projector import PerspectiveProjector, View
from src.pipeline import FisheyePanoramaPipeline
from src.stitching.known_geometry import KnownGeometryStitcher

def camera(): return RadialFisheyeModel(960,583,814,810,190,"equalarea")
def test_optical_axis_maps_to_principal_point():
    p,valid=camera().project_rays(np.array([[0.,0.,1.]])); assert valid[0]; np.testing.assert_allclose(p[0],[960,583],atol=1e-5)
def test_projection_center_and_shape():
    mx,my,valid=PerspectiveProjector(camera(),960,720,90,85).maps(View())
    assert mx.shape==my.shape==valid.shape==(720,960)
    np.testing.assert_allclose([mx[359:361,479:481].mean(),my[359:361,479:481].mean()],[960,583],atol=1)
    # Image orientation must be preserved: output top samples above the
    # principal point, while output bottom samples below it.
    assert my[0, 480] < 583 < my[-1, 480]

def test_projection_mask_excludes_pixels_outside_cropped_input():
    projector=PerspectiveProjector(camera(),160,120,90,85)
    image=np.zeros((400,400,3),np.uint8)
    _,mask=projector.project(image,View())
    # The configured optical center is outside this deliberately small crop.
    assert not mask.any()

def stitcher(feather_px=0, cache_dir=None):
    return KnownGeometryStitcher(
        views=[View(yaw=-50), View(yaw=0), View(yaw=50)],
        hfov_deg=90,
        vfov_deg=85,
        panorama_width=380,
        panorama_height=170,
        seam_method="nearest_axis",
        feather_px=feather_px,
        border_margin_px=0,
        blend="feather",
        cache_dir=cache_dir,
    )

def test_spherical_panorama_shape_and_angular_ownership():
    images=[np.full((120,160,3),x,np.uint8) for x in (20,100,220)]
    compositor=stitcher()
    result,mask=compositor.stitch(images)
    assert result.shape==(170,380,3)
    assert mask.shape==(170,380)

    lon_min,lon_max=compositor.last_debug["longitude_bounds_deg"]
    def x_at(longitude):
        return int((longitude-lon_min)/(lon_max-lon_min)*result.shape[1])
    y=result.shape[0]//2
    # The central panel does not get averaged across its 40-degree overlaps:
    # the nearest optical axis owns each ray.
    assert result[y,x_at(-40),0]==20
    assert result[y,x_at(0),0]==100
    assert result[y,x_at(40),0]==220
    assert compositor.last_debug["ownership"][y,x_at(-40)]==0
    assert compositor.last_debug["ownership"][y,x_at(0)]==1
    assert compositor.last_debug["ownership"][y,x_at(40)]==2

def test_blending_is_limited_to_narrow_seam_bands():
    images=[np.full((120,160,3),x,np.uint8) for x in (20,100,220)]
    compositor=stitcher(feather_px=4)
    compositor.stitch(images)
    weights=compositor.last_debug["blend_weights"]
    y=weights.shape[1]//2
    contributors=(weights[:,y,:]>1e-4).sum(axis=0)
    # Two seams, each mixed only for approximately twice feather_px.
    assert np.count_nonzero(contributors>1)<=4*compositor.feather_px+4

def test_projection_lut_is_loaded_from_disk(tmp_path):
    view=View(yaw=-50,pitch=-6)
    first=PerspectiveProjector(camera(),160,120,90,85,cache_dir=tmp_path)
    first_maps=first.maps(view)
    first_entry=first.cache_report()["entries"][0]
    assert first_entry["source"]=="computed"
    assert Path(first_entry["path"]).is_file()

    second=PerspectiveProjector(camera(),160,120,90,85,cache_dir=tmp_path)
    second_maps=second.maps(view)
    assert second.cache_report()["entries"][0]["source"]=="disk"
    for expected,actual in zip(first_maps,second_maps):
        np.testing.assert_array_equal(actual,expected)

def test_panorama_lut_is_loaded_and_geometry_change_invalidates_key(tmp_path):
    first=stitcher(cache_dir=tmp_path)
    first_report=first.prepare(120,160)
    assert first_report["source"]=="computed"

    second=stitcher(cache_dir=tmp_path)
    second_report=second.prepare(120,160)
    assert second_report["source"]=="disk"
    assert second_report["path"]==first_report["path"]

    changed=KnownGeometryStitcher(
        views=[View(yaw=-50),View(yaw=0),View(yaw=50)],
        hfov_deg=91,
        vfov_deg=85,
        panorama_width=380,
        panorama_height=170,
        seam_method="nearest_axis",
        blend="feather",
        cache_dir=tmp_path,
    )
    changed_report=changed.prepare(120,160)
    assert changed_report["source"]=="computed"
    assert changed_report["path"]!=first_report["path"]

def test_prepared_luts_remain_in_memory(tmp_path):
    view=View(yaw=-50,pitch=-6)
    projector=PerspectiveProjector(camera(),160,120,90,85,cache_dir=tmp_path)
    projection_maps=projector.maps(view)
    Path(projector.cache_report()["entries"][0]["path"]).unlink()
    assert projector.maps(view) is projection_maps

    compositor=stitcher(cache_dir=tmp_path)
    panorama_geometry=compositor._get_geometry(120,160)
    Path(compositor.cache_report()["path"]).unlink()
    assert compositor._get_geometry(120,160) is panorama_geometry

def test_identity_pipeline_stitches_only_once():
    config={
        "camera": {
            "model": "equalarea", "cx": 50.0, "cy": 50.0,
            "radius_x": 48.0, "radius_y": 48.0,
            "fisheye_fov_deg": 180.0,
        },
        "geometry_cache": {"enabled": False},
        "projection": {
            "width": 40, "height": 30, "hfov_deg": 70.0, "vfov_deg": 60.0,
            "interpolation": "linear", "views": [{"yaw": 0.0}],
        },
        "restoration": {"enabled": False},
        "stitching": {
            "backend": "known_geometry", "panorama_width": 50,
            "panorama_height": 40, "ownership": "nearest_axis",
            "seam": {"method": "nearest_axis", "feather_px": 0},
            "border_margin_px": 0, "blend": "feather",
        },
    }
    pipeline=FisheyePanoramaPipeline(config,"none")
    original_stitch=pipeline.stitcher.stitch
    calls=0
    def counted_stitch(*args,**kwargs):
        nonlocal calls
        calls+=1
        return original_stitch(*args,**kwargs)
    pipeline.stitcher.stitch=counted_stitch
    result=pipeline.process(np.full((100,100,3),127,np.uint8))
    assert calls==1
    assert result.panorama_raw is not None
    np.testing.assert_array_equal(result.panorama,result.panorama_raw)

def test_direct_panorama_lut_is_reused_from_disk_and_memory(tmp_path):
    kwargs={
        "camera": RadialFisheyeModel(50,50,48,48,180,"equalarea"),
        "views": [View(yaw=-35),View(yaw=0),View(yaw=35)],
        "hfov_deg": 70,
        "vfov_deg": 60,
        "panorama_width": 100,
        "panorama_height": 50,
        "cache_dir": tmp_path,
    }
    image=np.full((100,100,3),127,np.uint8)
    first=DirectFisheyePanoramaProjector(**kwargs)
    assert first.prepare()["source"]=="computed"
    first_result=first.project(image)
    source_lut=first._source_luts[(100,100)]
    first.project(image)
    assert first._source_luts[(100,100)] is source_lut

    second=DirectFisheyePanoramaProjector(**kwargs)
    assert second.prepare()["source"]=="disk"
    second_result=second.project(image)
    np.testing.assert_array_equal(second_result,first_result)
