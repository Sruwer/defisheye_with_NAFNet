import numpy as np
from src.fisheye.camera_models import RadialFisheyeModel
from src.fisheye.projector import PerspectiveProjector, View
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

def stitcher(feather_px=0):
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
