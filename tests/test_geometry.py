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
def test_three_panel_result_shape():
    images=[np.full((720,960,3),x,np.uint8) for x in (20,40,60)]
    result,mask=KnownGeometryStitcher(.1).stitch(images)
    assert result.shape==(720,2688,3); assert mask.shape==(720,2688); assert mask.all()

