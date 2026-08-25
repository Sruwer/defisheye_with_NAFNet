from __future__ import annotations
import numpy as np

class KnownGeometryStitcher:
    """Deterministically compose yaw-ordered perspective panels with overlap blending."""
    def __init__(self, overlap_fraction: float = .1, blend_strength: int = 5):
        if not 0 <= overlap_fraction < 1: raise ValueError("overlap_fraction must be in [0, 1)")
        self.overlap_fraction, self.blend_strength = overlap_fraction, blend_strength

    def stitch(self, images: list[np.ndarray], masks: list[np.ndarray] | None = None):
        if not images: raise ValueError("At least one image is required")
        h, w = images[0].shape[:2]
        if any(x.shape[:2] != (h,w) for x in images): raise ValueError("All panels must have equal size")
        overlap = int(round(w*self.overlap_fraction)); step = w-overlap
        width = w + step*(len(images)-1)
        accum = np.zeros((h,width,3), np.float32); weights = np.zeros((h,width), np.float32)
        masks = masks or [np.full((h,w),255,np.uint8) for _ in images]
        for i, (image, mask) in enumerate(zip(images,masks)):
            x0=i*step; weight=(mask.astype(np.float32)/255)
            if overlap and i: weight[:,:overlap] *= np.linspace(0,1,overlap,dtype=np.float32)[None,:]
            if overlap and i<len(images)-1: weight[:,-overlap:] *= np.linspace(1,0,overlap,dtype=np.float32)[None,:]
            accum[:,x0:x0+w] += image.astype(np.float32)*weight[...,None]
            weights[:,x0:x0+w] += weight
        valid=weights>1e-6
        result=np.zeros_like(accum,dtype=np.uint8)
        result[valid]=np.clip(accum[valid]/weights[valid,None],0,255).astype(np.uint8)
        return result, valid.astype(np.uint8)*255

def crop_to_valid_region(image: np.ndarray, mask: np.ndarray):
    ys,xs=np.nonzero(mask)
    if not len(xs): return image[:0,:0], mask[:0,:0]
    box=(int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1))
    x0,y0,x1,y1=box
    return image[y0:y1,x0:x1], mask[y0:y1,x0:x1]

