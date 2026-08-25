import numpy as np
from .base import ImageRestorer
class IdentityRestorer(ImageRestorer):
    def restore(self, image: np.ndarray) -> np.ndarray:
        return image.copy()

