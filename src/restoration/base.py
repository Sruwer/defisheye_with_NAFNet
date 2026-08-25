from abc import ABC, abstractmethod
import numpy as np
class ImageRestorer(ABC):
    @abstractmethod
    def restore(self, image: np.ndarray) -> np.ndarray: ...

