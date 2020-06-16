import numpy as np

CYTHON_DTYPE = np.float32
ID_DTYPE = np.int32

try:
    __LIGHTFM_SETUP__
except NameError:
    from .model import LightFM

__version__ = '1.13.20'

__all__ = ['LightFM', 'datasets', 'evaluation']
