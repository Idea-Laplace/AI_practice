import numpy as np
import MyDeZero

gpu_enable = True
try:
    import cupy as cp
    cupy = cp
except ImportError:
    gpu_enable = False


def get_array_module(x):
    if isinstance(x, MyDeZero.Variable):
        x = x.data
    
    if not gpu_enable:
        return np
    xp = cp.get_array_module(x)
    return xp

def as_numpy(x):
    if isinstance(x, MyDeZero.Variable):
        x = x.data
    
    if np.isscalar(x):
        return np.array(x)

    elif isinstance(x, np.ndarray):
        return x

    return cp.asnumpy(x)
    
def as_cupy(x):
    if isinstance(x, MyDeZero.Variable):
        x = x.data
    
    if not gpu_enable:
        raise Exception('Cannot load CuPy, install cupy module.')
    
    return cp.asarray(x)