import numpy as np
import torch

def pack_scale_numpy(scale):
    scale_type = type(scale)
    if hasattr(scale, 'shape'):
        return scale
    elif scale_type in [int, float]:
        return np.array([scale])
    elif scale_type in [list]:
        return np.array(scale)
    else:
        raise RuntimeError('pack_scale_numpy Unsupport type:{}'.format(scale_type))
def pack_scale_tensor(scale, dtype=np.float32):
    scale_type = type(scale)
    if torch.is_tensor(scale):
        return scale
    elif isinstance(scale, np.ndarray):
        return torch.from_numpy(scale)
    elif scale_type in [int, float]:
        return torch.from_numpy(np.array([scale], dtype=dtype))
    elif scale_type in [list]:
        return torch.from_numpy(np.array(scale, dtype=dtype))
    else:
        raise RuntimeError('pack_scale_tensor Unsupport type:{}'.format(scale_type))