import onnx
from onnx import TensorProto
import numpy as np


CHIP_CONVERT_MAP = {
    'DT56': 'TX511',
    'DT57': 'TX232',
    'DT57B': 'TX232',
    'DT56B': 'DT53'
}
chip_map = {
    'TX231': ['TX2311R'],
    'TX232': ['TX232'],
    'TX510': ['TX5105C'],
    'TX511': ['TX5368AV200'],
    'DT53': ['TX5336AV200'],
    'DT56': ['TX5215CV200'],
    'DT57': ['TX5112CV201', 'TX5239CV201'],
    'DT57B': ['TX5110CV200'],
    'DT56B': ['DT56B']
}

