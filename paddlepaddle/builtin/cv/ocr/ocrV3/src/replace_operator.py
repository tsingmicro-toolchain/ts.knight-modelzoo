import json
import torch
import onnx
import numpy as np
import logging
import argparse
from onnx import helper
from onnx import TensorProto
from replace import *
from onnx_quantize_tool.utils.constants import chip_map, CHIP_CONVERT_MAP

LOG = logging.getLogger(__name__)

# class_name : ReplaceClass
SPEED_OPS = {
    'Conv': ReplaceConv,
    'Input': ReplaceInput,
    'Reduction': ReplaceReduction,
    'MatMul': ReplaceMatMul,
    'EltwiseWeight': ReplaceEltwiseWeight,
    'Eltwise': ReplaceEltwise,
    'MulScalar': ReplaceMulScalar,
    'Activation': ReplaceActivation,
    'Prelu': ReplacePrelu,
    'BatchNorm': ReplaceBatchNorm,
    'L2Norm': ReplaceL2Norm,
    'Pool': ReplacePool,
    'Pad': ReplaceShape,
    'Roll': ReplaceShape,
    'Slice': ReplaceShape,
    'Clip': ReplaceShape,
    'Split': ReplaceShape,
    'Upsample': ReplaceShape,
    'BilinearInterpolation': ReplaceShape,
    'ChannelShuffle': ReplaceShape,
    'Softmax': ReplaceSoftmax,
    'Neg': ReplaceActivation,
    'LayerNormPyop': ReplaceLayernorm,
    'Concat': ReplaceConcat,
    'Scale': ReplaceScale,
    'QuantDequant': ReplaceDequant,
    'Abs': ReplaceAbs,
    'Where': ReplaceWhere,
    'AddMask': ReplaceWhere,
    'InstanceNormalization': ReplaceInstancenorm,
    'RMSNormalization': ReplaceRMSNorm,
    'RMSNormPyop': ReplaceRMSNorm,
}

class ReplaceBaseModel:
    def __init__(self, model, save_file):
        self.model = onnx.load(model)
        self.graph = self.model.graph
        self.save_file = save_file

    def get_chip(self):
        global version
        chip = self.model.producer_name

        for key, values in chip_map.items():
            for v in values:
                if v in chip:
                    return key

    def get_tensor_shape(self, tensor):
            dims = tensor.type.tensor_type.shape.dim
            n = len(dims)
            dim_new = []
            for i in range(n):
                if dims[i].dim_value == 0:
                    dim_new.append(dims[i].dim_param)
                else:
                    dim_new.append(dims[i].dim_value)
            return dim_new

    def init_shape_info(self):
        # model = onnx.shape_inference.infer_shapes(model)
        shape_dicts = {}
        for tensor in self.graph.input:
            shape_dicts[str(tensor.name)] = self.get_tensor_shape(tensor)
        for tensor in self.graph.value_info:
            shape_dicts[str(tensor.name)] = self.get_tensor_shape(tensor)
        for tensor in self.graph.output:
            shape_dicts[str(tensor.name)] = self.get_tensor_shape(tensor)
        return shape_dicts

    def get_nodes(self):
        node_list = []
        for node in self.graph.node:
            node_list.append(node)
        return node_list

    def get_output_dtype(self):
        return self.graph.output[0].type.tensor_type.elem_type

    def get_attribute(self, node, attr_name, default_value=None):
        found = [attr for attr in node.attribute if attr.name == attr_name]
        if found:
            if found[0].HasField('s'):
                return str(helper.get_attribute_value(found[0]), 'utf-8')
            else:
                return helper.get_attribute_value(found[0])
        return default_value

    def replace_pyop_operator(self):
        """ Replace the onnx quantized pyop operator, given the quantized model_file and save_file path.
            # Arguments
                model_file: the quantized onnx model_file path.
                save_file : the replaced model path you want to save.
            # Returns: None
            # Notes: the scrip only support [Input, AveragePool, GlobalAveragePool, MaxPool, Conv, Gemm,
                    eltwise-add, eltwise-mul, Pad, Activation] op, and chip-tx510, bit-8bit, float32 or float64.
            More support, continue to improve it.
        """

        nodes = self.get_nodes()
        dtype = self.get_output_dtype()
        if dtype==7: #混淆
            dtype = get_output_dtype()
        shape_dicts = self.init_shape_info()
        version = CHIP_CONVERT_MAP[self.get_chip()] if self.get_chip() in CHIP_CONVERT_MAP.keys() else self.get_chip()
        ReplacePyopBase.version = version
        ReplacePyopBase.dtype = dtype
        ReplacePyopBase.shape_dicts = shape_dicts
        ReplacePyopBase.graph = self.graph

        unsupport_node_list = []
        for node in nodes:
            if node.op_type == 'PyOp':
                pyop_type = self.get_attribute(node, 'pyop_type')
                class_name = self.get_attribute(node, 'class_name')
                replace_flag = False
                if class_name in SPEED_OPS:
                    replace_func = SPEED_OPS[class_name]
                    replace_flag = replace_func(node).replace()
                if replace_flag:
                    self.graph.node.remove(node)
                else:
                    unsupport_node_list.append(pyop_type)
        unsupport_node_list = set(unsupport_node_list)
        kwargs = {}
        if "producer_name" not in kwargs:
            kwargs = {"producer_name": self.model.producer_name,
                      "producer_version": " 1.0"}

        if "opset_imports" not in kwargs:
            opsets = [helper.make_opsetid("", 14)]
            opsets.append(helper.make_opsetid("ai.onnx.ml", 2))
            kwargs["opset_imports"] = opsets

        model_def = helper.make_model(self.graph, **kwargs)
        # 有没被替换的pyop算子跳过check
        try:
            onnx.checker.check_model(model_def)
        except Exception as e:
            e_str = str(e)
            if  'OpType: PyOp' not in e_str:
                raise e
        if len(unsupport_node_list) >= 1:
            LOG.warning('Unsupport pyop_types {}'.format(unsupport_node_list))
        onnx.save(model_def, self.save_file)
        return model_def

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tsingmicro Onnx Graph Tool')
    parser.add_argument('-m', '--model', type=str, help='the path of quantized model')
    parser.add_argument('-o', '--output', type=str, help='the output accelerated quantized model')
    # 解析参数
    args, unknown = parser.parse_known_args()
    if unknown:
        for i in unknown:
            print('Not support param: {}'.format(i))
        sys.exit(-1)
    if args.output is None:
        accelerate_model_path = args.model[:-5] + '_speed.onnx'
    else:
        accelerate_model_path = args.output
    replace = ReplaceBaseModel(args.model, accelerate_model_path)
    model_speed = replace.replace_pyop_operator()
    print('Sucessfully generating accelerate model: {}'.format(accelerate_model_path))