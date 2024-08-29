#!/usr/bin/env python3
import os
import sys
import onnx
import argparse
import numpy as np
import onnx_graphsurgeon as gs
def print_red_strings(info):
    print('\033[31m{}\033[0m'.format(info))
def print_green_strings(info):
    print('\033[32m{}\033[0m'.format(info))
def tensor2variable(tensor):
    dtype = np.float32 if tensor.dtype is None else tensor.dtype
    variable =tensor.to_variable(dtype=dtype,shape=tensor.shape)
    return variable
def cut_subgraph(args):
    model_path = args.model
    graph = gs.import_onnx(onnx.load(model_path))
    save_path = args.save_dir if args.save_dir else os.path.dirname(model_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    tensors = graph.tensors()
    input_names = args.input_names if args.input_names else [inp.name for inp in graph.inputs]
    output_names = args.output_names if args.output_names else [out.name for out in graph.outputs]
    inputs =[]
    outputs =[]
    for inp in input_names:
        if inp not in tensors:
            print_red_strings(f'Args input-names:{inp} not exist in model.')
            sys.exit(-1)
        inputs.append(tensor2variable(tensors[inp]))
    for out in output_names:
        if out not in tensors:
            print_red_strings(f'Args output-names:{fout} not exist in model.')
            sys.exit(-1)
        outputs.append(tensor2variable(tensors[out]))
    graph.inputs = inputs
    graph.outputs = outputs
    graph.cleanup()
    print_green_strings('Generated subgraph:{}'.format(os.path.join(save_path, f'{args.submodel_name}.onnx')))
    if gs.export_onnx(graph).ByteSize()> onnx.checker.MAXIMUM_PROTOBUF:
        location = os.path.basename(os.path.join(save_path, f'{args.submodel_name}.onnx'))+ ".data"
        onnx.save(gs.export_onnx(graph),
            os.path.join(save_path,f'{fargs.submodel_name}.onnx'),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=location)
    else:
        onnx.save(gs.export_onnx(graph),os.path.join(save_path, f'{args.submodel_name}.onnx'))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tsingmicro Onnx Graph Tool')
    parser.add_argument('-sn','--submodel-name',type=str,default='subgraph')
    parser.add_argument('-m','--model',type=str, help='the path of ori model')
    parser.add_argument('-s','--save-dir',type=str, help='the save path of subgraph')
    parser.add_argument('-in','--input-names', nargs='+', help='the input names of subgraph')
    parser.add_argument('-on', '--output-names', nargs='+', help='the output names of subgraph')
    # 解析参数
    args,unknown =parser.parse_known_args()
    if unknown :
        for i in unknown:
            print_red_strings('Not support param: {}'.format(i))
            sys.exit(-1)
    if args.input_names is None and args.output_names is None:
        print_red_strings('Must set args input-names or output-names.')
    if not os.path.exists(args.model):
        print_red_strings(f'Model:{args.model} not exists.')
    cut_subgraph(args)