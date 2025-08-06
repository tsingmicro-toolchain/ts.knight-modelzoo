import sys
import glob
import os.path
import numpy as np

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from onnx_quantize_tool.common.register import onnx_infer_func
from onnx_quantize_tool.utils.constants import *

image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG',]
txt_extensions = ['*.npy']
bin_extensions = ['*.bin']
support_keys = ['input_name', 'quant_data_format', 'data_dir', 'color_space', 'mean', 'std', 'quantize_input_dtype', 'padding_constant_value']

def error_exit(log_info, exit_code=-1):
    print(log_info)
    sys.exit(exit_code)

def print_yellow_strings(info):
    print('\033[33m{}\033[0m'.format(info))

def check_args(executor):
    model_input = len(executor.input_nodes)
    config_input = len(executor.input_configs_dict)
    if model_input != config_input:
        error_exit('The model has {} inputs, but input-config only includes {}.'.format(model_input, config_input))

    for input_configs_dict in executor.input_configs_dict:
        icd = input_configs_dict.keys()
        for key in icd:
            if key not in support_keys:
                error_exit(f'Not support input-configs parameter: {key}')
        data_dir = input_configs_dict.get('data_dir', None)
        if data_dir is None or (not os.path.exists(data_dir)):
            error_exit('input-config error: data_dir:{} must exist.'.format(data_dir))
        quant_input_dtype = input_configs_dict.get('quantize_input_dtype', None)
        if quant_input_dtype:
            if quant_input_dtype not in ['float32', 'uint8']:
                error_exit('quantize-input-dtype error: {} not in [float32, uint8].'.format(quant_input_dtype))
            if quant_input_dtype == 'uint8':
                std = input_configs_dict.get('std', None)
                mean = input_configs_dict.get('mean', None)
                if std is None or mean is None:
                    error_exit('when quant_input_dtype is uint8, std and mean must set')
        input_name = input_configs_dict.get('input_name', None)
        if input_name is None or input_name not in executor.input_nodes:
            error_exit('input-config error: input_name:{} is set incorrectly.'.format(input_name))
        quant_data_format = input_configs_dict.get('quant_data_format', None)
        if quant_data_format is None or quant_data_format not in ['Image', 'Numpy', 'Bin']:
            error_exit('input-config error: quant_data_format:{} is set incorrectly.'.format(quant_data_format))

        if quant_data_format == 'Image':
            color_space = input_configs_dict.get('color_space', 'BGR')
            if color_space in ['RGB', 'BGR', 'GRAY']:
                std = input_configs_dict.get('std', None)
                mean = input_configs_dict.get('mean', None)
                std_mean_lens = 1 if color_space == 'GRAY' else 3
                if not executor.shape_dicts[input_name][1] == std_mean_lens:
                    error_exit('input-config error: input_name:{} color_space {} dose not correspond to the channel.'
                               .format(input_name, color_space))
                if std and mean:
                    if not (len(std) == std_mean_lens and len(mean) == std_mean_lens):
                        error_exit('std and mean length must correspond to the channel.')
                    else:
                        for i in range(len(std)):
                            if not (isinstance(std[i], (float, int)) and isinstance(mean[i], (float, int))):
                                error_exit('input-config error: mean or std must be number.')
                elif std is None and mean is None:
                    pass
                else:
                    print_yellow_strings('std and mean must be set concurrently, the norm will skip.')
            else:
                error_exit(
                    'input-config error: color_space {} not in [\'RGB\', \'BGR\', \'GRAY\'].'.format(color_space))
            if 'padding_constant_value' in input_configs_dict:
                if not isinstance(input_configs_dict['padding_constant_value'], int):
                    raise ValueError('input-config error: padding_constant_value must be integer.')

    return executor.input_configs_dict

def get_padding(image_shape, size):
    image_w, image_h = image_shape
    h, w = size
    new_w = int(image_w * min(w*1.0/image_w, h*1.0/image_h))
    new_h = int(image_h * min(w*1.0/image_w, h*1.0/image_h))

    dw = (w - new_w) / 2
    dh = (h - new_h) / 2
    padding = (
        int(round(dw - 0.1)),   # left
        int(round(dh - 0.1)),   # top
        int(round(dw + 0.1)),   # right
        int(round(dh + 0.1))    # bottom
    )
    new_shape = (new_h, new_w)
    return new_shape, padding

def load_input_data(executor):
    input_data_dict = {}
    for input_configs_dict in executor.input_configs_dict:
        transform = None
        input_name = input_configs_dict['input_name']
        input_shape = executor.shape_dicts[input_name]
        input_data_dict[input_name] = []
        quant_data_format = input_configs_dict['quant_data_format']
        data_dir = input_configs_dict['data_dir']
        padding_constant_value = input_configs_dict.get('padding_constant_value', None)
        quant_input_dtype = input_configs_dict.get('quantize_input_dtype', None)

        data_files = []
        if quant_data_format == 'Image':
            exts = image_extensions
        elif quant_data_format == 'Numpy':
            exts = txt_extensions
        elif quant_data_format == 'Bin':
            exts = bin_extensions
        for ext in exts:
            data_files.extend([os.path.basename(f) for f in glob.glob(f'{data_dir}/{ext}')])
        data_files = sorted(data_files)
        input_dtypes = executor.input_types
        if len(data_files) == 0:
            error_exit('input-config error: data_dir:{} is empty.'.format(data_dir))
        if quant_data_format == 'Image':
            std = input_configs_dict.get('std', None)
            mean = input_configs_dict.get('mean', None)
            image_h, image_w = input_shape[-2:]
            transforms_block = [
                   transforms.Resize((image_h, image_w), interpolation=InterpolationMode.BICUBIC)                
            ]
            if padding_constant_value is not None:
                transforms_block.append(transforms.Pad((0,0), fill=0, padding_mode='constant'))
            if quant_input_dtype == 'uint8':
                transforms_block.append(transforms.PILToTensor())
            else:
                transforms_block.append(transforms.ToTensor())

            if quant_input_dtype == 'float32' and std is not None and mean is not None:
                transforms_block.append(transforms.Normalize(tuple(np.array(mean)/255.), tuple(np.array(std)/255.)))
            transform = transforms.Compose(transforms_block)
        for data_file in data_files:
            data_path = os.path.join(data_dir, data_file)
            input_dtype = input_dtypes.get(input_name, 1)
            if quant_data_format == 'Numpy':
                data_cur = np.load(data_path)
            elif quant_data_format == 'Bin':
                data_cur = np.fromfile(data_path, dtype=NP_TYPE_DICT[input_dtype]).reshape(*input_shape)
            else:
                color_space = input_configs_dict.get('color_space', 'BGR')
                raw_image = Image.open(data_path)
                if color_space == 'BGR':
                    r, g, b = raw_image.split()
                    raw_image = Image.merge("RGB", (b,g,r))
                elif color_space == 'GRAY':
                    raw_image = raw_image.convert('L')
                if padding_constant_value is not None:
                    new_shape, padding = get_padding(raw_image.size, (image_h, image_w))
                    transforms_block[0] = transforms.Resize(new_shape, interpolation=InterpolationMode.BICUBIC)
                    transforms_block[1] = transforms.Pad(padding, fill=padding_constant_value, padding_mode='constant')
                    transform = transforms.Compose(transforms_block)
                data_cur = transform(raw_image).numpy()
            if np.ndim(data_cur) == len(input_shape) - 1:
                data_cur = np.expand_dims(data_cur, axis=0)

            input_data_dict[input_name].append(data_cur.astype(NP_TYPE_DICT[input_dtype]))

    return input_data_dict


@onnx_infer_func.register("infer_common")
def infer_common(executor):
    if not executor.shape_dicts:
        executor.init_shape_info()
    check_args(executor)
    input_data_dict = load_input_data(executor)
    total_len = max([len(input_data_dict[input_name]) for input_name in executor.input_nodes])
    with tqdm(total=total_len, desc='Calibrate data') as pbar:
        for index in range(total_len):
            input_datas = []
            for input_name in executor.input_nodes:
                if index < len(input_data_dict[input_name]):
                    input_datas.append(input_data_dict[input_name][index])
                else:
                    input_datas.append(input_data_dict[input_name][index%len(input_data_dict[input_name])])
            executor.forward(*input_datas)
            pbar.update(1)
            if index+1 == executor.iteration:
                break

    return None

