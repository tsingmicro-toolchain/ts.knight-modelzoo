# -*- coding: UTF-8 -*-
import numpy as np
import argparse
import os
import glob
import cv2
DEBUG=0
from tqdm import tqdm
from ppdet.core.workspace import create, load_config
from ppdet.data.source.category import get_categories

parser = argparse.ArgumentParser(description='make input data scripts')
parser.add_argument(
    '--input',
    type=str,
    default='./',
    help='path of images')

parser.add_argument('--outpath',
                    type=str,
                    default='./',
                    help='the path of output bin')
parser.add_argument('--proc_mode',
                    type=str,
                    default='onnx',
                    help='the proc mode as onnx2onnx')
 
FLAGS, unparsed = parser.parse_known_args()

def write_infer_numpy_to_file(data, fileName, is_float=False, dim=4):
    # N=1
    if DEBUG:
        print(f'write_infer_numpy_to_file shape:{data.shape} ,fileName:{fileName}, is_float:{is_float},dim:{dim}')
    N,C, H, W=1,1,1,1
    if dim==4:
        N,C, H, W = data.shape
    else:
        C, H, W = data.shape
    file = open(fileName, 'w+')
    head = 'SHAPE:(N:{0}, C:{1}, H:{2}, W:{3})\n'.format(N, C, H, W)
    file.write(head)
    for i in range(C):
        str_C = ""
        file.write('C[{0}]:\n'.format(i))
        for j in range(H):
            for k in range(W):
                if is_float:
                    if dim==4:
                        str_C = str_C + '{0},'.format(float(data[0][i][j][k]))
                    else:
                        str_C = str_C + '{0},'.format(float(data[i][j][k]))
                else:
                    if dim==4:
                        str_C = str_C + '{0},'.format(int(data[0][i][j][k]))
                    else:
                        str_C = str_C + '{0},'.format(int(data[i][j][k]))
            str_C = str_C + '\n'
        file.write(str_C)
    file.close()

# 仅将第一条数据进行转换
if __name__ == '__main__':
    input = FLAGS.input
    outpath = FLAGS.outpath
    proc_mode= FLAGS.proc_mode
    # input_data = np.load(input).astype(np.float32)
    input_name = glob.glob(os.path.join(input, "*.jpg"))
    input_name.extend(glob.glob(os.path.join(input, "*.JPEG")))
    input_name.extend(glob.glob(os.path.join(input, "*.png")))
    input_name = sorted(input_name, key=os.path.getctime)
    Num=10

    root = os.path.dirname(os.path.dirname(__file__))
    font = os.path.join(root, 'src/yolov6/utils/Arial.ttf')
    config = os.path.dirname(__file__)
    cfg = load_config(config + '/configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml')
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    print(outpath)
    capital_mode = 'Test'
    dataset = create(
            '{}Dataset'.format(capital_mode))()
    dataset.set_images([input])
    loader = create('TestReader')(dataset, 0)
    
    imid2path = dataset.get_imid2path()

    anno_file = dataset.get_anno()
    clsid2catid, catid2name = get_categories(
        cfg.metric, anno_file=anno_file)

    # Run Infer
    #self.status['mode'] = 'test'
    #self.model.eval()
    if cfg.get('print_flops', False):
        flops_loader = create('TestReader')(dataset, 0)
        self._flops(flops_loader)
    results = []
    for step_id, data in enumerate(tqdm(loader)):
        # forward
        input_data = data['image'].numpy()*255
        input_data.astype(np.uint8).flatten().tofile(os.path.join(outpath,"model_input.bin"))
        outpath = os.path.join(outpath,"model_input.bin")
        print(f"Success to save bin {outpath}")
        break