import os
import numpy as np
import argparse
import torch

def run():
    numpys = opt.numpys
    output = torch.from_numpy(np.load(numpys[0])).data
    # measure accuracy and record loss
    topk=(1, 5)
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    dicts = {}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, 'data/labels.txt')) as f:
        lines = f.readlines()
        for line in lines:
            num, label = line.strip('\n').split(':')
            dicts[num] = label
    index = pred.numpy().flatten()
    print('predict label top5:')
    for idx in index:
        print(idx, dicts[str(idx)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--numpys', nargs='+', type=str, help='model output numpy')
    opt = parser.parse_args()

    run()