import os
import numpy as np
import argparse
import torch

def run():
    ed_out = np.load(opt.numpys[0])
    labels = ['_slilence_', '_unknown_', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    pred =np.argmax(ed_out,1)
    print('predict label:')
    print(labels[pred.flatten()[0]])
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--numpys', nargs='+', type=str, help='model output numpy')
    opt = parser.parse_args()

    run()