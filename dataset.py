import os
from glob import glob
import numpy as np

def parse_label(file_path):
    name = os.path.basename(file_path)
    return int(name[5]) #get x from "classx_...."

def parse_file(file_path):
    x = []
    with open(file_path, 'r') as f:
        for line in f:
            x.append([float(val) for val in line.strip().split(' ')])
    return x

def parse_data(root_dir, mode):
    '''
    mode : 'train', 'val', 'test'
    '''
    file_list = glob(os.path.join(root_dir, '*'))
    file_list = [x for x in file_list if mode in os.path.basename(x)]
    X = []
    Y = []
    for file_path in file_list:
        x_class = parse_file(file_path)
        y_class = [parse_label(file_path)] * len(x_class)
        X += x_class
        Y += y_class
    return np.array(X), np.array(Y)


if __name__ == '__main__':
    from pprint import pprint

    root_dir=os.path.join('dataset', 'linearlySeparable')
    X_train, Y_train = parse_data(root_dir, 'train')
    X_val, Y_val = parse_data(root_dir, 'val')
    X_test, Y_test = parse_data(root_dir, 'test')

    print(X_train.shape, Y_train.shape)
    
