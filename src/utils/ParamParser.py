# -*- coding: utf-8 -*-  
# Created Date: Saturday September 1st 2018
# Author: duxin
# Email: duxin_be@outlook.com
# Github: @Dosann
# -------------------------------

import argparse

def ParseLstmParams():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_length', type = int,
                        help = 'serie length of input serie')
    parser.add_argument('input_size', type = int,
                        help = 'dimension of input vector')
    parser.add_argument('--test', type = bool, default = False,
                        help = 'train phase')
    parser.add_argument('test_length', type = int, default = 100,
                        help = '# of timesteps to forecast when test')
    return vars(parser.parse_args())

if __name__ == '__main__':
    parser = ParseLstmParams()
    print(parser['input_length'])
    print(parser['input_size'])
    print(parser)