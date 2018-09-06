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
    return vars(parser.parse_args())

if __name__ == '__main__':
    parser = ParseLstmParams()
    parser = vars(parser)
    print(parser['input_length'])
    print(parser['input_size'])