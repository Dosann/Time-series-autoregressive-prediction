# -*- coding: utf-8 -*-  
# Created Date: Saturday September 1st 2018
# Author: duxin
# Email: duxin_be@outlook.com
# Github: @Dosann
# -------------------------------

import os
import sys
import tsap
from tsap.solver.LstmSolver import LstmSolverKeras
from tsap.model import sequential
from tsap.predictor import DeterministicAutoregressivePredictor as dap
from tsap.solver.templates.LstmSolverStructures \
    import SolverStructure1, SolverStructure2, SolverStructure3
from tsap.utils import data_util
import numpy as np
import pandas as pd
import argparse
import matplotlib

import platform
system = platform.system()
if system != "Windows":
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


def RemoveQuotes(string): 
    # Quotes will be appended on head/tail of input string arguments.
    # This function removes them
    if string[0] in ['"', "'"]:
        string = string[1:]
    if string[-1] in ['"', "'"]:
        string = string[:-1]
    return string

def ParseLstmParams():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_length', type = int,
                        help = 'serie length of input serie')
    parser.add_argument('input_size', type = int,
                        help = 'dimension of input vector')
    parser.add_argument('--data_path', type = str,
                        help = 'path of data (raw sequential numpy array)')
    parser.add_argument('--model_path', type = str,
                        help = 'path of model\n'
                               '\tformat like: "../model/lstm1.solver')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type = str, default = None)
    # nn construction
    parser.add_argument('--Solver', type = str,
                        help = 'Solver object name')
    parser.add_argument('--hidden_units', type = int,
                        help = '# of hidden units in LSTM')
    # train
    parser.add_argument('--fit_generator', action = 'store_true', default = False,
                        help = '(train phase) use generator to feed data')
    parser.add_argument('--end_time', type = str, default = None,
                        help = '(train phase, optional) time to end training.\n'
                               '\tformat like: "20180910 00:00:00')
    parser.add_argument('--stages', type = int, default = 1,
                        help = '(train phase) # of stages when training. '
                               'A stage includes one of several epochs')
    parser.add_argument('--epochs', type = int,
                        help = '(train phase) # of epochs when training')
    parser.add_argument('--batch_size', type = int,
                        help = '(train phase) batch size when training')
    parser.add_argument('--batches_per_epoch', type = int, 
                        help = '(train phase) # of batches per epoch when training')
    # test
    parser.add_argument('--test', action = 'store_true', default = False,
                        help = '(test phase) start test phase')
    parser.add_argument('--test_length', type = int,
                        help = '(test phase) # of timesteps to forecast when test')
    parser.add_argument('--n_figs', type = int, default = 1,
                        help = '(test phase) # of dimensions to draw when test')
    params = vars(parser.parse_args())

    if 'data_path' not in params:
        print("parameter '--data_path' is not specified.")
    if 'model_path' not in params:
        print("parameter '--model_path' is not specified.")
    if 'Solver' not in params:
        print("parameter '--Solver' is not specified.")
    if 'hidden_units' not in params:
        print("parameter '--hidden_units' is not specified.")
    if params['test']: # test phase
        if 'test_length' not in params:
            print("parameter '--test_length' is not specified.")
    else: # train phase
        if 'stages' not in params:
            print("parameter '--stages' is not specified.")
        if 'epochs' not in params:
            print("parameter '--epochs' is not specified.")
        if 'batch_size' not in params:
            print("parameter '--batch_size' is not specified.")
        if params['fit_generator']: # fit data generator
            if 'batches_per_epoch' not in params:
                print("parameter '--batches_per_epoch' is not specified "
                      "while training is set to fit generator")
    if params['CUDA_VISIBLE_DEVICES'] is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = RemoveQuotes(params['CUDA_VISIBLE_DEVICES'])
    
    params['data_path'] = RemoveQuotes(params['data_path'])
    params['model_path'] = RemoveQuotes(params['model_path'])
    print("""data path : {}""".format(params['data_path']))
    print("""model path : {}""".format(params['model_path']))
    return params



def train(model, train_X, train_Y, params):
    save_dir, file_name = os.path.split(params['model_path'])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    return model.fit(train_X, train_Y, stages = params['stages'],
                    epochs = params['epochs'], 
                    batch_size = params['batch_size'], 
                    end_time = params['end_time'], 
                    save_path = params['model_path'])

def train_with_generator(model, data_generator, valid_X, valid_Y, params):
    save_dir, file_name = os.path.split(params['model_path'])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    return model.fit_generator(data_generator, valid_X = valid_X, valid_Y = valid_Y,
                               stages = params['stages'], epochs = params['epochs'],
                               batches_per_epoch = params['batches_per_epoch'],
                               end_time = params['end_time'],
                               save_path = params['model_path'])

def test(model, test_X, test_length):
    print(test_X)
    return model.multistep_predict(test_X, test_length)

def draw_prediction(test_X, test_Y, prediction, test_length, n_figs, save_path):
    test_length_r = min(test_X.shape[0], test_length)
    n_figs = min(n_figs, test_X.shape[2])
    real = test_Y.T
    pred = prediction.T
    for i in range(n_figs):
        f = plt.figure(dpi = 200)
        path,file = os.path.split(save_path)
        suffix = '.dim{:0>2d}.jpg'.format(i)
        plt.title(file + suffix)
        raw = test_X[0,:,i]
        plt.plot(np.hstack([raw, real[i,:test_length_r]]), 'b', label = 'real')
        plt.plot(range(raw.shape[0], raw.shape[0]+pred.shape[1]),
                 pred[i], 'r', label = 'pred')
        plt.legend()
        plt.savefig(save_path + suffix)

def load_data(data_path):
    suffix = data_path[data_path.rfind('.')+1:]
    if suffix == 'csv':
        data = pd.read_csv(data_path, index_col = 'datetime').values
    elif suffix == 'npy':
        data = np.load(data_path)
    else:
        raise ValueError("Undefined data format : '{}'".format(suffix))
    return data

def prepare_data(data_path, input_length):
    data = load_data(data_path)
    if data.squeeze().ndim == 1: # if data is saved as an 1-d array, reshape it to 2-D
        data = data.reshape([-1,1])
    return data_util.SerieToPieces(data, piece_length = input_length)

def prepare_data_generator(data_path, params):
    data = load_data(data_path)
    data_feeder = data_util.SequentialRandomChannelDataFeeder(
                    data, params['batch_size'], params['batches_per_epoch'], 
                    params['input_length'], params['input_size'])
    valid_X, valid_Y, test_X, test_Y = data_feeder.extract_valid_test_data(0.05, 0.05)
    return data_feeder, valid_X, valid_Y, test_X, test_Y
    

if __name__ == '__main__':
    params = ParseLstmParams()

    if not params['fit_generator']:
        [train_X, train_Y, valid_X, valid_Y, test_X, test_Y] = \
                prepare_data(params['data_path'], params['input_length'])
    else:
        [data_feeder, valid_X, valid_Y, test_X, test_Y] = \
                prepare_data_generator(params['data_path'], params)

    if not params['test']: # train phase
        solver = eval(params['Solver'])(params)
        model = sequential.SequentialModel(solver = solver)
        if not params['fit_generator']:
            train(model, train_X, train_Y, params)
        else:
            train_with_generator(model, data_feeder, valid_X, valid_Y, params)
    else: # test phase
        model = sequential.SequentialModel.load_model(params['model_path'])
        model._set_predictor(dap.DeterministicAutoregressivePredictor())
        prediction = test(model, test_X[0:1], params['test_length'])
        # draw prediction results
        draw_prediction(test_X, test_Y, prediction, params['test_length'], 
                        params['n_figs'], params['model_path'])
        # save prediction results
        np.save(params['model_path'] + '.pred', prediction)



        