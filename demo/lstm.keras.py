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
import numpy as np
import pandas as pd
import argparse
import matplotlib

import platform
system = platform.system()
if system != "Windows":
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Activation, TimeDistributed, AveragePooling1D, Average, Reshape, Flatten
from keras import regularizers
from keras.utils import plot_model

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
    parser.add_argument('--Solver', type = str,
                        help = 'Solver object name')
    parser.add_argument('--data_scale', action = 'store_true', default = False)
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type = str, default = None)
    # train
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
    if params['test']:
        if 'test_length' not in params:
            print("parameter '--test_length' is not specified.")
    else:
        if 'epochs' not in params:
            print("parameter '--epochs' is not specified.")
        if 'batch_size' not in params:
            print("parameter '--batch_size' is not specified.")
    if params['CUDA_VISIBLE_DEVICES'] is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = RemoveQuotes(params['CUDA_VISIBLE_DEVICES'])
    
    params['data_path'] = RemoveQuotes(params['data_path'])
    params['model_path'] = RemoveQuotes(params['model_path'])
    print("""data path : {}""".format(params['data_path']))
    print("""model path : {}""".format(params['model_path']))
    return params


class SolverStructure1(LstmSolverKeras):

    def __init__(self, params):
        super(SolverStructure1, self).__init__(params)
        self._solver_construction(params)

    def _solver_construction(self, params):
        self.input_length = params['input_length']
        self.input_size   = params['input_size']
        inputs = Input(shape = (self.input_length, 
                                self.input_size))
        lstm1 = LSTM(100, return_sequences = True)(inputs)
        lstm2 = LSTM(100, return_sequences = True)(lstm1)
        lstm3 = LSTM(100, return_sequences = True)(lstm2)
        #reshape1 = Reshape((-1,))(lstm5)
        flatten1 = Flatten()(lstm3)
        predicts = Dense(self.input_size, activation = 'linear', 
                        kernel_regularizer = regularizers.l2(0.2))(flatten1)
        model = Model(inputs = inputs, outputs = predicts)
        model.compile(loss = 'mse', optimizer = 'adam')
        self._solver = model

class SolverStructure2(LstmSolverKeras):

    def __init__(self, params):
        super(SolverStructure2, self).__init__(params)
        self._solver_construction(params)

    def _solver_construction(self, params):
        self.input_length = params['input_length']
        self.input_size   = params['input_size']
        inputs = Input(shape = (self.input_length, 
                                self.input_size))
        lstm1 = LSTM(100, return_sequences = True)(inputs)
        #reshape1 = Reshape((-1,))(lstm5)
        flatten1 = Flatten()(lstm1)
        predicts = Dense(self.input_size, activation = 'linear', 
                        kernel_regularizer = regularizers.l2(0.2))(flatten1)
        model = Model(inputs = inputs, outputs = predicts)
        model.compile(loss = 'mse', optimizer = 'adam')
        self._solver = model

def train(model, train_X, train_Y, params):
    save_dir, file_name = os.path.split(params['model_path'])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    return model.fit(train_X, train_Y, stages = params['stages'],
                    epochs = params['epochs'], 
                    batch_size = params['batch_size'], 
                    end_time = params['end_time'], 
                    save_path = params['model_path'])

def test(model, test_X, test_length):
    print(test_X)
    return model.predictor.multistep_predict(test_X, test_length)

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


if __name__ == '__main__':
    params = ParseLstmParams()
    Solver = eval(params['Solver'])

    if not params['test']:
        solver = Solver(params)
        [train_X, train_Y, valid_X, valid_Y, test_X, test_Y] = \
            solver.get_data(params['data_path'], params['input_length'], params['data_scale'])
        model = sequential.SequentialModel(solver = solver)
        train(model, train_X, train_Y, params)
    else:
        solver = Solver.load_model(params['model_path'])
        [train_X, train_Y, valid_X, valid_Y, test_X, test_Y] = \
            solver.get_data(params['data_path'], params['input_length'], params['data_scale'])
        predictor = dap.DeterministicAutoregressivePredictor(solver)
        model = sequential.SequentialModel(solver = solver, predictor = predictor)
        prediction = test(model, test_X[0:1], params['test_length'])
        # draw prediction results
        draw_prediction(test_X, test_Y, prediction, params['test_length'], 
                        params['n_figs'], params['model_path'])
        # save prediction results
        np.save(params['model_path'] + '.pred', prediction)