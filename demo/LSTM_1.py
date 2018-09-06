# -*- coding: utf-8 -*-  
# Created Date: Saturday September 1st 2018
# Author: duxin
# Email: duxin_be@outlook.com
# Github: @Dosann
# -------------------------------

import sys
sys.path.append('../src/')
from solver.LstmSolver import LstmSolver
from model import sequential
from utils.ParamParser import ParseLstmParams
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils import data_util

import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Activation, TimeDistributed, AveragePooling1D, Average, Reshape, Flatten
from keras import regularizers
from keras.utils import plot_model

class LstmSolver_1(LstmSolver):

    def __init__(self, params):
        super(LstmSolver_1, self).__init__(params)
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

def get_data(input_length):
    prices = pd.read_csv('../data/prices.5min.top100volume/top80volumestocks.csv', index_col = 'datetime')
    scaler = MinMaxScaler()
    prices = scaler.fit_transform(prices)
    return data_util.SerieToPieces(prices, piece_length = input_length)

if __name__ == '__main__':
    params = ParseLstmParams()
    solver = LstmSolver_1(params)
    print(solver)
    model = sequential.SequentialModel(solver = solver)
    [train_X, train_Y, valid_X, valid_Y, test_X, test_Y] = get_data(params['input_length'])
    model.fit(train_X, train_Y, epochs = 10, batch_size = 64, end_time = '20180907 10:00:00', save_path = '../model/LSTM_1/lstm1.solver')