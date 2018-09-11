# -*- coding: utf-8 -*-  
# Created Date: Saturday September 1st 2018
# Author: duxin
# Email: duxin_be@outlook.com
# Github: @Dosann
# -------------------------------

from ..LstmSolver import LstmSolverKeras
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Activation, TimeDistributed, AveragePooling1D, Average, Reshape, Flatten
from keras import regularizers
from keras.utils import plot_model

class SolverStructure1(LstmSolverKeras):

    def __init__(self, params):
        super(SolverStructure1, self).__init__(params)
        self._solver_construction(params)

    def _solver_construction(self, params):
        self.input_length = params['input_length']
        self.input_size   = params['input_size']
        inputs = Input(shape = (self.input_length, 
                                self.input_size))
        lstm1 = LSTM(params['hidden_units'], return_sequences = True)(inputs)
        lstm2 = LSTM(params['hidden_units'], return_sequences = True)(lstm1)
        lstm3 = LSTM(params['hidden_units'], return_sequences = True)(lstm2)
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
        lstm1 = LSTM(params['hidden_units'], return_sequences = True)(inputs)
        #reshape1 = Reshape((-1,))(lstm5)
        flatten1 = Flatten()(lstm1)
        predicts = Dense(self.input_size, activation = 'linear', 
                        kernel_regularizer = regularizers.l2(0.2))(flatten1)
        model = Model(inputs = inputs, outputs = predicts)
        model.compile(loss = 'mse', optimizer = 'adam')
        self._solver = model

class SolverStructure3(LstmSolverKeras):
    
    def __init__(self, params):
        super(SolverStructure3, self).__init__(params)
        self._solver_construction(params)

    def _solver_construction(self, params):
        self.input_length = params['input_length']
        self.input_size   = params['input_size']
        inputs = Input(shape = (self.input_length, 
                                self.input_size))
        lstm1 = LSTM(params['hidden_units'], return_sequences = True)(inputs)
        lstm2 = LSTM(params['hidden_units'], return_sequences = True)(lstm1)
        lstm3 = LSTM(params['hidden_units'], return_sequences = True)(lstm2)
        lstm4 = LSTM(params['hidden_units'], return_sequences = True)(lstm3)
        lstm5 = LSTM(params['hidden_units'], return_sequences = True)(lstm4)
        #reshape1 = Reshape((-1,))(lstm5)
        flatten1 = Flatten()(lstm5)
        predicts = Dense(self.input_size, activation = 'linear', 
                        kernel_regularizer = regularizers.l2(0.2))(flatten1)
        model = Model(inputs = inputs, outputs = predicts)
        model.compile(loss = 'mse', optimizer = 'adam')
        self._solver = model