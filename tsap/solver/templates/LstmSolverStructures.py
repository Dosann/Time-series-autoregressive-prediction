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

class Lstm1Layer(LstmSolverKeras):

    def __init__(self, params):
        super(Lstm1Layer, self).__init__(params)
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

class Lstm3Layer(LstmSolverKeras):

    def __init__(self, params):
        super(Lstm3Layer, self).__init__(params)
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

class DiscreteLstm1Layer(LstmSolverKeras):

    def __init__(self, params):
        super(DiscreteLstm1Layer, self).__init__(params)
        self._solver_construction(params)
    
    def _solver_construction(self, params):
        self.input_length = params['input_length']
        self.input_size = params['input_size']
        self.n_classes = params['n_classes']
        inputs = Input(shape=(self.input_length,
                              self.input_size))
        lstm1 = LSTM(params['hidden_units'], return_sequences=True)(inputs)
        flatten1 = Flatten()(lstm1)
        predicts = []
        for i in range(self.input_size):
            predicts.append(Dense(self.n_classes, activation='softmax',
                                  kernel_regularizer=None, 
                                  name='out_{}'.format(i))(flatten1))
        model = Model(inputs=inputs, outputs=predicts)
        losses = {'out_{}'.format(i):'categorical_crossentropy' 
                  for i in range(self.input_size)}
        model.compile(loss=losses,
                      optimizer='adam', metrics=['accuracy'])
        self._solver = model
    
    def _set_intervals(self, intervals):
        self.intervals = intervals

class DiscreteLstm3Layer(LstmSolverKeras):

    def __init__(self, params):
        super(DiscreteLstm3Layer, self).__init__(params)
        self._solver_construction(params)
    
    def _solver_construction(self, params):
        self.input_length = params['input_length']
        self.input_size = params['input_size']
        self.n_classes = params['n_classes']
        inputs = Input(shape=(self.input_length,
                              self.input_size))
        lstm1 = LSTM(params['hidden_units'], return_sequences=True)(inputs)
        lstm2 = LSTM(params['hidden_units'], return_sequences=True)(lstm1)
        lstm3 = LSTM(params['hidden_units'], return_sequences=True)(lstm2)
        
        flatten1 = Flatten()(lstm3)
        predicts = []
        for i in range(self.input_size):
            predicts.append(Dense(self.n_classes, activation='softmax',
                                  kernel_regularizer=None, 
                                  name='out_{}'.format(i))(flatten1))
        model = Model(inputs=inputs, outputs=predicts)
        losses = {'out_{}'.format(i):'categorical_crossentropy' 
                  for i in range(self.input_size)}
        model.compile(loss=losses,
                      optimizer='adam', metrics=['accuracy'])
        self._solver = model
    
    def _set_intervals(self, intervals):
        self.intervals = intervals


class DiscreteLstm1LayerStateful(LstmSolverKeras):

    def __init__(self, params):
        super(DiscreteLstm1LayerStateful, self).__init__(params)
        self._solver_construction(params)
    
    def _solver_construction(self, params):
        self.batch_size = params['batch_size']
        self.input_length = params['input_length']
        self.input_size = params['input_size']
        self.n_classes = params['n_classes']
        inputs = Input(batch_shape=(self.batch_size,
                                    self.input_length,
                                    self.input_size))
        lstm1 = LSTM(params['hidden_units'], return_sequences=False, stateful=True)(inputs)
        predicts = []
        for i in range(self.input_size):
            predicts.append(Dense(self.n_classes, activation='softmax',
                                  kernel_regularizer=None, 
                                  name='out_{}'.format(i))(lstm1))
        model = Model(inputs=inputs, outputs=predicts)
        losses = {'out_{}'.format(i):'categorical_crossentropy' 
                  for i in range(self.input_size)}
        model.compile(loss=losses,
                      optimizer='adam', metrics=['accuracy'])
        self._solver = model
    
    def _set_intervals(self, intervals):
        self.intervals = intervals

class DiscreteLstm3LayerStateful(LstmSolverKeras):

    def __init__(self, params):
        super(DiscreteLstm3LayerStateful, self).__init__(params)
        self._solver_construction(params)
    
    def _solver_construction(self, params):
        self.batch_size = params['batch_size']
        self.input_length = params['input_length']
        self.input_size = params['input_size']
        self.n_classes = params['n_classes']
        inputs = Input(batch_shape=(self.batch_size,
                                    self.input_length,
                                    self.input_size))
        lstm1 = LSTM(params['hidden_units'], return_sequences=True, stateful=True)(inputs)
        lstm2 = LSTM(params['hidden_units'], return_sequences=True, stateful=True)(lstm1)
        lstm3 = LSTM(params['hidden_units'], return_sequences=False, stateful=True)(lstm2)
        
        predicts = []
        for i in range(self.input_size):
            predicts.append(Dense(self.n_classes, activation='softmax',
                                  kernel_regularizer=None, 
                                  name='out_{}'.format(i))(lstm3))
        model = Model(inputs=inputs, outputs=predicts)
        losses = {'out_{}'.format(i):'categorical_crossentropy' 
                  for i in range(self.input_size)}
        model.compile(loss=losses,
                      optimizer='adam', metrics=['accuracy'])
        self._solver = model
    
    def _set_intervals(self, intervals):
        self.intervals = intervals