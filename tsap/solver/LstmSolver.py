# -*- coding: utf-8 -*-  
# Created Date: Saturday September 1st 2018
# Author: duxin
# Email: duxin_be@outlook.com
# Github: @Dosann
# -------------------------------

from .base import Solver
from ..utils import data_util
from types import GeneratorType
import numpy as np
import pandas as pd
import pickle
from keras import models
from sklearn.preprocessing import MinMaxScaler



class LstmSolverKeras(Solver):

    def __init__(self, params=None):
        super(LstmSolverKeras, self).__init__()
        self._init_params(params)
        self._solver = None
        self._scaler = None

    def _init_params(self, params):
        if params is None:
            params = dict()
        self.input_length = hasattr(params, 'input_length') and \
                                params['input_length'] or None
        self.input_size   = hasattr(params, 'input_size') and \
                                params['input_size'] or None
        self._params = params

    def _solver_construction(self):
        pass

    def _check_input_X(self, X):
        if len(X.shape) != 3:
            raise ValueError("Dimension of input X is {}, but should "
                             "be 3.".format(len(X.shape)))
        if self.input_length is None:
            self.input_length = X.shape[1]
        if self.input_size is None:
            self.input_size = X.shape[2]
        if self.input_length != X.shape[1]:
            raise ValueError("Input serie length mismatches between X "
                             "and solver.")
        if self.input_size != X.shape[2]:
            raise ValueError("input Serie size mismatches between X "
                             "and solver.")
    
    def _check_input_Y(self, Y):
        if len(Y.shape) != 2:
            raise ValueError("Dimension of input Y is {}, but should "
                             "be 2.".format(len(Y.shape)))

    def _check_input(self, X, Y):
        assert(isinstance(X, np.ndarray))
        assert(isinstance(Y, np.ndarray))
        self._check_input_X(X)
        self._check_input_Y(Y)
        if X.shape[0] != Y.shape[0]:
            raise ValueError("#X is not equal to #Y: #X({}), #Y({})."
                            .format(X.shape[0], Y.shape[0]))
    
    def _check_input_generator(self, X, Y):
        assert(isinstance(X, GeneratorType))
        assert(isinstance(Y, GeneratorType))
        X_sample = next(X)
        Y_sample = next(Y)
        self._check_input_X(X_sample)
        self._check_input_Y(Y_sample)
        if X_sample.shape[0] != Y_sample.shape[0]:
            raise ValueError("#X is not equal to #Y: #X({}), #Y({})."
                            .format(X_sample.shape[0], Y_sample.shape[0]))

    def fit(self, X, Y, epochs, batch_size):
        self._check_input(X, Y)
        return self._solver.fit(X, Y, epochs = epochs, 
                         batch_size = batch_size, 
                         validation_split = 0.01)

    def fit_generator(self, X, Y, batches_per_epoch, epochs):
        self._check_input_generator(X, Y)
        self._solver.fit_generator(X, Y, epochs = epochs,
                         steps_per_epoch = batches_per_epoch,
                         validation_steps = 5)
    
    def predict(self, X):
        assert(isinstance(X, np.ndarray))
        self._check_input_X(X)
        return self._solver.predict(X)

    def save(self, path):
        self._solver.save(path + '.keras')
        _solver = self._solver
        self._solver = None
        with open(path, 'wb') as f:
            f.write(pickle.dumps(self))
        self._solver = _solver
    
    @staticmethod
    def load_model(path, load_keras_model=True):
        with open(path, 'rb') as f:
            solver = pickle.loads(f.read())
        if load_keras_model:
            solver._solver = models.load_model(path + '.keras')
        return solver

    def get_data(self, data_path, input_length, data_scale):
        suffix = data_path[data_path.rfind('.')+1:]
        if suffix == 'csv':
            data = pd.read_csv(data_path, index_col = 'datetime')
        elif suffix == 'npy':
            data = np.load(data_path)
        else:
            raise ValueError("Undefined data format : '{}'".format(suffix))
        print(data.squeeze())
        if data_scale:
            if self._scaler is None:
                self._scaler = MinMaxScaler(feature_range = (-1, 1))
                self._scaler.fit(data)
            else:
                print("Use saved scaler")
            data = self._scaler.transform(data)
        return data_util.SerieToPieces(data, piece_length = input_length)