# -*- coding: utf-8 -*-  
# Created Date: Saturday September 1st 2018
# Author: duxin
# Email: duxin_be@outlook.com
# Github: @Dosann
# -------------------------------

from .base import Predictor
import numpy as np
import pickle

class DeterministicAutoregressivePredictor(Predictor):

    def __init__(self):
        super(DeterministicAutoregressivePredictor, self).__init__()
    
    def do_predict(self, solver, X):
        predict = solver.predict(X)
        predict = predict.reshape((predict.size, ))
        return predict

    def multistep_predict(self, solver, X, n_steps):
        print("X.shape: {}".format(X.shape))
        n_samples, input_length, input_size = X.shape
        if input_length != solver.input_length:
            print("input length of X ({}) equals not the model input length "
                  "({}). The newest data has been used for prediction."
                  .format(input_length, solver.input_length))
            input_length = solver.input_length
            X = X[:,-input_length:,:]
        X_hist = np.zeros([n_samples, input_length+n_steps, input_size])
        X_hist[:,:input_length,:] = X
        for i in range(n_steps):
            print('progress: %d / %d'%(i+1, n_steps))
            X_hist[:,i+input_length,:] = self.do_predict(solver, X_hist[:,i:i+input_length,:])
        return X_hist[:,input_length:,:]
    
    def _save_others(self, path):
        pass
    
    def _load_others(self, path):
        pass
    
    def _pickle_self(self):
        return pickle.dumps(self)


