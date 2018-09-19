# -*- coding: utf-8 -*-  
# Created Date: Saturday September 1st 2018
# Author: duxin
# Email: duxin_be@outlook.com
# Github: @Dosann
# -------------------------------

from .base import Predictor
import numpy as np
import pickle
from tsap.utils.data_util import discrete2continue

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


class DetermDiscreteAGPredictor(Predictor):

    def __init__(self, params, intervals):
        super(DetermDiscreteAGPredictor, self).__init__()
        self.n_classes = params['n_classes']
        self.intervals = intervals
    
    def do_predict(self, solver, X):
        prob = solver.predict(X[np.newaxis,...]) # [(1, n_classes)] or (1, n_classes)
        if type(prob) is list:
            prob = np.concatenate(prob, axis=0) # (input_size, n_classes)
        pred = discrete2continue(prob.argmax(axis=-1), self.intervals)
        return prob, pred
    
    def singlstep_predict(self, solver, X):
        # X : (length, input_length, input_size)
        # return : 
        #   prob_history : (length, input_size, n_classes)
        #   pred_history : (input_length+length, input_size)
        # probabilities for all future steps
        n_steps, input_length, input_size = X.shape
        prob_history = np.zeros((n_steps, input_size, self.n_classes),
                                dtype=np.float32)
        # past history + future prediction
        pred_history = np.zeros((input_length+n_steps, input_size),
                                dtype=np.float32)
        pred_history[:input_length] = X[0]
        n_periods = min(n_steps, 10)
        deci_progress = int(n_steps / n_periods)
        for i in range(n_steps):
            if (i+1) % deci_progress == 0:
                print("Current progress: {} / {}".format(i+1, n_steps))
            prob_history[i], pred_history[input_length+i] = self.do_predict(
                solver, X[i])
        return prob_history, pred_history
    
    def multistep_predict(self, solver, X, n_steps):
        # X : (1, input_length, input_size)
        # return : 
        #   prob_history : (length, input_size, n_classes)
        #   pred_history : (input_length+length, input_size)
        _ , input_length, input_size = X.shape
        # probabilities for all future steps
        prob_history = np.zeros((n_steps, input_size, self.n_classes),
                                dtype=np.float32)
        # past history + future prediction
        pred_history = np.zeros((input_length+n_steps, input_size),
                                dtype=np.float32)
        pred_history[:input_length] = X[0]
        n_periods = min(n_steps, 10)
        deci_progress = int(n_steps / n_periods)
        for i in range(n_steps):
            if (i+1) % deci_progress == 0:
                print("Current progress: {} / {}".format(i+1, n_steps))
            prob_history[i], pred_history[input_length+i] = self.do_predict(
                solver, pred_history[i:i+input_length])
        return prob_history, pred_history
    
    def _save_others(self, path):
        pass
    
    def _load_others(self, path):
        pass
    
    def _pickle_self(self):
        return pickle.dumps(self)