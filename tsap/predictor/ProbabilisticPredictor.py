# -*- coding: utf-8 -*-  
# Created Date: Saturday September 1st 2018
# Author: duxin
# Email: duxin_be@outlook.com
# Github: @Dosann
# -------------------------------

from .base import Predictor
#from ..utils.random_util import UniformRandomGenerator
from ..utils.data_util import discrete2continue
import numpy as np
import pickle

class MCMCPredictor(Predictor):

    def __init__(self, params, intervals):
        super(MCMCPredictor, self).__init__()
        self.n_classes = params['n_classes']
        self.n_samples = params['n_samples']
        self.intervals = intervals
        # self.randomg = UniformRandomGenerator(
        #     MIN=0, MAX=1, querycount=1000, cache_size=100000)
    
    def do_predict(self, solver, X):
        # X : (n_samples, input_length, input_size)
        # return : 
        #   prob : (n_samples, input_size, n_classes)
        #   pred : (n_samples, input_size)
        prob = solver.predict(X)
        if type(prob) is list:
            prob = [p[:,np.newaxis,...] for p in prob]
            prob = np.concatenate(prob, axis=1)
        else:
            prob = prob[:,np.newaxis,...]
        pred = discrete2continue(prob.argmax(axis=-1), self.intervals)
        return prob, pred
    
    def singlstep_predict(self, solver, X):
        # X : (length, input_length, input_size)
        # return : 
        #   prob_history : (length, input_size, n_classes)
        #   pred_history : (input_length+length, input_size)
        n_steps, input_length, input_size = X.shape
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
            prob_tmp, pred_tmp = self.do_predict(solver, X[i:i+1])
            prob_history[i] = prob_tmp[0,...]
            pred_history[input_length+i] = pred_tmp[0,...]
        return prob_history, pred_history
    
    def multistep_predict(self, solver, X, n_steps):
        # X : (1, input_length, input_size)
        # return : 
        #   prob_history : (n_samples, length, input_size, n_classes)
        #   pred_history : (n_samples, input_length+length, input_size)
        _ , input_length, input_size = X.shape
        # probabilities for all future steps
        prob_history = np.zeros((self.n_samples, n_steps, input_size, self.n_classes),
                                dtype=np.float32)
        # past history + future prediction
        pred_history = np.zeros((self.n_samples, input_length+n_steps, input_size),
                                dtype=np.float32)
        pred_history[:,:input_length,...] = np.repeat(X, self.n_samples, axis=0)
        n_periods = min(n_steps, 10)
        deci_progress = int(n_steps / n_periods)
        for i in range(n_steps):
            if (i+1) % deci_progress == 0:
                print("Current progress: {} / {}".format(i+1, n_steps))
            # prob : (n_samples, input_size, n_classes)
            prob, _ = self.do_predict(
                solver, pred_history[:,i:i+input_length,...])
            prob_history[:,i,...] = prob
            # sample from class prob
            pred = [None]*self.n_samples
            for s,samp in enumerate(prob):
                pred[s] = [np.random.multinomial(1,chan)
                    .argmax() for chan in samp]
            pred = np.array(pred) # (n_samples, input_size)
            pred_history[:,input_length+i,...] = discrete2continue(pred, self.intervals)
        return prob_history, pred_history
    
    def _save_others(self, path):
        pass
    
    def _load_others(self, path):
        pass
    
    def _pickle_self(self):
        return pickle.dumps(self)