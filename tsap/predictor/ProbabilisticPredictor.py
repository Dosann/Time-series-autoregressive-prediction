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
    
    def singlstep_predict(self, solver, X, verbose=0):
        # X : (length, input_length, input_size)
        # return : 
        #   prob_history : (length, input_size, n_classes)
        #   pred_history : (input_length+length, input_size)
        length, input_length, input_size = X.shape
        # probabilities for all future steps
        prob_history = np.zeros((length, input_size, self.n_classes),
                                dtype=np.float32)
        # past history + future prediction
        pred_history = np.zeros((input_length+length, input_size),
                                dtype=np.float32)
        pred_history[:input_length] = X[0]
        n_periods = min(length, 10)
        deci_progress = int(length / n_periods)
        for i in range(length):
            if verbose != 0:
                if (i+1) % deci_progress == 0:
                    print("Current progress: {} / {}".format(i+1, length))
            prob_tmp, pred_tmp = self.do_predict(solver, X[i:i+1])
            prob_history[i] = prob_tmp[0,...]
            pred_history[input_length+i] = pred_tmp[0,...]
        return prob_history, pred_history
    
    def multistep_predict(self, solver, X, length, verbose=0):
        # X : (1, input_length, input_size)
        # return : 
        #   prob_history : (n_samples, length, input_size, n_classes)
        #   pred_history : (n_samples, input_length+length, input_size)
        _ , input_length, input_size = X.shape
        # probabilities for all future steps
        prob_history = np.zeros((self.n_samples, length, input_size, self.n_classes),
                                dtype=np.float32)
        # past history + future prediction
        pred_history = np.zeros((self.n_samples, input_length+length, input_size),
                                dtype=np.float32)
        pred_history[:,:input_length,...] = np.repeat(X, self.n_samples, axis=0)
        n_periods = min(length, 10)
        deci_progress = int(length / n_periods)
        for i in range(length):
            if verbose != 0:
                if (i+1) % deci_progress == 0:
                    print("Current progress: {} / {}".format(i+1, length))
            # prob : (n_samples, input_size, n_classes)
            prob, _ = self.do_predict(
                solver, pred_history[:,i:i+input_length,...])
            prob_history[:,i,...] = prob
            # sample from class prob
            pred = [None]*self.n_samples
            for s,samp in enumerate(prob):
                pred[s] = [np.random.multinomial(1,chan/1.001) \
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

class MCMCPredictorStateful(Predictor):

    def __init__(self, params, intervals):
        super(MCMCPredictorStateful, self).__init__()
        self.n_classes = params['n_classes']
        self.n_samples = params['n_samples']
        self.intervals = intervals
        # self.randomg = UniformRandomGenerator(
        #     MIN=0, MAX=1, querycount=1000, cache_size=100000)
    
    def do_predict(self, solver, X):
        # input_length is by default 1, since the predictor is used for stateful model.
        # X : (n_samples, 1, input_size)
        # return : 
        #   prob : (n_samples, input_size, n_classes)
        #   pred : (n_samples, input_size)
        #print("before solver.predict(X). solver's input_shape: ", solver._solver.input_shape)
        #print("shape of X: ", X.shape)
        prob = solver.predict(X, batch_size=self.n_samples)
        #print("after solver.predict(X)")
        if type(prob) is list:
            prob = [p[:,np.newaxis,...] for p in prob]
            prob = np.concatenate(prob, axis=1)
        else:
            prob = prob[:,np.newaxis,...]
        pred = discrete2continue(prob.argmax(axis=-1), self.intervals)
        return prob, pred
    
    def singlstep_predict(self, solver, X, lead_length, verbose=0):
        # input_length is by default 1, since the predictor is used for stateful model.
        # X : (lead_length+length, 1, input_size)
        # return : 
        #   prob_history : (length, input_size, n_classes)
        #   pred_history : (lead_length+1+length, input_size)
        length, _, input_size = X.shape
        length += -lead_length
        # probabilities for all future steps
        prob_history = np.zeros((length, input_size, self.n_classes),
                                dtype=np.float32)
        # past history + future prediction
        pred_history = np.zeros((lead_length+1+length, input_size),
                                dtype=np.float32)
        pred_history[:lead_length+1] = X[:lead_length+1,0,:]
        n_periods = min(length, 6)
        milestones = np.linspace(0,lead_length+length,n_periods)
        m = 1
        for i in range(lead_length+length):
            if verbose != 0:
                if i+1 >= milestones[m]:
                    print("Current progress: {} / {}".format(i+1, lead_length+length))
                    m += 1
            prob, pred = self.do_predict(solver, X[i:i+1])
            if i >= lead_length:
                prob_history[i-lead_length] = prob[0,...]
                pred_history[i+1] = pred[0,...]
        return prob_history, pred_history
    
    def multistep_predict(self, solver, X, length, lead_length, verbose=0):
        # input_length is by default 1, since the predictor is used for stateful model.
        # X : (lead_length+1, 1, input_size)
        # return : 
        #   prob_history : (n_samples, length, input_size, n_classes)
        #   pred_history : (n_samples, lead_length+1+length, input_size)
        _, _, input_size = X.shape
        # probabilities for all future steps
        prob_history = np.zeros((self.n_samples, length, input_size, self.n_classes),
                                dtype=np.float32)
        # past history + future prediction
        pred_history = np.zeros((self.n_samples, lead_length+1+length, input_size),
                                dtype=np.float32)
        pred_history[:,:lead_length+1,...] = np.repeat(X[np.newaxis,:,0,:],
                                                self.n_samples, axis=0)
        n_periods = min(length, 6)
        milestones = np.linspace(0, lead_length+length, n_periods)
        m = 1
        for i in range(lead_length+length):
            if verbose != 0:
                if i+1 >= milestones[m]:
                    print("Current progress: {} / {}".format(i+1, lead_length+length))
                    m += 1
            # prob : (n_samples, input_size, n_classes)
            prob, _ = self.do_predict(
                solver, pred_history[:,i:i+1,:])
            if i >= lead_length:
                prob_history[:,i-lead_length,...] = prob
                # sample from class prob
                pred = [None]*self.n_samples
                for s,samp in enumerate(prob):
                    pred[s] = [np.random.multinomial(1,chan/1.001) \
                        .argmax() for chan in samp]
                pred = np.array(pred) # (n_samples, input_size)
                pred_history[:,i+1,:] = discrete2continue(pred, 
                                                self.intervals, random=True)
        return prob_history, pred_history
    
    def _save_others(self, path):
        pass
    
    def _load_others(self, path):
        pass
    
    def _pickle_self(self):
        return pickle.dumps(self)