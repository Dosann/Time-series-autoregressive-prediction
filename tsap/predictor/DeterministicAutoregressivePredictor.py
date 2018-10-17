# -*- coding: utf-8 -*-  
# Created Date: Saturday September 1st 2018
# Author: duxin
# Email: duxin_be@outlook.com
# Github: @Dosann
# -------------------------------

from .base import Predictor
from ..utils.data_util import discrete2continue
import numpy as np
import pickle

class DeterministicAutoregressivePredictor(Predictor):

    def __init__(self):
        super(DeterministicAutoregressivePredictor, self).__init__()
    
    def do_predict(self, solver, X):
        # X: (batch_size, input_length, input_size)
        # return:
        #   predict: (batch_size, input_size)
        predict = solver.predict(X)
        if predict.ndim == 1:
            predict = predict[:, np.newaxis]
        return predict
    
    def singlstep_predict(self, solver, X):
        # X: (length, input_length, input_size):
        # return:
        #   pred_history: (input_length+length, input_size)
        pred = self.do_predict(solver, X)
        pred_history = np.vstack([
            X[0], pred])
        return pred_history

    def multistep_predict(self, solver, X, length, verbose=0):
        # X: (1, input_length, input_size)
        # return : 
        #   pred_history: (input_length+length, input_size)
        _, input_length, input_size = X.shape
        if input_length != solver.input_length:
            print("input length of X ({}) equals not the model input length "
                  "({}). The newest data has been used for prediction."
                  .format(input_length, solver.input_length))
            input_length = solver.input_length
            X = X[:,-input_length:,:]
        pred_history = np.zeros([input_length+length, input_size])
        pred_history[:input_length,:] = X
        for i in range(length):
            if verbose != 0:
                print('progress: %d / %d'%(i+1, length))
            pred_history[i+input_length,:] = self.do_predict(solver, pred_history[np.newaxis, i:i+input_length,:])[0]
        return pred_history
    
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
        # X : (input_length, input_size)
        # return :
        #   prob : (input_size, n_classes)
        #   pred : (input_size)
        prob = solver.predict(X[np.newaxis,...]) # [(1, n_classes)] or (1, n_classes)
        if type(prob) is list:
            prob = np.concatenate(prob, axis=0) # (input_size, n_classes)
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
            prob_history[i], pred_history[input_length+i] = self.do_predict(
                solver, X[i])
        return prob_history, pred_history
    
    def multistep_predict(self, solver, X, length, verbose=0):
        # X : (1, input_length, input_size)
        # return : 
        #   prob_history : (length, input_size, n_classes)
        #   pred_history : (input_length+length, input_size)
        _ , input_length, input_size = X.shape
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
            prob_history[i], pred_history[input_length+i] = self.do_predict(
                solver, pred_history[i:i+input_length])
        return prob_history, pred_history
    
    def _save_others(self, path):
        pass
    
    def _load_others(self, path):
        pass
    
    def _pickle_self(self):
        return pickle.dumps(self)

class DetermDiscreteAGPredictorStateful(Predictor):
    # deterministic discrete auto-regressive predictor for stateful model.

    def __init__(self, params, intervals):
        super(DetermDiscreteAGPredictorStateful, self).__init__()
        self.n_classes = params['n_classes']
        self.intervals = intervals
    
    def do_predict(self, solver, X):
        # input_length is by default 1, since the predictor is used for stateful model.
        # X : (1, input_size)
        # return :
        #   prob : (input_size, n_classes)
        #   pred : (input_size)
        print("determ.before solver.predict(X). solver's input_shape: ", solver._solver.input_shape)
        print("shape of X: ", X.shape)
        prob = solver.predict(X[np.newaxis,...]) # [(1, n_classes)] or (1, n_classes)
        if type(prob) is list:
            prob = np.concatenate(prob, axis=0) # (input_size, n_classes)
        pred = discrete2continue(prob.argmax(axis=-1), self.intervals, random=True)
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
            prob, pred = self.do_predict(solver, X[i])
            if i >= lead_length:
                prob_history[i-lead_length] = prob
                pred_history[i+1] = pred
        return prob_history, pred_history
    
    def multistep_predict(self, solver, X, length, lead_length, verbose=0):
        # input_length is by default 1, since the predictor is used for stateful model.
        # X : (lead_length+1, 1, input_size)
        # return : 
        #   prob_history : (length, input_size, n_classes)
        #   pred_history : (lead_length+1+length, input_size)
        _, _, input_size = X.shape
        # probabilities for all future steps
        prob_history = np.zeros((length, input_size, self.n_classes),
                                dtype=np.float32)
        # past history + future prediction
        pred_history = np.zeros((lead_length+1+length, input_size),
                                dtype=np.float32)
        pred_history[:lead_length+1] = X[:,0,:]
        n_periods = min(length, 6)
        milestones = np.linspace(0, lead_length+length, n_periods)
        m = 1
        for i in range(lead_length+length):
            if verbose != 0:
                if i+1 >= milestones[m]:
                    print("Current progress: {} / {}".format(i+1, lead_length+length))
                    m += 1
            prob, pred = self.do_predict(solver, pred_history[i,np.newaxis,:])
            if i >= lead_length:
                prob_history[i-lead_length] = prob
                pred_history[i+1] = pred
        return prob_history, pred_history
    
    def _save_others(self, path):
        pass
    
    def _load_others(self, path):
        pass
    
    def _pickle_self(self):
        return pickle.dumps(self)