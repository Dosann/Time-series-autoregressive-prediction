# -*- coding: utf-8 -*-  
# Created Date: Saturday September 1st 2018
# Author: duxin
# Email: duxin_be@outlook.com
# Github: @Dosann
# -------------------------------

from types import GeneratorType
import numpy as np
import datetime as dt
import pickle
import tensorflow as tf

from .base import Model

class SequentialModel(Model):

    def __init__(self, solver=None, predictor=None): # TODO
        super(SequentialModel, self).__init__()
        self.solver = solver
        self.predictor = predictor
        self.end_by_timestamp = False
        self.current_epoch = 0
        self.train_hist = {}

    def _set_solver(self, solver):
        self.solver = solver
    
    def _set_predictor(self, predictor):
        self.predictor = predictor
    
    def _check_parameters(self):
        pass
    
    def _check_solver(self):
        if self.solver is None:
            raise ValueError("Solver is not yet set up.")
        attrs = ['input_length', 'input_size']
        for attr in attrs:
            if not hasattr(self.solver, attr):
                raise ValueError("Solver lacks the attribution {}."
                                .format(attr))
    
    def _check_input(self, X, Y):
        self.solver._check_input(X, Y)
    
    def _check_input_generator(self, X, Y):
        self.solver._check_input_generator(X, Y)
    
    def _is_end(self):
        if self.end_by_timestamp:
            if dt.datetime.now() < self.end_time:
                return False
            else:
                return True
        if self.stages > 0:
            return False
        else:
            return True
    
    def _print_epoch(self):
        print("------ Current epoch: {}. {} ------"
            .format(self.current_epoch, dt.datetime.now().strftime('%Y%m%d %H:%M:%S')))

    def _append_history(self, new_hist):
        hist = new_hist.history
        for key, value in hist.items():
            if key in self.train_hist: # combine list
                self.train_hist[key] += value
            else:
                self.train_hist[key] = value

    def fit(self, X, Y, stages = 1, epochs=10, batch_size=64, end_time=None, save_path = None):
        if end_time is not None:
            self.end_by_timestamp = True
            try:
                self.end_time = dt.datetime.strptime(end_time, '%Y%m%d %H:%M:%S')
            except:
                raise ValueError("End time format is wrong. Please input end "
                                 "time in the format like '20180904 22:50:00'.")
        else:
            self.end_by_timestamp = False
            self.end_time = None
        self.stages = stages
        self.epochs = epochs
        self.batch_size = batch_size
        # Check
        self._check_solver()
        self._check_input(X, Y)
        self._check_parameters()
        # fit
        if save_path is not None:
            self.solver.save('{}.{:0>4d}'.format(save_path, self.current_epoch))
        while not self._is_end():
            self._print_epoch()
            hist = self.solver.fit(X, Y, epochs = self.epochs, batch_size = self.batch_size)
            self._append_history(hist)
            self.current_epoch += self.epochs
            self.stages += -1
            if save_path is not None:
                self.save('{}.{:0>4d}'.format(save_path, self.current_epoch))
        self._print_epoch()
    
    def fit_generator(self, X, Y, batches_per_epoch, epochs=10, end_time = None): 
        # UNFINISHED
        if end_time is None:
            self.end_by_timestamp = True
            try:
                self.end_time = dt.datetime.strptime(end_time, '%Y%m%d %H:%M:%S')
            except:
                raise ValueError("End time format is wrong. Please input end "
                                 "time in the format like '20180904 22:50:00'.")
        else:
            self.end_by_timestamp = False
            self.end_time = None
        self.epochs = epochs
        self.batches_per_epoch = batches_per_epoch
        # Check
        self._check_input_generator(X, Y)
        self._check_solver()
        self._check_parameters()
        # fit
        while not self._is_end():
            if save_path is not None:
                self.solver.save('../model/{}.{:0>4d}'.format(save_path, self.current_epoch))
            self._print_epoch()
            self.solver.fit_generator(X, Y, epochs = self.epochs, 
                                      batches_per_epoch = self.batches_per_epoch)
            self.current_epoch += self.epochs
        self._print_epoch()
    
    def _pickle_self(self):
        _solver = self.solver
        _predictor = self.predictor
        self.solver = None
        self.predictor = None
        model_p = pickle.dumps(self)
        self.solver = _solver
        self.predictor = _predictor
        return model_p
    
    def save(self, path):
        model_p = None
        solver_p = None
        predictor_p = None
        # pickle solver
        if self.solver is not None:
            self.solver._save_others(path)
            solver_p = self.solver._pickle_self()
        # pickle predictor
        if self.predictor is not None:
            self.predictor._save_others(path)
            predictor_p = self.predictor._pickle_self()
        # pickle self
        model_p = self._pickle_self()
        model = {'model':model_p,
                 'solver':solver_p,
                 'predictor':predictor_p}
        with open(path, 'wb') as f:
            f.write(pickle.dumps(model))
        print("# Model saved to {}".format(path))
    
    @staticmethod
    def load_model(path):
        with open(path, 'rb') as f:
            model = pickle.loads(f.read())
        solver = model['solver']
        predictor = model['predictor']
        model = model['model']
        model = pickle.loads(model)
        if solver is not None:
            solver = pickle.loads(solver)
            solver._load_others(path)
        if predictor is not None:
            predictor = pickle.loads(predictor)
            predictor._load_others(path)
        model._set_solver(solver)
        model._set_predictor(predictor)
        return model
    
    def load_weights(self, path): # TODO
        pass

    def predict(self, X):
        return self.predictor.do_predict(self.solver, X)
    
    def multistep_predict(self, X, n_steps):
        return self.predictor.multistep_predict(self.solver, X, n_steps)
    


