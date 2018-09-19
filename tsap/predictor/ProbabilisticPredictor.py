# -*- coding: utf-8 -*-  
# Created Date: Saturday September 1st 2018
# Author: duxin
# Email: duxin_be@outlook.com
# Github: @Dosann
# -------------------------------

from .base import Predictor
import pickle

class MCMCPredictor(Predictor):

    def __init__(self):
        pass
    
    def do_predict(self):
        pass
    
    def singlstep_predict(self, solver, X):
        pass
    
    def multistep_predict(self, solver, X, n_steps):
        pass
    
    def _save_others(self, path):
        pass
    
    def _load_others(self, path):
        pass
    
    def _pickle_self(self):
        return pickle.dumps(self)