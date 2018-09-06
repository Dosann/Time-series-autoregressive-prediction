# -*- coding: utf-8 -*-  
# Created Date: Saturday September 1st 2018
# Author: duxin
# Email: duxin_be@outlook.com
# Github: @Dosann
# -------------------------------

class Predictor:

    def __init__(self, solver):
        self._set_solver(solver)
    
    def _set_solver(self, solver):
        self._solver = solver
    
    def multistep_predict(self, X, n_steps):
        pass