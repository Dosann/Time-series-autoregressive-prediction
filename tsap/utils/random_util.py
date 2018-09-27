# -*- coding: utf-8 -*-  
# Created Date: Saturday September 1st 2018
# Author: duxin
# Email: duxin_be@outlook.com
# Github: @Dosann
# -------------------------------

import numpy as np

class RandomGenerator:
    def __init__(self, **params):
        for key,value in params.items():
            setattr(self, key, value)
        self._reset_cache()
        self.pos = 0

    def _reset_cache(self):
        pass
    
    def set_querycount(self, querycount):
        self.querycount = querycount

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.pos + self.querycount >= self.cache_size:
            self._reset_cache()
            self.pos = 0
        self.pos += self.querycount
        return self.cache[self.pos-self.querycount : self.pos]

class UniformRandomGenerator(RandomGenerator):
    def __init__(self, MIN, MAX, querycount,
                 cache_size=100000):
        super(UniformRandomGenerator, self).__init__(
            **{
                'MIN':MIN,
                'MAX':MAX,
                'querycount':querycount,
                'cache_size':cache_size
            })
    
    def _reset_cache(self):
        self.cache = np.random.uniform(
            self.MAX, self.MIN, self.cache_size)