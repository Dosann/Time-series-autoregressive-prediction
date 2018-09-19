# -*- coding: utf-8 -*-  
# Created Date: Saturday September 1st 2018
# Author: duxin
# Email: duxin_be@outlook.com
# Github: @Dosann
# -------------------------------

import numpy as np
import pandas as pd

def Seq2SuccessiveXYBatch(seq, batch_size, length):
    if batch_size == 0:
        return [], []
    X = [None] * batch_size
    Y = [None] * batch_size
    for i in range(batch_size):
        X[i] = seq[np.newaxis, i:i+length]
        Y[i] = seq[np.newaxis, i+length]
    X = np.concatenate(X, axis = 0)
    Y = np.concatenate(Y, axis = 0)
    return X, Y


class SequentialRandomChannelDataFeeder:
    # input : data of 1-dim (n_samples,) or
    #         2-dim shape (n_samples, n_channels)
    # output : X and Y. Channels of them are randomly choosen.
    #         X is of shape (batch_size, timesteps, n_channels),
    #         Y is of shape (batch_size, n_channels). 

    def __init__(self, data, batch_size, batches_per_epoch, 
                 out_length, out_size):
        self.data = data
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.out_length = out_length
        self.out_size = out_size
        self._check_data()
        self._reset()

    def extract_valid_test_data(self, valid_split, test_split,
                               channels = None):
        # default: select the first 'out_size' channels as valid/test dataset.
        if channels is None:
            channels = range(self.out_size)
        # split
        n_valid = int((self._to - self._from) * valid_split)
        n_test  = int((self._to - self._from) * test_split)
        _from_test   = self._to - n_test
        _from_valid  = _from_test - n_valid
        valid_X, valid_Y = Seq2SuccessiveXYBatch(self.data[_from_valid:, channels], 
                                                 n_valid, self.out_length)
        test_X, test_Y   = Seq2SuccessiveXYBatch(self.data[_from_test:, channels], 
                                                 n_test, self.out_length)
        self._to = _from_valid
        return valid_X, valid_Y, test_X, test_Y

    def _check_data(self):
        if not isinstance(self.data, np.ndarray):
            raise ValueError("Input data should be of 'np.ndarray' "
                             "but is '{}'".format(type(self.data)))
        if self.data.ndim == 1:
            self.data = self.data.reshape([-1,1])
        elif self.data.ndim != 2:
            raise ValueError("Input data shape should be 1 or 2, "
                             "but is {}".format(self.data.shape))

        self.n_timestep, self.n_chan = self.data.shape
        if self.n_chan < self.out_size:
            raise ValueError("Data size (# of channel) is smaller "
                             "than predefined 'out_size'")
        self._from = 0
        self._to   = self.n_timestep - self.out_length - 1
        
    def _reset(self):
        self.batch = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.batch >= self.batches_per_epoch:
            self._reset()
            raise StopIteration
        X = [None] * self.batch_size
        Y = [None] * self.batch_size
        _from = np.random.randint(self._from, self._to, self.batch_size)
        _to   = _from + self.out_length
        for i in range(self.batch_size):
            channel_ids = np.random.permutation(self.n_chan)[:self.out_size]
            X[i] = self.data[np.newaxis, _from[i]:_to[i], channel_ids]
            Y[i] = self.data[np.newaxis, _to[i], channel_ids]
        X = np.concatenate(X, axis = 0)
        Y = np.concatenate(Y, axis = 0)
        self.batch += 1
        return X, Y


if __name__ == '__main__':
    data = pd.read_csv('../../data/prices.5min.top100volume/top80volumestocks.csv', 
                        index_col = 'datetime').values
    datafeeder = SequentialRandomChannelDataFeeder(data, 64, 1000, 100, 6)
    valid_X, valid_Y, test_X, test_Y = datafeeder.extract_valid_test_data(0.1, 0.1)
    print(valid_X, valid_Y)