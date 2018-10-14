import os
import sys
import numpy as np
import pickle as pkl
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tsap.utils.data_util import SequentialRandomChannelDataFeeder

def load_data(path):
    df = SequentialRandomChannelDataFeeder(
        np.load(path),
        50000, 1000, input_length, 1)
    X, Y = next(df)
    X = X.squeeze()
    Y = Y.squeeze()
    return df, X, Y

def train(lr, X, Y):
    lr.fit(X, Y)
    return lr

def singlstep_predict(X):
    # X: (length, input_length)
    # return:
    #   pred: (input_length+length)
    pred = lr.predict(X)
    pred = np.hstack([X[0],pred])
    return pred

def multistep_predict(X, length):
    # X: (1, input_length)
    # return:
    #   pred: (input_length+length)
    _, input_length = X.shape
    pred = np.zeros(length + input_length)
    pred[:input_length] = X[0]
    for i in range(length):
        pred[input_length+i] = lr.predict(pred[np.newaxis, i:i+input_length])[0]
    return pred

def rmse(true, pred):
    RMSE = np.square(true - pred).mean()**0.5
    return RMSE

input_length = 10
length = 100
TRAIN_PATH = '../../data/NYSEtop80.1h.preprcd/train.npy'
VALID1_PATH = '../../data/NYSEtop80.1h.preprcd/valid.1.npy'
VALID2_PATH = '../../data/NYSEtop80.1h.preprcd/valid.2.npy'

lr = LinearRegression()
train_df, X, Y = load_data(TRAIN_PATH)
valid1_df, _, _ = load_data(VALID1_PATH)
valid2_df, _, _ = load_data(VALID2_PATH)
lr = train(lr, X, Y)

X, Y = train_df.get_singlstep_test_data(length)
pred_singlstep = singlstep_predict(X.reshape([X.shape[0],-1]))
real_singlstep = np.hstack([np.array([None]*input_length),
                            Y.squeeze()])
print("# rmse for singlstep prediction on dataset 'train': {}"
    .format(rmse(real_singlstep[input_length:], pred_singlstep[input_length:])))
X, Y = train_df.get_multistep_test_data(length)
pred_multistep = multistep_predict(X.reshape([1,-1]), length)
real_multistep = np.hstack([np.array([None]*input_length),
                            Y.squeeze()])
print("# rmse for multistep prediction on dataset 'train': {}"
    .format(rmse(real_singlstep[input_length:], pred_singlstep[input_length:])))

X, Y = valid1_df.get_singlstep_test_data(length)
pred_singlstep_valid = singlstep_predict(X.reshape([X.shape[0],-1]))
real_singlstep_valid = np.hstack([np.array([None]*input_length),
                            Y.squeeze()])
print("# rmse for singlstep prediction on dataset 'valid.1': {}"
    .format(rmse(real_singlstep_valid[input_length:], pred_singlstep_valid[input_length:])))
X, Y = valid1_df.get_multistep_test_data(length)
pred_multistep_valid = multistep_predict(X.reshape([X.shape[0],-1]), length)
real_multistep_valid = np.hstack([np.array([None]*input_length),
                            Y.squeeze()])
print("# rmse for multistep prediction on dataset 'valid.1': {}"
    .format(rmse(real_multistep_valid[input_length:], pred_multistep_valid[input_length:])))

X, Y = valid2_df.get_singlstep_test_data(length)
pred_singlstep_valid = singlstep_predict(X.reshape([X.shape[0],-1]))
real_singlstep_valid = np.hstack([np.array([None]*input_length),
                            Y.squeeze()])
print("# rmse for singlstep prediction on dataset 'valid.2': {}"
    .format(rmse(real_singlstep_valid[input_length:], pred_singlstep_valid[input_length:])))
X, Y = valid2_df.get_multistep_test_data(length)
pred_multistep_valid = multistep_predict(X.reshape([X.shape[0],-1]), length)
real_multistep_valid = np.hstack([np.array([None]*input_length),
                            Y.squeeze()])
print("# rmse for multistep prediction on dataset 'v': {}"
    .format(rmse(real_multistep_valid[input_length:], pred_multistep_valid[input_length:])))

f1 = plt.figure(dpi=200)
plt.plot(pred_singlstep, label='singlstep.pred')
plt.plot(real_singlstep, label='singlstep.real')
plt.legend()
f2 = plt.figure(dpi=200)
plt.plot(pred_multistep, label='multistep.pred')
plt.plot(real_multistep, label='multistep.real')
plt.legend()
plt.show()