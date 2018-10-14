
import sys
import os
import tsap
from tsap.solver.LstmSolver import LstmSolverKeras
from tsap.model import sequential
from tsap.predictor import DeterministicAutoregressivePredictor as dap
from tsap.predictor import ProbabilisticPredictor as pp
from tsap.solver.templates.LstmSolverStructures \
    import DiscreteLstm1Layer, DiscreteLstm3Layer
from tsap.utils import data_util
from tsap.utils.data_util import equalprob_interval_dividing, equalwidth_interval_dividing
from tsap.utils.data_util import discrete2continue, continue2discrete
import numpy as np
import pandas as pd
from scipy.stats import mode
import argparse
import matplotlib
import pickle as pkl

import platform
system = platform.system()
if system != "Windows":
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_data(data_path):
    suffix = data_path[data_path.rfind('.')+1:]
    if suffix == 'csv':
        data = pd.read_csv(data_path, index_col = 'datetime').values
    elif suffix == 'npy':
        data = np.load(data_path)
    else:
        raise ValueError("Undefined data format : '{}'".format(suffix))
    return data

model = sequential.SequentialModel.load_model(
    "../../model/20180930.DiscreteRangeLSTM.EqualProbInterval.truncatedata/"
    "FinData-CR-G3.1lay.50ts.32hu.0300")

params = {
    "batch_size":64,
    "batches_per_epoch":1000,
    "input_length":50,
    "input_size":1,
    "n_classes":128,
    "train_path":"../../data/NYSEtop80.1h.preprcd/train.truncate.npy"
}

predictor = dap.DetermDiscreteAGPredictor(
    params=params, intervals=model.predictor.intervals
)

model.predictor = predictor
train_feeder = data_util.SequentialDiscreteRCDF(
    load_data(params['train_path']), 
    params['batch_size'], params['batches_per_epoch'], 
    params['input_length'], params['input_size'],
    params['n_classes'], 
    intervals = None, 
    interv_dividing_method = equalwidth_interval_dividing)

X, Y = train_feeder.get_multistep_test_data(length=1)
PRED_LENGTH = 100
PRED_EPOCHS = 10
preds = []
for epoch in range(PRED_EPOCHS):
    _, pred = model.multistep_predict(
        X=X, n_steps=PRED_LENGTH)
    preds += list(pred.squeeze())[-PRED_LENGTH:]
    print("Current progress: {}/{}".format(epoch+1,PRED_EPOCHS))
    X = pred[np.newaxis,:-params["input_length"],:]

np.save("20180930.DiscreteRangeLSTM.EqualProbInterval."
    "truncatedata_{}".format(PRED_LENGTH*PRED_EPOCHS), np.array(preds))