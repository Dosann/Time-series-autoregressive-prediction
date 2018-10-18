
import sys
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
parser.add_argument('--model_path', type=str)
parser.add_argument('--data_path', type=str)
parser.add_argument('--save_path', type=str)
parser.add_argument('--pred_length', type=int)
parser.add_argument('--lead_length', type=int)
params = vars(parser.parse_args())
os.environ['CUDA_VISIBLE_DEVICES'] = params['CUDA_VISIBLE_DEVICES']

import tsap
from tsap.solver.LstmSolver import LstmSolverKeras
from tsap.model import sequential
from tsap.predictor import DeterministicAutoregressivePredictor as dap
from tsap.predictor import ProbabilisticPredictor as pp
from tsap.solver.templates.LstmSolverStructures \
    import DiscreteLstm1LayerStateful, DiscreteLstm3LayerStateful
from tsap.utils import data_util
from tsap.utils.data_util import equalprob_interval_dividing, equalwidth_interval_dividing
from tsap.utils.data_util import discrete2continue, continue2discrete
import numpy as np
import pandas as pd
from scipy.stats import mode
import argparse
import matplotlib
import pickle as pkl
import copy

import platform
system = platform.system()
if system == "Linux":
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

model = sequential.SequentialModel.load_model(params['model_path'])
# "../../model/20181017.DiscreteRangeStatefulLSTM.EqualWidthInterval.truncatedata/"
# "FinData-CR-G4.1lay.10ts.128hu.0400"


params['batch_size']=1
params['batches_per_epoch']=1000
params['input_length']=1
params['input_size']=1
params['n_classes']=128
params['hidden_units']=128
solver2 = DiscreteLstm1LayerStateful(params)
solver2._solver.set_weights(model.solver._solver.get_weights())
model._set_solver(solver2)


predictor = dap.DetermDiscreteAGPredictorStateful(
    params=params, intervals=model.predictor.intervals
)

model.predictor = predictor
train_feeder = data_util.SeqDiscrtzdRandomChanlStatefulDF(
    load_data(params['data_path']), 
    params['batch_size'], params['batches_per_epoch'], 
    params['input_length'], params['input_size'],
    params['n_classes'], 
    intervals = None, 
    interv_dividing_method = equalwidth_interval_dividing)

X, Y = train_feeder.get_multistep_test_data(length=1, lead_length=params['lead_length'])
_, pred = model.multistep_predict(
        X=X, length=params['pred_length'], lead_length=params['lead_length'])

f = plt.figure()
plt.plot(np.array(pred))
f.savefig(params['save_path']+'.jpg')

np.save(params['save_path'], pred)