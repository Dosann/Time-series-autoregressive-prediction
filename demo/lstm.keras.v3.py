# -*- coding: utf-8 -*-  
# Created Date: Saturday September 1st 2018
# Author: duxin
# Email: duxin_be@outlook.com
# Github: @Dosann
# -------------------------------

import os
import sys
import tsap
from tsap.solver.LstmSolver import LstmSolverKeras
from tsap.model import sequential
from tsap.predictor import DeterministicAutoregressivePredictor as dap
from tsap.predictor import ProbabilisticPredictor as pp
from tsap.solver.templates.LstmSolverStructures \
    import DiscreteLstm1Layer
from tsap.utils import data_util
import numpy as np
import pandas as pd
import argparse
import matplotlib

import platform
system = platform.system()
if system != "Windows":
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


def RemoveQuotes(string): 
    # Quotes will be appended on head/tail of input string arguments.
    # This function removes them
    if string[0] in ['"', "'"]:
        string = string[1:]
    if string[-1] in ['"', "'"]:
        string = string[:-1]
    return string

def ParseDiscreteLstmParams():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_length', type = int,
                        help = 'serie length of input serie')
    parser.add_argument('input_size', type = int,
                        help = 'dimension of input vector')
    parser.add_argument('n_classes', type=int, 
                        help = 'num of intervals the all value '
                               'the whole range is divided into')
    parser.add_argument('--train_path', type = str,
                        help = 'path of train data (raw sequential numpy array)')
    parser.add_argument('--valid_path', type = str,
                        help = 'path of valid data (raw sequential numpy array)')
    parser.add_argument('--test_path', type = str,
                        help = 'path of test data (raw sequential numpy array)')
    parser.add_argument('--model_path', type = str,
                        help = 'path of model\n'
                               '\tformat like: "../model/lstm1.solver')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type = str, default = None)
    # nn construction
    parser.add_argument('--Solver', type = str,
                        help = 'Solver object name')
    parser.add_argument('--hidden_units', type = int,
                        help = '# of hidden units in LSTM')
    # train
    parser.add_argument('--fit_generator', action = 'store_true', default = False,
                        help = '(train phase) use generator to feed data')
    parser.add_argument('--end_time', type = str, default = None,
                        help = '(train phase, optional) time to end training.\n'
                               '\tformat like: "20180910 00:00:00')
    parser.add_argument('--stages', type = int, default = 1,
                        help = '(train phase) # of stages when training. '
                               'A stage includes one of several epochs')
    parser.add_argument('--epochs', type = int,
                        help = '(train phase) # of epochs when training')
    parser.add_argument('--batch_size', type = int,
                        help = '(train phase) batch size when training')
    parser.add_argument('--batches_per_epoch', type = int, 
                        help = '(train phase) # of batches per epoch when training')
    # test
    parser.add_argument('--test', action = 'store_true', default = False,
                        help = '(test phase) start test phase')
    parser.add_argument('--test_length', type = int,
                        help = '(test phase) # of timesteps to forecast when test')
    parser.add_argument('--montecarlo', action='store_true', default=False,
                        help = 'Use monte carlo multi-step prediction '
                               'if this argument included.')
    # parser.add_argument('--n_figs', type = int, default = 1,
    #                     help = '(test phase) # of dimensions to draw when test')
    params = vars(parser.parse_args())

    if 'model_path' not in params:
        print("parameter '--model_path' is not specified.")
    if 'Solver' not in params:
        print("parameter '--Solver' is not specified.")
    if 'hidden_units' not in params:
        print("parameter '--hidden_units' is not specified.")
    if params['test']: # test phase
        if 'test_path' not in params:
            raise ValueError("parameter '--test_path' is not specified.")
        params['test_path'] = RemoveQuotes(params['test_path'])
        print("""test data path : {}""".format(params['test_path']))
        if 'test_length' not in params:
            raise ValueError("parameter '--test_length' is not specified.")
    else: # train phase
        if 'train_path' not in params:
            raise ValueError("parameter '--train_path' is not specified.")
        params['train_path'] = RemoveQuotes(params['train_path'])
        print("""train data path : {}""".format(params['train_path']))
        if 'valid_path' not in params:
            raise ValueError("parameter '--valid_path' is not specified.")
        params['valid_path'] = RemoveQuotes(params['valid_path'])
        print("""valid data path : {}""".format(params['valid_path']))
        if 'stages' not in params:
            raise ValueError("parameter '--stages' is not specified.")
        if 'epochs' not in params:
            raise ValueError("parameter '--epochs' is not specified.")
        if 'batch_size' not in params:
            raise ValueError("parameter '--batch_size' is not specified.")
        if params['fit_generator']: # fit data generator
            if 'batches_per_epoch' not in params:
                print("parameter '--batches_per_epoch' is not specified "
                      "while training is set to fit generator")
    if params['CUDA_VISIBLE_DEVICES'] is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = RemoveQuotes(params['CUDA_VISIBLE_DEVICES'])
    
    params['model_path'] = RemoveQuotes(params['model_path'])
    print("""model path : {}""".format(params['model_path']))
    return params

# def train(model, train_X, train_Y, params):
#     save_dir, file_name = os.path.split(params['model_path'])
#     if not os.path.exists(save_dir):
#         os.mkdir(save_dir)
#     return model.fit(train_X, train_Y, stages = params['stages'],
#                     epochs = params['epochs'], 
#                     batch_size = params['batch_size'], 
#                     end_time = params['end_time'], 
#                     save_path = params['model_path'])

def train_with_generator(model, train_generator, valid_generator, params):
    save_dir, file_name = os.path.split(params['model_path'])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    return model.fit_generator(train_generator, validation_data=valid_generator,
                               stages = params['stages'], epochs = params['epochs'],
                               batches_per_epoch = params['batches_per_epoch'],
                               end_time = params['end_time'],
                               save_path = params['model_path'],
                               callback=to_graphic)

def test(model, test_X, test_length):
    print(test_X)
    return model.multistep_predict(test_X, test_length)

def draw(true, pred, prob=None):
    # true : (length, input_size)
    # pred : (input_length+length, input_size)
    # prob : (length, input_size, n_classes)
    # if prob is not None, 'true' / 'pred' should be classes,
    # else 'true' / 'pred' should be continuous real values.
    length, input_size = true.shape
    input_length, _    = pred.shape
    input_length += -length
    # patch for true
    patch_true = np.array([[None]*input_size for i in range(input_length)])
    true = np.vstack([patch_true, true])
    # patch for prob
    if prob is not None:
        _, _, n_classes = prob.shape
        patch_prob = np.zeros([input_length, input_size, n_classes])
        prob = np.concatenate([patch_prob, prob], axis=0)
    # draw figure
    f = plt.figure(figsize=(15,15))
    layout = (input_size, 1)
    for dim in range(input_size):
        ax = plt.subplot2grid(layout, (dim,0))
        if prob is not None:
            ax.imshow(prob[:,dim,:].T, aspect='auto', origin='lower')
        ax.plot(true[:,dim], color='b', label='true')
        ax.plot(pred[:,dim], color='r', label='pred')
        f.add_axes(ax)
    return f

def to_graphic(model):
    # singlstep test
    X, Y = valid_feeder.get_singlstep_test_data(params['test_length'])
    true_class = Y.argmax(axis=-1)
    true_value = data_util.discrete2continue(true_class, intervals)
    prob, pred_value = model.singlstep_predict(X)
    pred_class = data_util.continue2discrete(pred_value, intervals)
    print("Predicted classes of single-step test", pred_class.reshape(-1))
    # draw figure
    f1 = draw(true_value, pred_value)
    f2 = draw(true_class, pred_class, prob)
    f1.savefig(params['model_path'] + 
        '.singlstep_pred.epoch{:0>4d}.jpg'.format(model.current_epoch))
    f2.savefig(params['model_path'] + 
        '.singlstep_pred.epoch{:0>4d}.hm.jpg'.format(model.current_epoch))

    # multistep test
    X, Y = valid_feeder.get_multistep_test_data(params['test_length'])
    true_class = Y.argmax(axis=-1)
    true_value = data_util.discrete2continue(true_class, intervals)
    prob, pred_value = model.multistep_predict(X, params['test_length'])
    pred_class = data_util.continue2discrete(pred_value, intervals)
    print("Predicted classes of multi-step test", pred_class.reshape(-1))
    # draw figure
    f1 = draw(true_value, pred_value)
    f2 = draw(true_class, pred_class, prob)
    f1.savefig(params['model_path'] + 
        '.multistep_pred.epoch{:0>4d}.jpg'.format(model.current_epoch))
    f2.savefig(params['model_path'] + 
        '.multistep_pred.epoch{:0>4d}.hm.jpg'.format(model.current_epoch))


# def draw_prediction(test_X, test_Y, prob, pred, test_length, n_figs, save_path):
#     test_length_r = min(test_X.shape[0], test_length)
#     n_figs = min(n_figs, test_X.shape[2])
#     real = test_Y.T
#     pred = prediction.T
#     for i in range(n_figs):
#         f = plt.figure(dpi = 200)
#         path,file = os.path.split(save_path)
#         suffix = '.dim{:0>2d}.jpg'.format(i)
#         plt.title(file + suffix)
#         raw = test_X[0,:,i]
#         plt.plot(np.hstack([raw, real[i,:test_length_r]]), 'b', label = 'real')
#         plt.plot(range(raw.shape[0], raw.shape[0]+pred.shape[1]),
#                  pred[i], 'r', label = 'pred')
#         plt.legend()
#         plt.savefig(save_path + suffix)

def load_data(data_path):
    suffix = data_path[data_path.rfind('.')+1:]
    if suffix == 'csv':
        data = pd.read_csv(data_path, index_col = 'datetime').values
    elif suffix == 'npy':
        data = np.load(data_path)
    else:
        raise ValueError("Undefined data format : '{}'".format(suffix))
    return data

# def prepare_data(data_path, input_length):
#     data = load_data(data_path)
#     if data.squeeze().ndim == 1: # if data is saved as an 1-d array, reshape it to 2-D
#         data = data.reshape([-1,1])
#     return data_util.SerieToPieces(data, piece_length = input_length)
    

if __name__ == '__main__':
    params = ParseDiscreteLstmParams()
    if not params['fit_generator']:
        raise ValueError("Only data generator supported here!\n"
                         "Try add argument '--fit_generator'")
    else:
        train_feeder = data_util.SequentialDiscreteRCDF(
                    load_data(params['train_path']), 
                    params['batch_size'], params['batches_per_epoch'], 
                    params['input_length'], params['input_size'],
                    params['n_classes'], None)
        intervals = train_feeder.get_intervals()
        valid_feeder = data_util.SequentialDiscreteRCDF(
                    load_data(params['valid_path']), 
                    params['batch_size'], params['batches_per_epoch'], 
                    params['input_length'], params['input_size'],
                    params['n_classes'], intervals)

    if params['montecarlo']:
        predictor = pp.MCMCPredictor()
    else:
        predictor = dap.DetermDiscreteAGPredictor(params, intervals)
    
    if not params['test']: # train phase
        solver = eval(params['Solver'])(params)
        model = sequential.SequentialModel(solver=solver, predictor=predictor)
        if not params['fit_generator']:
            raise ValueError("Only data generator supported here!\n"
                             "Try add argument '--fit_generator'")
        else:
            train_with_generator(model, train_feeder, valid_feeder, params)
    else: # test phase
        test_feeder = data_util.SequentialDiscreteRCDF(
                    load_data(params['test_path']), 
                    params['batch_size'], params['batches_per_epoch'], 
                    params['input_length'], params['input_size'],
                    params['n_classes'], intervals)
        model = sequential.SequentialModel.load_model(params['model_path'])
        model._set_predictor(predictor)

        # multistep test   
        X, Y = test_feeder.get_singlstep_test_data(params['test_length'])
        true_class = Y.argmax(axis=-1)
        true_value = data_util.discrete2continue(true_class, intervals)
        prob, pred_value = model.singlstep_predict(X)
        pred_class = data_util.continue2discrete(pred_value, intervals)
        # draw figure
        f1 = draw(true_value, pred_value)
        f2 = draw(true_class, pred_class, prob)
        f1.savefig(params['model_path'] + 
            '.singlstep_pred.epoch{:0>4d}.test.jpg'.format(model.current_epoch))
        f2.savefig(params['model_path'] + 
            '.singlstep_pred.epoch{:0>4d}.hm..test.jpg'.format(model.current_epoch))

        # multistep test
        X, Y = test_feeder.get_multistep_test_data(params['test_length'])
        true_class = Y.argmax(axis=-1)
        true_value = data_util.discrete2continue(true_class, intervals)
        prob, pred_value = model.multistep_predict(X, params['test_length'])
        pred_class = data_util.continue2discrete(pred_value, intervals)
        # draw figure
        f1 = draw(true_value, pred_value)
        f2 = draw(true_class, pred_class, prob)
        f1.savefig(params['model_path'] + 
            '.multistep_pred.epoch{:0>4d}.test.jpg'.format(model.current_epoch))
        f2.savefig(params['model_path'] + 
            '.multistep_pred.epoch{:0>4d}.hm.test.jpg'.format(model.current_epoch))

        