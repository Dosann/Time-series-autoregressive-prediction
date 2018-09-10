import numpy as np
from sklearn import preprocessing
from scipy import sparse

def SerieToPieces(data, piece_length = 50, valid_ratio = 0.1, test_ratio = 0.1, shuffle = True):
    length = len(data)
    n_pieces = length - piece_length
    pieces = np.array([data[i : i + piece_length + 1] for i in range(n_pieces)])
    n_train = int((1 - valid_ratio - test_ratio) * n_pieces)
    n_valid = int(valid_ratio * n_pieces)
    n_test = n_pieces - n_train - n_valid
    train = pieces[: n_train]
    valid = pieces[n_train : n_train+n_valid]
    if shuffle:
        np.random.shuffle(train)
        np.random.shuffle(valid)
    test  = pieces[n_train+n_valid :]
    return (train[:,:-1], train[:,-1], valid[:,:-1], valid[:,-1], test[:,:-1], test[:,-1])

def DatasetAbsoluteDiscretization(data, padding = '10%', num_class = 12, piece_length = 50, 
                                  valid_ratio = 0.1, test_ratio = 0.1, shuffle = True, return_sparse = False):
    ub = np.max(data)
    lb = np.min(data)
    # extend upperbound and lowerbound
    if type(padding) == str:
        pad_ratio = int(padding[:-1]) / 100
        ub += (ub - lb) * pad_ratio
        lb += -(ub - lb) * pad_ratio
    else:
        ub += padding
        lb += -padding
    # discretization
    interval_width = (ub - lb) / (num_class - 2)
    thresholds = np.arange(lb, lb + interval_width*(num_class-1), interval_width)
    find_class = lambda y:np.sum(thresholds < y)
    ohe = preprocessing.OneHotEncoder(sparse = sparse)
    ohe.fit_transform(np.array([np.arange(0, num_class)]).T)
    to_digits = lambda y:ohe.transform(np.array([[find_class(y)]]))
    train_x, train_y, valid_x, valid_y, test_x, test_y = SerieToPieces(data, piece_length, valid_ratio,
                                                                      test_ratio, shuffle)
    transfered_ys = []
    for ys in [train_y, valid_y, test_y]:
        digits_list = []
        for y in ys:
            digits_list.append(to_digits(y))
        if digits_list == []:
            transfered_ys.append([])
        else:
            if sparse:
                transfered_ys.append(sparse.vstack(digits_list))
            else:
                transfered_ys.append(np.vstack(digits_list))
    return (train_x, train_y, valid_x, valid_y, test_x, test_y), transfered_ys, to_digits, thresholds