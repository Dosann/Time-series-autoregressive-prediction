import numpy as np
from numpy import newaxis
from sklearn import preprocessing
from scipy import sparse
import pickle as pkl

# ---------------------------- normal data preprocessing -------------------------

def SerieToPieces(data, piece_length = 50, valid_ratio = 0.1, test_ratio = 0.1, shuffle = True):
    data = np.array(data)
    if len(data.shape) == 1:
        data = data.reshape([-1, 1])
    length = data.shape[0]
    n_pieces = length - piece_length
    pieces = [data[i : i + piece_length + 1] for i in range(n_pieces)]
    pieces = np.array(pieces)
    n_train = int((1 - valid_ratio - test_ratio) * n_pieces)
    n_valid = int(valid_ratio * n_pieces)
    n_test = n_pieces - n_train - n_valid
    train = pieces[: n_train]
    valid = pieces[n_train : n_train+n_valid]
    if shuffle:
        np.random.shuffle(train)
        np.random.shuffle(valid)
    test  = pieces[n_train+n_valid :]
    return (train[:,:-1,:], train[:,-1,:], valid[:,:-1,:], valid[:,-1,:], test[:,:-1,:], test[:,-1,:])


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


# ----------------------------- data generator ------------------------------

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
        print("out_size is {}".format(out_size))
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


# --------------------------- data normalization -------------------------

class Normalizer:

    def __init__(self):
        self._normalizer = preprocessing.MinMaxScaler(feature_range = (-1, 1))

    def _check_data(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError("data to be normalized should be of type '{}'".format(np.ndarray))
        if data.ndim == 1:
            data = data.reshape([-1, 1])
        elif data.ndim >= 3:
            raise ValueError("data should be with dim == 1 or 2")
    
    def normalize(self, data):
        self._check_data(data)
        return self._normalizer.fit_transform(data)
    
    def save(self, path):
        with open(path, 'wb') as f:
            f.write(pkl.dumps(self))
