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

def equalprob_interval_dividing(data, n_intervals):
    st_data = np.sort(data.reshape(-1))
    n_samples = st_data.shape[0]
    divide_ids = np.linspace(0,n_samples,n_intervals+1).astype(np.int32)
    divide_ids[-1] += -1
    intervals = st_data[divide_ids]
    # modify interval bound for first/last interval
    origin_bound = [intervals[0], intervals[-1]]
    finterv_mean = st_data[divide_ids[0]:divide_ids[1]].mean()
    linterv_mean = st_data[divide_ids[-2]:divide_ids[-1]+1].mean()
    intervals[0] += -(intervals[1]-finterv_mean)
    intervals[-1] += (linterv_mean-intervals[-2])
    print("Original bound ({:3.3f} , {:3.3f}) is modified to ({:3.3f} , {:3.3f})"
          .format(origin_bound[0], origin_bound[1], intervals[0], intervals[-1]))
    return intervals

def equalwidth_interval_dividing(data, n_intervals):
    st_data = np.sort(data.reshape(-1))
    n_samples = st_data.shape[0]
    start = st_data[0]
    end = st_data[-1]
    intervals = np.linspace(start, end, n_intervals+1)
    # modify interval bound for first/last interval
    origin_bound = [intervals[0], intervals[-1]]
    finterv_mean = st_data[np.logical_and(st_data>=intervals[0], st_data<intervals[1])] \
                        .mean()
    linterv_mean = st_data[np.logical_and(st_data>intervals[-2], st_data<=intervals[-1])] \
                        .mean()
    intervals[0] += -(intervals[1]-finterv_mean)
    intervals[-1] += (linterv_mean-intervals[-2])
    print("Original bound ({:3.3f} , {:3.3f}) is modified to ({:3.3f} , {:3.3f})"
          .format(origin_bound[0], origin_bound[1], intervals[0], intervals[-1]))
    return intervals

def continue2discrete(data, intervals):
    dsc_data = np.zeros_like(data)
    for i,(start,end) in enumerate(zip(intervals[:-1], intervals[1:])):
        dsc_data[np.logical_and(data>=start, data<end)] = i
    return dsc_data.astype(np.int32)

def discrete2continue(data, intervals, random=False):
    n_intervals = len(intervals)
    if random:
        MAP = {i:(intervals[i], intervals[i+1]) for i in range(n_intervals-1)}
        ctn_interval = np.array(np.vectorize(MAP.get)(data))
        rand = np.random.uniform(0, 1e5, size=ctn_interval[0].shape)
        ctn_data = rand % (ctn_interval[1]-ctn_interval[0]) + (ctn_interval[0])
    else:
        MAP = {i:(intervals[i]+intervals[i+1])/2 for i in range(n_intervals-1)}
        ctn_data = np.vectorize(MAP.get)(data)
    return ctn_data

def label2onehot(data, n_labels):
    shape = data.shape
    ohe = preprocessing.OneHotEncoder(n_values=n_labels)
    ohe.fit(np.arange(n_labels, dtype=np.int32).reshape(-1,1))
    return ohe.transform(data.reshape(-1,1)).reshape(shape)


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
        self._data_preparation()

    # DO NOT USE THIS FUNCTION !!!
    # def extract_valid_test_data(self, valid_split, test_split,
    #                            channels = None):
    #     # default: select the first 'out_size' channels as valid/test dataset.
    #     if channels is None:
    #         channels = range(self.out_size)
    #     # split
    #     n_valid = int((self._to - self._from) * valid_split)
    #     n_test  = int((self._to - self._from) * test_split)
    #     _from_test   = self._to - n_test
    #     _from_valid  = _from_test - n_valid
    #     valid_X, valid_Y = Seq2SuccessiveXYBatch(self.data[_from_valid:, channels], 
    #                                              n_valid, self.out_length)
    #     test_X, test_Y   = Seq2SuccessiveXYBatch(self.data[_from_test:, channels], 
    #                                              n_test, self.out_length)
    #     self._to = _from_valid
    #     return valid_X, valid_Y, test_X, test_Y

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
        
    def _data_preparation(self):
        self.X = self.data
        self.Y = self.data

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
            X[i] = self.X[_from[i]:_to[i], channel_ids]
            Y[i] = self.Y[_to[i], channel_ids]
        X = np.array(X)
        Y = np.array(Y)
        self.batch += 1
        return X, Y

    def get_multistep_test_data(self, length):
        # output:
        #   X: 1 sample (1, out_length, out_size)
        #   Y: 2-d array (length, out_size)
        if length > self.n_timestep - self.out_length:
            raise ValueError("Queried length exceeded limit.\n"
                             "Maximum length: {}"
                             .format(self.n_timestep-self.out_length))
        return (self.X[np.newaxis, :self.out_length, :self.out_size], 
                self.Y[self.out_length:self.out_length+length, :self.out_size])

    def get_singlstep_test_data(self, length):
        # output:
        #   X: 'length' samples (length, out_length, out_size)
        #   Y: 2-d array (length, out_size)
        #      starting from 'out_length'
        if length > self.n_timestep - self.out_length:
            raise ValueError("Queried length exceeded limit.\n"
                             "Maximum length: {}"
                             .format(self.n_timestep-self.out_length))
        return (np.array([self.X[i:i+self.out_length, :self.out_size] 
                    for i in range(length)]),
                self.Y[self.out_length:self.out_length+length, :self.out_size])


class SequentialDiscreteRCDF:

    def __init__(self, data, batch_size, batches_per_epoch, 
                 out_length, out_size, n_classes,
                 intervals=None,
                 interv_dividing_method = equalprob_interval_dividing):
        print("out_size is {}".format(out_size))
        self.data = data
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.out_length = out_length
        self.out_size = out_size
        self.n_classes = n_classes
        self._check_data()
        self._reset()
        self._set_intervals(intervals, interv_dividing_method)
        self._data_preparation()
    
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

    def _set_intervals(self, intervals, interv_dividing_method):
        if intervals is None:
            self.intervals = interv_dividing_method(
                self.data, self.n_classes)
        else:
            self.intervals = intervals
    
    def get_intervals(self):
        return self.intervals
        
    def _data_preparation(self):
        # self.X : (n_timestep, n_chan)
        self.X = self.data
        # self.Y : (n_timestep, n_chan)
        self.Y = continue2discrete(self.data, self.intervals)
        self._onehotecd_preparation()
        # self.Y : (n_chan, n_timestep, n_classes)
        self.Y = np.array([self.ohe.transform(self.Y[:,i].reshape(-1,1)).todense() 
                    for i in range(self.Y.shape[1])])
    
    def _onehotecd_preparation(self):
        self.ohe = preprocessing.OneHotEncoder(n_values=self.n_classes)
        self.ohe.fit(np.arange(self.n_classes, dtype=np.int32)
                        .reshape(-1,1))
    
    def __next__(self):
        X = [None] * self.batch_size
        Y = [None] * self.batch_size
        _from = np.random.randint(self._from, self._to, self.batch_size)
        _to   = _from + self.out_length
        for i in range(self.batch_size):
            channel_ids = np.random.permutation(self.n_chan)[:self.out_size]
            X[i] = self.X[_from[i]:_to[i], channel_ids]
            Y[i] = self.Y[channel_ids, _to[i], :]
        # Y : list of (out_size, n_classes)
        X = np.array(X)
        Y = np.array(Y).transpose(1,0,2)
        Y = [y for y in Y]
        self.batch += 1
        return X, Y
    
    def get_multistep_test_data(self, length, random=False):
        # output:
        #   X: 1 sample (1, out_length, out_size)
        #   Y: 3-d array (length, out_size, n_classes)
        #      starting from 'out_length'
        if length > self.n_timestep - self.out_length:
            raise ValueError("Queried length exceeded limit.\n"
                             "Maximum length: {}"
                             .format(self.n_timestep-self.out_length))
        X, Y = (self.X[np.newaxis, :self.out_length, :self.out_size], 
                self.Y[:self.out_size, self.out_length:self.out_length+length, :]
                    .transpose(1,0,2))
        return X, Y

    def get_singlstep_test_data(self, length, random=False):
        # output:
        #   X: 'length' samples (length, out_length, out_size)
        #   Y: 3-d array (length, out_size, n_classes)
        #      starting from 'out_length'
        if length > self.n_timestep - self.out_length:
            raise ValueError("Queried length exceeded limit.\n"
                             "Maximum length: {}"
                             .format(self.n_timestep-self.out_length))
        return (np.array([self.X[i:i+self.out_length, :self.out_size] 
                    for i in range(length)]),
                self.Y[:self.out_size, self.out_length:self.out_length+length, :]
                    .transpose(1,0,2))

class SeqDiscrtzdRandomChanlStatefulDF:
    # sequential discretized random channel data feeder for stateful model

    def __init__(self, data, batch_size, batches_per_epoch, 
                 out_length, out_size, n_classes,
                 intervals=None,
                 interv_dividing_method = equalprob_interval_dividing):
        # data should be of shape (n_timestep, n_chan)
        print("out_size is {}".format(out_size))
        self.data = data
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.out_length = out_length
        self.out_size = out_size
        self.n_classes = n_classes
        self._check_data()
        self._reset()
        self._set_intervals(intervals, interv_dividing_method)
        self._data_preparation()
    
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
        self.curr_timestep = 0
        self.channel_ids = [np.random.permutation(self.n_chan)[:self.out_size] 
                                for x in range(self.batch_size)] # list of np.array of shape (output_size,)

    def _set_intervals(self, intervals, interv_dividing_method):
        if intervals is None:
            self.intervals = interv_dividing_method(
                self.data, self.n_classes)
        else:
            self.intervals = intervals
    
    def get_intervals(self):
        return self.intervals
        
    def _data_preparation(self):
        # self.X : (n_timestep, n_chan)
        self.X = self.data
        # self.Y : (n_timestep, n_chan)
        self.Y = continue2discrete(self.data, self.intervals)
        self._onehotecd_preparation()
        # self.Y : (n_chan, n_timestep, n_classes)
        self.Y = np.array([self.ohe.transform(self.Y[:,i].reshape(-1,1)).todense() 
                    for i in range(self.Y.shape[1])])
    
    def _onehotecd_preparation(self):
        self.ohe = preprocessing.OneHotEncoder(n_values=self.n_classes)
        self.ohe.fit(np.arange(self.n_classes, dtype=np.int32)
                        .reshape(-1,1))
    
    @property
    def reset_cycle_length(self):
        n_cycle = int((self.data.shape[0]-1) / self.out_length)
        return n_cycle

    def __next__(self):
        if self.curr_timestep + self.out_length >= self.data.shape[0]:
            self._reset()
        _from = self.curr_timestep
        _to   = _from + self.out_length
        X = [None] * self.batch_size
        Y = [None] * self.batch_size
        for i in range(self.batch_size):
            X[i] = self.X[_from:_to, self.channel_ids[i]]
            Y[i] = self.Y[self.channel_ids[i], _to, :]
        # Y : list of (out_size, n_classes)
        X = np.array(X) # (batch_size, out_length, out_size)
        Y = np.array(Y).transpose(1,0,2) # (out_size, batch_size, n_classes)
        Y = [y for y in Y] # list of (batch_size, n_classes)
        self.batch += 1
        self.curr_timestep += self.out_length
        return X, Y
    
    def get_multistep_test_data(self, length, lead_length=100, random=False):
        # input:
        #   lead_length: The length of the lead sequence not used for 
        #                evaluation (for stateful model).
        #   length: The length of the lead sequence used for evaluation.
        # output:
        #   out_length is set to 1 no matter what self.out_length is.
        #   X: lead+1 samples (lead_length+1, 1, out_size)
        #   Y: 3-d array (length, out_size, n_classes)
        #      starting from 'lead_length+1'
        print("lead_length+length: ", lead_length+length)
        if lead_length + length > self.n_timestep - self.out_length:
            raise ValueError("Queried total length exceeded limit.\n"
                             "Maximum total length: {}"
                             .format(self.n_timestep-self.out_length))
        X = [self.X[i:i+1, :self.out_size] 
                for i in range(lead_length+1)]
        X = np.array(X)
        Y = self.Y[:self.out_size, lead_length+1:lead_length+1+length, :] \
                    .transpose(1,0,2)
        print("X.shape : ", X.shape)
        print("Y.shape : ", Y.shape)
        return X, Y

    def get_singlstep_test_data(self, length, lead_length=100, random=False):
        # input:
        #   lead_length: The length of the lead sequence not used for 
        #                evaluation (for stateful model).
        #   length: The length of the lead sequence used for evaluation.
        # output:
        #   out_length is set to 1 no matter what self.out_length is.
        #   X: 'length' samples (lead_length+length, 1, out_size)
        #   Y: 3-d array (length, out_size, n_classes)
        #      starting from 'lead_length+1'
        if length > self.n_timestep - self.out_length:
            raise ValueError("Queried length exceeded limit.\n"
                             "Maximum length: {}"
                             .format(self.n_timestep-self.out_length))
        return (np.array([self.X[i:i+1, :self.out_size] 
                    for i in range(lead_length+length)]),
                self.Y[:self.out_size, lead_length+1:lead_length+1+length, :]
                    .transpose(1,0,2))



if __name__ == '__main__':
    data = np.load('../../data/NYSEtop80.1h.preprcd/train.npy')
    datafeeder = SequentialDiscreteRCDF(data, 64, 1000, 100, 6, 256)
    X, Y = next(datafeeder)
    print(X.shape)
    print(len(Y), Y[0].shape)


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

class NoshiftNormalizer:

    def __init__(self):
        self._normalizer = preprocessing.StandardScaler(with_mean=False, with_std=True)

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

