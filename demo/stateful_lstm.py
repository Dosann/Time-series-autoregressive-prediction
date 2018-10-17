import keras
from keras.callbacks import Callback
from keras.models import Sequential, Model, Input
from keras.layers import LSTM, Dense
import numpy as np
from numpy.random import choice

N_SAMPLES = 1200
INPUT_LENGTH = 20
N_TRAIN   = 1000
BATCH_SIZE = 2
X = np.zeros([N_SAMPLES, INPUT_LENGTH, 1])
Y = np.zeros([N_SAMPLES, 1])
one_indexes = choice(a=N_SAMPLES, size=int(N_SAMPLES/2), replace=False)
X[one_indexes, 0, 0] = 1  # very long term memory.
Y[one_indexes, 0] = 1
X_train, Y_train = X[:N_TRAIN], Y[:N_TRAIN]
X_test, Y_test  = X[N_TRAIN:], Y[N_TRAIN:]

print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)
print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)

print('Build STATEFUL model...')
INPUT = Input(batch_shape=(1, 1, 1))
lstm1 = LSTM(10, return_sequences=True, stateful=True)(INPUT)
dense = Dense(1, activation='sigmoid')(lstm1)
model = Model(inputs=INPUT, outputs=dense)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

class ResetStatesCallback(Callback):
    def __init__(self):
        self.counter = 0

    def on_batch_begin(self, batch, logs={}):
        if self.counter % BATCH_SIZE == 0:
            self.model.reset_states()
        self.counter += 1

permutation = np.array([1,3,2,4]*int(N_TRAIN*BATCH_SIZE/4)) + np.repeat(np.arange(N_TRAIN*BATCH_SIZE/4, dtype=int),4)*4 - 1
x = X_train.flatten().reshape([-1,int(INPUT_LENGTH/BATCH_SIZE),1])[permutation]
y = np.repeat(Y_train, BATCH_SIZE, axis=0)[permutation]

model.fit(x, y, callbacks=[ResetStatesCallback()], batch_size=BATCH_SIZE, epochs=1, shuffle=False)

x_test = X_test.flatten().reshape([-1,int(INPUT_LENGTH/2),1])
y_test = np.repeat(Y_test, 2, axis=0)

def evaluate(model, x, y):
    n_samples = x.shape[0]
    preds = []
    model.reset_states()
    for i in range(1, n_samples+1):
        pred = model.predict(x[i-1:i])
        preds.append(pred)
        if i % 2 == 0:
            model.reset_states()
    preds = (np.round(np.array(preds).squeeze()>=0.5)).astype(int)
    print(preds)
    accuracy = (preds==y).mean()
    return accuracy, preds

accuracy, preds = evaluate(model, x_test, y_test.squeeze())
print("test accuracy:", accuracy)
