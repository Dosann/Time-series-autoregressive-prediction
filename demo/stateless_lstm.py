import keras
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from numpy.random import choice

N_SAMPLES = 1200
INPUT_LENGTH = 10
N_TRAIN   = 1000
X = np.zeros([N_SAMPLES, INPUT_LENGTH*2, 1])
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

print('Build STATELESS model...')
model = Sequential()
model.add(LSTM(10, input_shape=(INPUT_LENGTH, 1), return_sequences=False, stateful=False))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x = np.concatenate([X_train[:,i:i+INPUT_LENGTH] for i in range(INPUT_LENGTH+1)], axis=0)
y = np.repeat(Y_train, INPUT_LENGTH+1, axis=1).T.flatten()[:,np.newaxis]

model.fit(x, y, batch_size=16, shuffle=True)

x_test = np.concatenate([X_test[:,i:i+INPUT_LENGTH] for i in range(INPUT_LENGTH+1)], axis=0)
y_test = np.repeat(Y_test, INPUT_LENGTH+1, axis=1).T.flatten()[:,np.newaxis]


losses, accuracy = model.evaluate(x_test, y_test)
print("test accuracy:", accuracy)
