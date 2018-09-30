
import sys
import os
import matplotlib.pyplot as plt
import pickle as pkl

path = sys.argv[1]
print('path : {}'.format(path))
hist = pkl.loads(open(path, 'rb').read())

x = [epoch['epoch'] for epoch in hist]
rmse_singlstep_train  = [epoch['train.singlstep.rmse'] for epoch in hist]
rmse_singlstep_valid1 = [epoch['valid.1.singlstep.rmse'] for epoch in hist]
rmse_singlstep_valid2 = [epoch['valid.2.singlstep.rmse'] for epoch in hist]

f = plt.figure(dpi=200)
plt.plot(x, rmse_singlstep_train, label='singlstep.train')
plt.plot(x, rmse_singlstep_valid1, label='singlstep.valid.1')
plt.plot(x, rmse_singlstep_valid2, label='singlstep.valid.2')
plt.legend()
plt.show()