
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

rmse_multistep_determ_train = [epoch['train.multistep.determ.rmse'] for epoch in hist]
rmse_multistep_determ_valid1 = [epoch['valid.1.multistep.determ.rmse'] for epoch in hist]
rmse_multistep_determ_valid2 = [epoch['valid.2.multistep.determ.rmse'] for epoch in hist]

if 'train.multistep.mcmc.rmse' in hist[0]:
    rmse_multistep_mcmc_train = [epoch['train.multistep.mcmc.rmse'] for epoch in hist]
    rmse_multistep_mcmc_valid1 = [epoch['valid.1.multistep.mcmc.rmse'] for epoch in hist]
    rmse_multistep_mcmc_valid2 = [epoch['valid.2.multistep.mcmc.rmse'] for epoch in hist]


print(rmse_singlstep_train[-1], rmse_singlstep_valid1[-1], rmse_singlstep_valid2[-1])
print(rmse_multistep_determ_train[-1], rmse_multistep_determ_valid1[-1],
    rmse_multistep_determ_valid2[-1])
if 'train.multistep.mcmc.rmse' in hist[0]:
    print(rmse_multistep_mcmc_train[-1], rmse_multistep_mcmc_valid1[-1],
        rmse_multistep_mcmc_valid2[-1])

f = plt.figure(dpi=200)
plt.plot(x, rmse_singlstep_train, marker='o', label='singlstep.train')
plt.plot(x, rmse_singlstep_valid1, marker='o', label='singlstep.valid.1')
plt.plot(x, rmse_singlstep_valid2, marker='o', label='singlstep.valid.2')
plt.legend()

f = plt.figure(dpi=200)
plt.plot(x, rmse_multistep_determ_train, marker='o', label='multistep.determ.train')
plt.plot(x, rmse_multistep_determ_valid1, marker='o', label='multistep.determ.valid.1')
plt.plot(x, rmse_multistep_determ_valid2, marker='o', label='multistep.determ.valid.2')
plt.legend()

if 'train.multistep.mcmc.rmse' in hist[0]:
    f = plt.figure(dpi=200)
    plt.plot(x, rmse_multistep_mcmc_train, marker='o', label='multistep.mcmc.train')
    plt.plot(x, rmse_multistep_mcmc_valid1, marker='o', label='multistep.mcmc.valid.1')
    plt.plot(x, rmse_multistep_mcmc_valid2, marker='o', label='multistep.mcmc.valid.2')
    plt.legend()



plt.show()