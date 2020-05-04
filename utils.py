from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.w = nn.Linear(2, 1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.out(self.w(x))
        return x


def train(net, optimizer, criterion, dataset, targets):
    net.train()
    for i, data in enumerate(dataset):
        target = targets[i, :]
        optimizer.zero_grad()
        y = net(data)
        loss = criterion(y, target)
        loss.backward()
        optimizer.step()


def eval(net, dataset, targets):
    net.eval()
    y = net(dataset)
    pred = torch.where(y >= 0.5, torch.ones_like(y), torch.zeros_like(y))
    #print(y)
    #print(pred)
    N_tot = len(dataset)
    N_correct = (pred == targets).sum().item()
    accuracy = 100*(1/N_tot)*N_correct
    print('Train accuracy: {} % ({} out of {})'.format(accuracy, N_correct, N_tot))
    return accuracy        
    

def build_dataset(R):
    #take raw data
    if R == 0:
        num_per_class = 10
        data = torch.tensor([[1, 0.7362],
                        [0.97959, 0.78947],
                        [1.14177, 0.7772],
                        [0.96386, 0.77922],
                        [0.98361, 0.77121],
                        [1.09389, 0.74166],
                        [0.84152, 0.7362],
                        [1.02564, 0.77821],
                        [1.0772, 0.7947],
                        [1.05079, 0.86083],

                        [0.83507, 1.2766],
                        [0.83276, 1.2],
                        [0.6993, 0.96931],
                        [0.80808, 1.21458],
                        [0.74813, 1.1583],
                        [0.81081, 1.19284], 
                        [0.75188, 1.21212],
                        [0.80214, 1.0582],
                        [0.7717, 1.19761],
                        [0.7109, 1.12994],

                        [0.42766, 1.36055],
                        [0.41523, 1.37931],
                        [0.44395, 1.57895],
                        [0.43368, 1.40187],
                        [0.44395, 1.73411],
                        [0.44843, 1.36674],
                        [0.44626, 1.52284],
                        [0.40445, 1.21704],
                        [0.42105, 1.35747],
                        [0.4187, 1.25786],

                        [1.1236, 1.19284],
                        [1.20724, 1.31868],
                        [1.07817, 1.373],
                        [1.10701, 1.5625],
                        [1.09489, 1.19048],
                        [1.07817, 1.2685],
                        [1.20241, 1.25786],
                        [1.05634, 1.34832],
                        [1.07431, 1.24224],
                        [1.54242, 1.28755]])   


    else:
        raw_data = pd.read_csv(os.getcwd() + '/' + 'DispersedDataBase/R='+ str(R) +'.txt', delimiter=",\t", header = None)
        raw_data = raw_data.to_numpy()
        n_voy = int(np.shape(raw_data)[0]/2)
        num_per_class = np.shape(raw_data)[1]
        data = np.zeros((n_voy*np.shape(raw_data)[1], 2))       
        for i in range(n_voy):
            data[i*num_per_class: (i + 1)*num_per_class, 0] = raw_data[2*i, :]/400
            data[i*num_per_class: (i + 1)*num_per_class, 1] = raw_data[2*i + 1, :]/400

        data = torch.tensor(data).float()

    #create data labels
    targets = {}
    classes_names = ['aw', 'er', 'iy', 'uw']

    for ind, name in enumerate(classes_names):
        targets_temp = torch.zeros((data.size(0) , 1))
        targets_temp[ind*num_per_class: (ind + 1)*num_per_class, :] = 1
        targets.update({name : targets_temp})
        del targets_temp
        
    #shuffle the data base
    permuted_indices = torch.randperm(data.size(0))
    data = data[permuted_indices, :]

    for key, value in targets.items():
        targets[key] = value[permuted_indices, :]
    

    return data, targets

def plot_data(data, targets):

    #uncomment to display all data
    '''
    fig = plt.figure()
    plt.rcParams.update({'font.size': 16})
    plt.plot(data[:, 0].cpu().numpy(), data[:, 1].cpu().numpy(), 'o')
    plt.xlabel(r'$t_{\rm delay, osc 2}(s)$')
    plt.ylabel(r'$t_{\rm delay, osc 3}(s)$')
    plt.grid()
    fig.tight_layout()
    #plt.show()
    '''

    fig = plt.figure(figsize = (15, 3))
    plt.rcParams.update({'font.size': 16})
    for ind, key in enumerate(targets.keys()): 
        plt.subplot(1, len(targets), ind + 1)           
        data_x = data[:, 0].unsqueeze(1)
        data_y = data[:, 1].unsqueeze(1)
        plt.plot(data_x[targets[key] == 1].cpu().numpy(), data_y[targets[key] == 1].cpu().numpy(), 'o', color = 'red')
        plt.plot(data_x[targets[key] == 0].cpu().numpy(), data_y[targets[key] == 0].cpu().numpy(), 'o', color = 'blue')
        plt.xlabel(r'$t_{\rm delay, osc 2}(s)$')
        plt.ylabel(r'$t_{\rm delay, osc 3}(s)$')
        plt.title(key)
        plt.grid()
        del data_x, data_y
    fig.tight_layout()
    plt.show()

def plot_results(data, targets, vowel, x_tab, acc_tab, net):

    fig0 = plt.figure()
    plt.rcParams.update({'font.size': 16})
    
    plt.plot(x_tab, acc_tab, linewidth = 3, alpha = 0.8)
    plt.grid()
    plt.xlabel(r'$\#$ samples')
    plt.ylabel('Accuracy')
    plt.ticklabel_format(style = 'sci', axis ='x',  scilimits = (0,0))
    fig0.tight_layout()	
    

    w = list(net.parameters())
    w0 = w[0].data.numpy()
    w1 = w[1].data.numpy()

    fig1 = plt.figure()
    plt.rcParams.update({'font.size': 16})
    data_x = data[:, 0].unsqueeze(1)
    data_y = data[:, 1].unsqueeze(1)
    plt.plot(data_x[targets[vowel] == 1].cpu().numpy(), data_y[targets[vowel] == 1].cpu().numpy(), 'o', color = 'red')
    plt.plot(data_x[targets[vowel] == 0].cpu().numpy(), data_y[targets[vowel] == 0].cpu().numpy(), 'o', color = 'blue')

    x_axis = np.linspace(0, 1.5, 100)
    y_axis = -(w1[0] + x_axis*w0[0][0]) / w0[0][1]
    line_up, = plt.plot(x_axis, y_axis,'r--')
    plt.xlabel(r'$t_{\rm delay, osc 2}(s)$')
    plt.ylabel(r'$t_{\rm delay, osc 3}(s)$')
    plt.grid()
    fig1.tight_layout()
    plt.show()

