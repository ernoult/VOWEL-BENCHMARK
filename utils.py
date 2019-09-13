from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.w = nn.Linear(2, 1)

    def forward(self, x):
        x = F.sigmoid(self.w(x))
        return x


def train(net, dataset, targets, epochs):
    net.train()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr = 0.00001)
    loss_tab = []
    for epoch in range(epochs):
        for i, data in enumerate(dataset):
            target = targets[i, :]
            optimizer.zero_grad()
            y = net(data)
            loss = criterion(y, target)
            loss.backward()
            optimizer.step()
            loss_tab.append(loss.item())

    fig = plt.figure(figsize = (15, 3))
    plt.rcParams.update({'font.size': 16})
    plt.plot(loss_tab)
    plt.ylabel(r'Loss')
    plt.xlabel(r'$\#$ Samples')
    plt.grid()
    fig.tight_layout()
    plt.show()

def build_dataset():
    #take raw data
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

    #normalize data by the maximal value
    data = data/torch.max(data)

    #create data labels
    targets = {}
    classes_names = ['aw', 'er', 'iy', 'uw']

    for ind, name in enumerate(classes_names):
        targets_temp = torch.zeros((data.size(0) , 1))
        targets_temp[ind*10: (ind + 1)*10, :] = 1
        targets.update({name : targets_temp})
        del targets_temp
        
    #shuffle the data base
    permuted_indices = torch.randperm(data.size(0))
    data = data[permuted_indices, :]

    for key, value in targets.items():
        targets[key] = value[permuted_indices, :]

    return data, targets
