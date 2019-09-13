from utils import *


if __name__ == '__main__':

    #build the data set
    data, targets = build_dataset()

    #build the net
    net = Net()

    #uncomment to display all data
    '''  
    fig = plt.figure()
    plt.rcParams.update({'font.size': 16})
    plt.plot(data[:, 0].cpu().numpy(), data[:, 1].cpu().numpy(), 'o')
    plt.xlabel(r'$t_{\rm delay, osc 2}(s)$')
    plt.ylabel(r'$t_{\rm delay, osc 3}(s)$')
    plt.grid()
    fig.tight_layout()
    plt.show()
    '''

    #uncomment to check data classes for each binary classification task
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
    '''

    #test forward pass with the net
    '''
    x = data[0, :]
    target = targets['aw'][0, :]
    y = net(x)
    y.size()
    print(y)
    print(target)
    '''

    #check list of network parameters that requires grad    
    '''
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)
            print(param.data)
    '''

    #train the net on vowel aw
    #classes_names = ['aw', 'er', 'iy', 'uw']
    vowel = 'er'
    targets_temp = targets[vowel]
    #lr = 0.01 for all vowels but er
    optimizer = optim.SGD(net.parameters(), lr = 0.1)
    criterion = nn.BCELoss()
    epochs = 1200
    acc_tab = []
    for epoch in range(epochs):
        train(net, optimizer, criterion, data, targets_temp)   
        acc = eval(net, data, targets_temp)
        acc_tab.append(acc)

    
    plt.figure()
    plt.rcParams.update({'font.size': 16})
    plt.plot(acc_tab)
    plt.grid()
    plt.xlabel(r'$\#$ samples')
    plt.ylabel('Accuracy')
    

    w = list(net.parameters())
    w0 = w[0].data.numpy()
    w1 = w[1].data.numpy()

    fig = plt.figure()
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
    fig.tight_layout()
    plt.show()








 
