from utils import *


if __name__ == '__main__':
    data, targets = build_dataset()
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

    #train the net on vowel aw
    targets_temp = targets['aw']
    train(net, data, targets_temp, 10)   










 
