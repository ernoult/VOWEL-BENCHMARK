import argparse
from utils import *

parser = argparse.ArgumentParser(description='Feature Binding benchmark')
parser.add_argument(
    '--action',
    type=str,
    default='train',
    help='action to execute (default: train)')  
parser.add_argument(
    '--vowel',
    type=str,
    default='aw',
    help='selected vowel (default: aw)')
parser.add_argument(
    '--R',
    type=int,
    default=0,
    help='noise level (default: 0)')
parser.add_argument(
    '--lr',
    type=float,
    default=0.1,
    help='learning rate (default: 0.1)')
parser.add_argument(
    '--epochs',
    type=int,
    default=100,
    help='number of epochs (default: 100)')
parser.add_argument(
    '--database',
    action='store_true',
    default=False,
    help='generate database in Excel (default: False)')

args = parser.parse_args()


if __name__ == '__main__':

    if args.action == 'data':

        if not args.database:
            #build the dataset
            data, targets = build_dataset(args)

            #plot the dataset
            plot_data(data, targets)
        else:
            database = build_dataset(args)
            #print(database)
            pd_database = pd.DataFrame(database)
            pd_database.to_csv(os.getcwd() +'/database_R=' + str(args.R) +'.csv', index = False)

    elif args.action == 'train':

        #build the data set
        data, targets = build_dataset(args)
        N_samples = data.size(0) 

        #build the net
        net = Net()

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

        #train the net
        vowel = args.vowel
        targets_temp = targets[vowel]
        optimizer = optim.SGD(net.parameters(), lr = args.lr)
        criterion = nn.BCELoss()
        epochs = args.epochs
        acc_tab = []
        x_tab = []
        for epoch in range(epochs):
            train(net, optimizer, criterion, data, targets_temp)   
            acc = eval(net, data, targets_temp)
            acc_tab.append(acc)
            x_tab.append((epoch + 1)*N_samples)

        print('Total number of samples used for training: {}'.format(x_tab[-1]))
        
        plot_results(data, targets, vowel, x_tab, acc_tab, net)


    elif args.action == 'results':

        vowels_tab = ['aw', 'er', 'iy', 'uw']
        R_tab = [0, 10, 50, 100]
        #R_tab = [10]
        N_trials = 10
        for R in R_tab:
            results = {}
            results['aw'], results['er'], results['iy'], results['uw'] = [], [], [], []
            for vowel in vowels_tab:
                for n in range(N_trials):
                    print('R = {}, vowel: {}, trial: {}'.format(R, vowel, 1 + n))                  
                    data, targets = build_dataset(args, R_in = R)
                    N_samples = data.size(0) 
                    net = Net()
                    targets_temp = targets[vowel]
                    optimizer = optim.SGD(net.parameters(), lr = args.lr)
                    criterion = nn.BCELoss()
                    epochs = args.epochs
                    acc_tab = []
                    for epoch in range(epochs):
                        train(net, optimizer, criterion, data, targets_temp)

                    acc = eval(net, data, targets_temp)
                    results[vowel].append(acc)

            pd_results = pd.DataFrame(results)
            pd_results.to_csv(os.getcwd() +'/results_R=' + str(R) +'.csv', index = False)


    elif args.action == 'results-new':

        vowels_tab = ['aw', 'er', 'iy', 'uw']
        R_tab = [0, 10, 50, 100]
        #R_tab = [10]
        N_trials = 10

        for vowel in vowels_tab:
            results = {}
            for R in R_tab:
                results['R = ' + str(R)] = []
           
            for n in range(N_trials):
                print('vowel: {}, trial: {}'.format(vowel, 1 + n))
                #WATCH OUT: we only train on the R = 0 data                  
                data, targets = build_dataset(args, R_in = 0)
                N_samples = data.size(0) 
                net = Net()
                targets_temp = targets[vowel]
                optimizer = optim.SGD(net.parameters(), lr = args.lr)
                criterion = nn.BCELoss()
                epochs = args.epochs
                acc_tab = []
                for epoch in range(epochs):
                    train(net, optimizer, criterion, data, targets_temp)

                for R in R_tab:
                    data_test, targets_test = build_dataset(args, R_in = R)
                    targets_test_temp = targets_test[vowel]
                    acc = eval(net, data_test, targets_test_temp)
                    results['R = ' + str(R)].append(acc)

            pd_results = pd.DataFrame(results)
            pd_results.to_csv(os.getcwd() +'/results_vowel-' + vowel +'.csv', index = False)
 
