import numpy as np
from scipy.io import loadmat


def load_one_epoch(subject, data_type, epoch_num):
    if data_type is 'train':
        epoch_data=loadmat("data/{}{}-allfilt20.mat".format(subject, epoch_num))

    x=epoch_data['x']
    y=epoch_data['y']
    code=epoch_data['code']

    return x, y, code

def load_data(subject, data_type, num_epoches):
    X = list()
    Y = list()
    C = list()
    for epoch_num in range(num_epoches):
        #print(epoch_num)
        x, y, code = load_one_epoch(subject, data_type, 1+epoch_num)

        X.append(x)
        Y.append(y)
        C.append(code)
        


    X = np.array(X)
    Y = np.array(Y)
    C = np.array(C)

    print('loaded:')
    print(X.shape)
    print(Y.shape)
    print(C.shape)    

    num_trials = X.shape[1]
    data_dim = X.shape[2]

    # stack epoches
    X = X.reshape(-1,data_dim)
    Y = Y.ravel()
    C = C.ravel()    

    print('stacked:')
    print(X.shape)
    print(Y.shape)
    print(C.shape)    
    
    return X, Y, C


def load_channel_mask(subject, num_channels):
    return np.loadtxt('mask/{}_chosen_channel_mask_{}.txt'.format(subject,num_channels), dtype=int)


def apply_mask(X, mask):
    data_dim = X.shape[1]
    num_channels = len(mask)
    samples_per_channel = int(data_dim / num_channels)
    
#     print(X.shape)
#     print(data_dim)
#     print(num_channels)
#     print(samples_per_channel)    
    
    index = np.repeat(mask,samples_per_channel)
    
#     print(mask)
#     print(np.repeat(mask,samples_per_channel))

    return X[:,index>0]


