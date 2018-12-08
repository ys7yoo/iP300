import numpy as np
from scipy.io import loadmat

def load_one_epoch(subject, data_type, epoch_num):
    if data_type is 'train':
        epoch_data = loadmat("data/{}{}-allfilt10.mat".format(subject, epoch_num))

        x = epoch_data['x']
        y = epoch_data['y']
        code = epoch_data['code']

    elif data_type is 'test':
        epoch_data = loadmat("data/{}t{}-allfilt10.mat".format(subject, epoch_num))

        x = epoch_data['x']
        # y=epoch_data['y']
        code = epoch_data['code']

        # decode y
        # print('decode y')
        true_code = np.loadtxt("data/{}t_{}_true_code.txt".format(subject, epoch_num))

        y = -np.ones(code.shape)
        idx = (code == true_code[0]) | (code == true_code[1])
        y[idx] = 1

    else:
        pass

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


def extract_by_code(X, Y, C, code):
    idx = (C == code)

    return X[idx,:], Y[idx], C[idx]


def sort_by_code(X, Y, C):

    Xs = list()
    Ys = list()
    Cs = list()
    for code in range(12):
        xx, yy, cc = extract_by_code(X, Y, C, code+1)

        # append to lists
        Xs.append(xx)
        Ys.append(yy)
        Cs.append(cc)


    XX = np.swapaxes(np.array(Xs), 0, 1)
    YY = np.swapaxes(np.array(Ys), 0, 1)
    CC = np.swapaxes(np.array(Cs), 0, 1)

    num_samples = YY.shape[0] * YY.shape[1]
    #print(num_samples)

    XX = XX.reshape(num_samples, -1)
    YY = YY.reshape(num_samples, -1)
    CC = CC.reshape(num_samples, -1)

    return XX, YY, CC


def to_one_hot_vec(idx):
    n = idx.shape[0]

    one_hot_vec = np.zeros((n,6))

    for i in range(n):
        one_hot_vec[i,idx[i]] = 1

    return one_hot_vec



def decode_rc(log_prob_diff):
    c_idx = np.argmax(log_prob_diff[:, :6], axis=1)

    r_idx = np.argmax(log_prob_diff[:, 6:], axis=1)

    return 2*np.hstack((to_one_hot_vec(c_idx),to_one_hot_vec(r_idx)))-1


def calc_accuracy_rc(YY_trial, decoded):
    return np.mean(np.sum(decoded == YY_trial, axis=1) == 12)
