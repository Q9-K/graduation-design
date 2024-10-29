import numpy as np
import math
class kFoldGenerator():
    '''
    Data Generator
    '''
    k = -1      # the fold number
    x_list = [] # x list with length=k
    y_list = [] # x list with length=k
    dist_list = []

    # Initializate
    def __init__(self, x, y, d):
        if len(x) != len(y):
            assert False, 'Data generator: Length of x or y is not equal to k.'
        self.k = len(x)
        self.x_list = x
        self.y_list = y
        self.dist_list = d

    # Get i-th fold
    def getFold(self, i, fold):
        isFirst = True
        mults = self.k/fold
        val_first = True
        for p in range(self.k):
            if (p < i*mults) or (p >= (i+1)*mults):
                if isFirst:
                    train_data = self.x_list[p]
                    train_targets = self.y_list[p]
                    train_dist = self.dist_list[p]

                    isFirst = False
                else:
                    train_data = np.concatenate((train_data, self.x_list[p]))
                    train_targets = np.concatenate((train_targets, self.y_list[p]))
                    train_dist = np.concatenate((train_dist, self.dist_list[p]))

            else:
                if val_first:
                    val_data = self.x_list[p]
                    val_targets = self.y_list[p]
                    val_dist = self.dist_list[p]
                    val_first = False
                else:
                    val_data = np.concatenate((val_data, self.x_list[p]))
                    val_targets = np.concatenate((val_targets, self.y_list[p]))
                    val_dist = np.concatenate((val_dist, self.dist_list[p]))

        return train_data, train_targets, val_data, val_targets, train_dist, val_dist

    # Get all data x
    def getX(self):
        All_X = self.x_list[0]
        for i in range(1, self.k):
            All_X = np.append(All_X, self.x_list[i], axis=0)
        return All_X

    # Get all label y (one-hot)
    def getY(self):
        All_Y = self.y_list[0][2:-2]
        for i in range(1, self.k):
            All_Y = np.append(All_Y, self.y_list[i][2:-2], axis=0)
        return All_Y

    # Get all label y (int)
    def getY_int(self):
        All_Y = self.getY()
        return np.argmax(All_Y, axis=1)


class DominGenerator():
    '''
    Domin Generator
    '''
    k = -1       # the fold number
    l_list = []  # length of each domin
    d_list = []  # d list with length=k

    # Initializate
    def __init__(self, len_list):    # dsy --> len_list 50*[150]
        # self.l_list = len_list[0:36]  # !fix 4 class
        # self.k = len(len_list) - 4
        self.l_list = len_list[0:45]
        self.k = len(len_list) - 5
        # self.l_list = len_list[0:40]   # fix dsy
        # self.k = len(len_list) - 10

    # Get i-th fold
    def getFold(self, i):
        isFirst = True
        isFirstVal = True
        fold = 9   # fix dsy
        mults = self.k / fold
        j = 0   #1~7  dsy 1~2
        ii = 0  #1~8  dsy 1~3
        for l in self.l_list:
            if (ii < i*mults) or (ii >= (i+1)*mults):
                a = np.zeros((l, fold-1), dtype=int)
                a[:, math.floor(j/mults)] = 1
                if isFirst:
                    train_domin = a
                    isFirst = False
                else:
                    train_domin = np.concatenate((train_domin, a))
                j += 1
            else:
                if isFirstVal:
                    val_domin = np.zeros((l, fold-1), dtype=int)
                    isFirstVal = False
                else:
                    a = np.zeros((l, fold-1), dtype=int)
                    val_domin = np.concatenate((val_domin, a))
            ii += 1
        return train_domin, val_domin

class DominGenerator_pretrain():
    '''
    Domin Generator
    '''
    k = -1       # the fold number
    l_list = []  # length of each domin
    d_list = []  # d list with length=k

    # Initializate
    def __init__(self, len_list):
        self.l_list = np.full(len_list[0], len(len_list))
        self.k = len_list[0]

    # Get i-th fold
    def getFold(self, i):
        isFirst = True
        isFirstVal = True
        fold = 12
        mults = self.k / fold
        j = 0   #1~7
        ii = 0  #1~8
        for l in self.l_list:
            if (ii < i*mults) or (ii >= (i+1)*mults):
                a = np.zeros((l, fold-1), dtype=int)
                a[:, math.floor(j/mults)] = 1
                if isFirst:
                    train_domin = a
                    isFirst = False
                else:
                    train_domin = np.concatenate((train_domin, a))
                j += 1
            else:
                if isFirstVal:
                    val_domin = np.zeros((l, fold-1), dtype=int)
                    isFirstVal = False
                else:
                    a = np.zeros((l, fold-1), dtype=int)
                    val_domin = np.concatenate((val_domin, a))
            ii += 1
        return train_domin, val_domin

class DominGenerator_new():
    '''
    Domin Generator
    '''
    k = -1  # the fold number
    l_list = []  # length of each domin
    d_list = []  # d list with length=k

    # Initializate
    def __init__(self, len_list):
        self.l_list = len_list
        self.k = len(len_list)

    # Get i-th fold
    def getFold(self, i):
        isFirst = True
        isFirstVal = True
        fold = 11
        mults = self.k / fold
        class_fold = 2
        j = 0  # 1~7
        ii = 0  # 1~8
        for l in self.l_list:
            if (ii < i * mults) or (ii >= (i + 1) * mults):
                a = np.zeros((l, 2), dtype=int)
                a[:, 1] = 1
                if isFirst:
                    train_domin = a
                    isFirst = False
                else:
                    train_domin = np.concatenate((train_domin, a))
                j += 1
            else:
                if isFirstVal:
                    a = np.zeros((l, 2), dtype=int)
                    a[:, 0] = 1
                    val_domin = a
                    isFirstVal = False
                else:
                    a = np.zeros((l, 2), dtype=int)
                    a[:, 0] = 1
                    val_domin = np.concatenate((val_domin, a))
            ii += 1
        return train_domin, val_domin


class DominGeneratorALL():
    '''
    Domin Generator
    '''
    k = -1       # the fold number
    l_list = []  # length of each domin
    d_list = []  # d list with length=k

    # Initializate
    def __init__(self, len_list):
        self.l_list = len_list
        self.k = len(len_list)

    # Get i-th fold
    def getFold(self, i, fold_files):
        isFirst = True
        isFirstVal = True
        fold = 11
        mults = self.k / fold
        print('k = ', self.k)
        j = 0   #1~7
        ii = 0  #1~8
        for l in self.l_list:
            if (ii < i*mults) or (ii >= (i+1)*mults):
                a = np.zeros((l, self.k), dtype=int)
                a[:, ii] = 1
                if isFirst:
                    train_domin = a
                    isFirst = False
                else:
                    train_domin = np.concatenate((train_domin, a))
                j += 1
            else:
                if isFirstVal:
                    val_domin = np.zeros((l, self.k), dtype=int)
                    isFirstVal = False
                else:
                    a = np.zeros((l, self.k), dtype=int)
                    val_domin = np.concatenate((val_domin, a))
            ii += 1
        return train_domin, val_domin