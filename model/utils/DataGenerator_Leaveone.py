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
        mults = int(self.k/fold)
        val_first = True
        test_first = True
        for p in range(self.k - mults):
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
        for p in range(mults):
            if test_first:
                test_data = self.x_list[self.k - mults + p]
                test_targets = self.y_list[self.k - mults + p]
                test_dist = self.dist_list[self.k - mults + p]
                test_first = False
            else:
                test_data = np.concatenate((test_data, self.x_list[self.k - mults + p]))
                test_targets = np.concatenate((test_targets, self.y_list[self.k - mults + p]))
                test_dist = np.concatenate((test_dist, self.dist_list[self.k - mults + p]))

        return train_data, train_targets, val_data, val_targets, train_dist, val_dist, test_data, test_targets, test_dist

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

class kFoldGenerator_avg():
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
        mults = int(self.k/fold)
        val_first = True
        test_first = True
        for p in range(self.k - mults):
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
        for p in range(mults):
            if test_first:
                test_data = self.x_list[self.k - mults + p]
                test_targets = self.y_list[self.k - mults + p]
                test_dist = self.dist_list[self.k - mults + p]
                test_first = False
            else:
                test_data = np.concatenate((test_data, self.x_list[self.k - mults + p]))
                test_targets = np.concatenate((test_targets, self.y_list[self.k - mults + p]))
                test_dist = np.concatenate((test_dist, self.dist_list[self.k - mults + p]))
        if i == 7:
            for p in range(40, 45):
                train_data = np.concatenate((train_data, self.x_list[p], self.x_list[p]))
                train_targets = np.concatenate((train_targets, self.y_list[p], self.y_list[p]))
                train_dist = np.concatenate((train_dist, self.dist_list[p], self.dist_list[p]))
        if i == 8:
            for p in range(36, 40):
                train_data = np.concatenate((train_data, self.x_list[p], self.x_list[p], self.x_list[p]))
                train_targets = np.concatenate((train_targets, self.y_list[p], self.y_list[p], self.y_list[p]))
                train_dist = np.concatenate((train_dist, self.dist_list[p], self.dist_list[p], self.dist_list[p]))

        return train_data, train_targets, val_data, val_targets, train_dist, val_dist, test_data, test_targets, test_dist

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