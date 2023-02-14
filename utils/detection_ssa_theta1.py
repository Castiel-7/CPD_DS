import numpy as np

import math

class SubspaceMethod(object):
    def __init__(self, threshold = 0.95, r = None):
        self.threshold = threshold
        self.r = r
    
    def fit(self, train_X):
        if train_X.shape[0]<train_X.shape[1]:
            self.fit_dual(train_X)
        else:
            self.fit_primal(train_X)

    def _get_dim(self, e_val):
        if self.r is not None:
            return self.r
        else:
            sum_all = np.sum(e_val)
            sum_value = np.array([np.sum(e_val[:i])/sum_all for i in range(1,len(e_val)+1)])
            r = np.min(np.where(sum_value>=self.threshold)[0])+1
            return r

    def fit_primal(self, X):
        K = X.T@X/X.shape[0]
        e_val, e_vec = np.linalg.eigh(K)
        e_val, e_vec = e_val[::-1], e_vec.T[::-1].T
        zero_idx = np.where(e_val>0)
        e_val, e_vec = e_val[zero_idx], e_vec.T[zero_idx].T
        r = self._get_dim(e_val)
        self.coef_ = e_val[:r]
        self.components_ = e_vec.T[:r].T

    def fit_dual(self, X):
        #print(X.shape)
        K = X@X.T/X.shape[0]
        e_val, e_vec = np.linalg.eigh(K)
        e_val, e_vec = e_val[::-1], e_vec.T[::-1].T
        zero_idx = np.where(e_val>0)
        e_val, e_vec = e_val[zero_idx], e_vec.T[zero_idx].T
        r = self._get_dim(e_val)
        V = X.T@e_vec/np.sqrt(e_val.reshape(1,-1)*X.shape[0])
        self.coef_ = e_val[:r]
        self.components_ = V.T[:r].T

    def score(self, test_X):
        I = np.identity(test_X.shape[1])
        error = np.linalg.norm(test_X@(I-self.components_@self.components_.T), axis=1).reshape(-1)/np.linalg.norm(test_X, axis=1).reshape(-1)
        return np.fabs(error)

class SSAtheta1(object):
    def __init__(self, window_length = 128, order = 64, lag = 64, M = 5, N = 10):
        self.window_length = window_length
        self.order = order
        self.lag = lag
        self.M = M
        self.N = N
        
    def predict(self, x):
        start_idx = 0
        end_idx = len(x) - self.window_length - self.order - self.lag
        
        score_list2 = []
        count = 0
        #print(start_idx)
        #print(end_idx)
        for t in range(start_idx, end_idx):
            train_H = self._get_hankel(x, order = self.order,
                                       start = t,
                                       end = t + self.window_length)
            test_H = self._get_hankel(x, order = self.order,
                                     start = t + self.lag,
                                     end = t + self.window_length + self.lag)
            sm = SubspaceMethod(r = self.M)
            sm.fit(train_H.T)
            subspace1 = sm.components_
            #print("SP1", subspace1.shape)
            sm = SubspaceMethod(r = self.N)
            sm.fit(test_H.T)
            subspace2 = sm.components_
            #print("SP2", subspace2.shape)
            _, S, _ = np.linalg.svd(subspace1.T@subspace2)
            #print("S", S, S.shape)
            score_list1 = []
            score_list1.append(1 - np.max(S))
            score_list1.append(1 - np.mean(S[0:5]))
            score_list1.append(1 - np.mean(S))
            score_list2.append(score_list1)
        change_vector = np.array(score_list2)
        return np.array(change_vector)

    def _get_hankel(self, x, order, start, end):
        return np.array([x[start+i:end+i] for i in range(order)]).T