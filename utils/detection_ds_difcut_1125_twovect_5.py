import numpy as np
import matplotlib.pyplot as plt
import math

class SubspaceMethod(object):
    def __init__(self, threshold = 0.95, r = None):
        #print(1)
        self.threshold = threshold
        self.r = r
    
    def fit(self, train_X):
        if train_X.shape[0]<train_X.shape[1]:
            #print(2)
            self.fit_dual(train_X)
        else:
            #print(3)
            self.fit_primal(train_X)

    def _get_dim(self, e_val):
        if self.r is not None:
            #print(4)
            return self.r
        else:
            #print(5)
            sum_all = np.sum(e_val)
            sum_value = np.array([np.sum(e_val[:i])/sum_all for i in range(1,len(e_val)+1)])
            r = np.min(np.where(sum_value>=self.threshold)[0])+1
            return r

    def fit_primal(self, X):
        #print(6)
        K = X.T@X/X.shape[0]
        e_val, e_vec = np.linalg.eigh(K)
        e_val, e_vec = e_val[::-1], e_vec.T[::-1].T
        zero_idx = np.where(e_val>0)
        e_val, e_vec = e_val[zero_idx], e_vec.T[zero_idx].T
        r = self._get_dim(e_val)
        self.coef_ = e_val[:r]
        self.components_ = e_vec.T[:r].T

    def fit_dual(self, X):
        #print(7)
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
        #print(8)
        I = np.identity(test_X.shape[1])
        error = np.linalg.norm(test_X@(I-self.components_@self.components_.T), axis=1).reshape(-1)/np.linalg.norm(test_X, axis=1).reshape(-1)
        return np.fabs(error)

class SubspaceMethod1(object):
    def __init__(self, threshold = 0.95, r = None):
        #print(1)
        self.threshold = threshold
        self.r = r
    
    def fit(self, train_X):
        if train_X.shape[0]<train_X.shape[1]:
            #print(2)
            self.fit_dual(train_X)
        else:
            #print(3)
            self.fit_primal(train_X)

    def _get_dim(self, e_val):
        if self.r is not None:
            #print(4, self.r)
            return self.r
        else:
            #print(5)
            sum_all = np.sum(e_val)
            sum_value = np.array([np.sum(e_val[:i])/sum_all for i in range(1,len(e_val)+1)])
            r = np.min(np.where(sum_value>=self.threshold)[0])+1
            return r

    def fit_primal(self, X):
        #print(6)
        K = X.T@X/X.shape[0]
        e_val, e_vec = np.linalg.eigh(K)
        e_val, e_vec = e_val[::-1], e_vec.T[::-1].T
        zero_idx = np.where(e_val>0)
        e_val, e_vec = e_val[zero_idx], e_vec.T[zero_idx].T
        #print(e_val.shape)
        r = self._get_dim(e_val)
        self.coef_ = e_val[:r]
        self.components_ = e_vec.T[:r].T

    def fit_dual(self, X):
        #print(7)
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
        #print(8)
        I = np.identity(test_X.shape[1])
        error = np.linalg.norm(test_X@(I-self.components_@self.components_.T), axis=1).reshape(-1)/np.linalg.norm(test_X, axis=1).reshape(-1)
        return np.fabs(error)

class DetectionDS(object):
    def __init__(self, window_length=128, order=64, lag = 64, M = 5, N = 10, DS_dim = 5, PS_dim = 30, dif_cut = 0, rem = 1):
        self.window_length = window_length
        self.order = order
        self.lag = lag
        self.M = M
        self.N = N
        self.DS_dim = DS_dim
        self.PS_dim = PS_dim
        self.ds7 = np.zeros(10)
        self.dif_cut = dif_cut
        self.rem = rem

    def fit(self, x):
        start_idx = 0
        end_idx = x.shape[1] - self.window_length - self.order - self.lag
        ds_list = []
        pds_list = []
        score_list = []
        vol_ruiwa_train = []
        vol_log_train = []
        similarity_list_k1 = []
        count = 0
        count1 = 0
        for t in range(start_idx, end_idx):
            similarity_list_k0 = []
            for i in range(x.shape[0]):
                train_H = self._get_hankel(x[i, :], order = self.order,
                                        start = t,
                                        end = t + self.window_length)
                test_H = self._get_hankel(x[i, :], order = self.order,
                                        start = t + self.lag,
                                        end = t + self.window_length + self.lag)
                if i == 0:
                    self.past = train_H
                    self.present = test_H
                else:
                    self.past = np.concatenate([self.past, train_H])
                    self.present = np.concatenate([self.present, test_H])
            sm = SubspaceMethod(r = self.M)
            sm.fit(self.past.T)
            subspace1 = sm.components_
            
            sm = SubspaceMethod(r = self.N)
            sm.fit(self.present.T)
            subspace2 = sm.components_
            
            subspace1 = subspace1[:, self.rem:self.M]
            subspace2 = subspace2[:, self.rem:self.M]

            _, S, _ = np.linalg.svd(subspace1.T@subspace2)
            #print(np.mean(S))
            if np.mean(S) < self.dif_cut:
                continue
            count1 += 1
            volume_ruiwa = np.sum(S)
            volume_log = np.sum(np.log(S))
            similarity_list_k0.append(volume_ruiwa)
            similarity_list_k0.append(volume_log)
            vol_ruiwa_train.append(volume_ruiwa)
            vol_log_train.append(volume_log)

            G = subspace1 @ subspace1.T + subspace2 @ subspace2.T
            w, v = np.linalg.eigh(G)
            w, v = w[::-1], v[:, ::-1]
            rank = np.linalg.matrix_rank(G)
            w, v = w[:rank], v[:, :rank]

            d1 = v[:, self.DS_dim < w]
            w = w[0:d1.shape[1]]
            d2 = d1[:, 1 > w]
            if count1 == 1:
                self.ds7 = d2
            else:
                self.ds7 = np.concatenate([self.ds7, d2], 1)
            similarity_list_k1.append(similarity_list_k0)
        if self.ds7.shape[0] == 10:
            return np.array([[7, 7]])
        sm7 = SubspaceMethod1(r = self.PS_dim)
        sm7.fit(self.ds7.T)
        self.pds = sm7.components_
        self.vol_ruiwa_train = np.array(vol_ruiwa_train)
        self.mean_ruiwa = np.mean(self.vol_ruiwa_train)
        self.vol_log_train = np.array(vol_log_train)
        self.mean_log = np.mean(self.vol_log_train)
        similarity_list_k1 = np.array(similarity_list_k1)
        return similarity_list_k1
            
            
    def _get_hankel(self, x, order, start, end):
        return np.array([x[start+i:end+i] for i in range(order)]).T
    
    def predict_train(self, x):
        start_idx = 0
        end_idx = x.shape[1] - self.window_length - self.order - self.lag
        score_list = []
        similarity_list = []
        similarity_list_k1 = []
        vol_list = []
        count = 0
        for t in range(start_idx, end_idx):
            similarity_list_k0 = []
            for i in range(x.shape[0]):
                train_H = self._get_hankel(x[i, :], order = self.order,
                                        start = t,
                                        end = t + self.window_length)
                test_H = self._get_hankel(x[i, :], order = self.order,
                                        start = t + self.lag,
                                        end = t + self.window_length + self.lag)
                if i == 0:
                    self.past = train_H
                    self.present = test_H
                else:
                    self.past = np.concatenate([self.past, train_H])
                    self.present = np.concatenate([self.present, test_H])
            sm = SubspaceMethod(r = self.M)
            sm.fit(self.past.T)
            subspace1 = sm.components_
            
            sm = SubspaceMethod(r = self.N)
            sm.fit(self.present.T)
            subspace2 = sm.components_

            subspace1 = subspace1[:, self.rem:self.M]
            subspace2 = subspace2[:, self.rem:self.M]

            G = subspace1 @ subspace1.T + subspace2 @ subspace2.T
            w, v = np.linalg.eigh(G)
            w, v = w[::-1], v[:, ::-1]
            rank = np.linalg.matrix_rank(G)
            w, v = w[:rank], v[:, :rank]

            _, S, _ = np.linalg.svd(subspace1.T@subspace2)
            volume_ruiwa = np.sum(S)
            volume_log = np.sum(np.log(S))
            vol_list.append(volume_ruiwa)
            dif_ruiwa = (volume_ruiwa - self.mean_ruiwa) ** 2
            dif_log = (volume_log - self.mean_log) ** 2
            
            d3 = v[:, self.DS_dim < w]
            w = w[0:d3.shape[1]]
            d = d3[:, 1 > w]

            try:
                _, similarity, _ = np.linalg.svd(d.T@self.pds)
            except:
                similarity_list = []
                A = d.T@self.pds
                for i in range(A.shape[0]):
                    w1, v1 = np.linalg.eigh(np.dot(A[i,:,:].T, A[i,:,:]))
                    w1 = w1[::-1]; v1 = v1[:,::-1]
                    similarity_pre = np.sqrt(w1)
                    similarity_list.append(similarity_pre)
                similarity = np.array(similarity_list)
            similarity_mean1 = 1 - np.mean(similarity[0:1])
            if similarity_mean1 < 0:
                similarity_mean1 = similarity_mean1 * (-1)
            similarity_mean1_ruiwa = dif_ruiwa * similarity_mean1
            similarity_mean1_log = dif_log * similarity_mean1
            
            similarity_list_k0.append(similarity_mean1)
            similarity_list_k1.append(similarity_list_k0)
        similarity_list_k1 = np.array(similarity_list_k1)
        return similarity_list_k1

    def predict_test(self, x):
        start_idx = 0
        end_idx = x.shape[1] - self.window_length - self.order - self.lag
        score_list = []
        similarity_list = []
        similarity_list_k1 = []
        vol_list = []
        count = 0
        for t in range(start_idx, end_idx):
            similarity_list_k0 = []
            for i in range(x.shape[0]):
                train_H = self._get_hankel(x[i, :], order = self.order,
                                        start = t,
                                        end = t + self.window_length)
                test_H = self._get_hankel(x[i, :], order = self.order,
                                        start = t + self.lag,
                                        end = t + self.window_length + self.lag)
                if i == 0:
                    self.past = train_H
                    self.present = test_H
                else:
                    self.past = np.concatenate([self.past, train_H])
                    self.present = np.concatenate([self.present, test_H])
            sm = SubspaceMethod(r = self.M)
            sm.fit(self.past.T)
            subspace1 = sm.components_
            
            sm = SubspaceMethod(r = self.N)
            sm.fit(self.present.T)
            subspace2 = sm.components_

            subspace1 = subspace1[:, self.rem:self.M]
            subspace2 = subspace2[:, self.rem:self.M]

            G = subspace1 @ subspace1.T + subspace2 @ subspace2.T
            w, v = np.linalg.eigh(G)
            w, v = w[::-1], v[:, ::-1]
            rank = np.linalg.matrix_rank(G)
            w, v = w[:rank], v[:, :rank]

            _, S, _ = np.linalg.svd(subspace1.T@subspace2)
            volume_ruiwa = np.sum(S)
            volume_log = np.sum(np.log(S))
            similarity_list_k0.append(volume_ruiwa)
            similarity_list_k0.append(volume_log)
            vol_list.append(volume_ruiwa)
            dif_ruiwa = (volume_ruiwa - self.mean_ruiwa) ** 2
            dif_log = (volume_log - self.mean_log) ** 2
            
            d3 = v[:, self.DS_dim < w]
            w = w[0:d3.shape[1]]
            d = d3[:, 1 > w]

            try:
                _, similarity, _ = np.linalg.svd(d.T@self.pds)
            except:
                similarity_list = []
                A = d.T@self.pds
                for i in range(A.shape[0]):
                    w1, v1 = np.linalg.eigh(np.dot(A[i,:,:].T, A[i,:,:]))
                    w1 = w1[::-1]; v1 = v1[:,::-1]
                    similarity_pre = np.sqrt(w1)
                    similarity_list.append(similarity_pre)
                similarity = np.array(similarity_list)
            similarity_mean1 = 1 - np.mean(similarity[0:5])
            if similarity_mean1 < 0:
                similarity_mean1 = similarity_mean1 * (-1)
            similarity_mean1_ruiwa = dif_ruiwa * similarity_mean1
            similarity_mean1_log = dif_log * similarity_mean1
            
            similarity_list_k0.append(similarity_mean1)
            similarity_list_k1.append(similarity_list_k0)
        similarity_list_k1 = np.array(similarity_list_k1)
        return similarity_list_k1