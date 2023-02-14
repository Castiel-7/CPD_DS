import numpy as np
import math
import os
import matplotlib.pyplot as plt

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

class DetectionDS(object):
    def __init__(self, window_length=128, order=64, lag = 64, M = 5, N = 10, DS_dim = 5, PS_dim = 30, rem = 5, data_name = 0, info = 0, data_type = 0):
        self.window_length = window_length
        self.order = order
        self.lag = lag
        self.M = M
        self.N = N
        self.DS_dim = DS_dim
        self.PS_dim = PS_dim
        self.ds7 = np.zeros(10)
        self.rem = rem
        self.data_name = data_name
        self.info = info
        self.data_type = data_type
        
    def fit(self, x):
        start_idx = 0
        end_idx = len(x) - self.window_length - self.order - self.lag
        #print("K_num:", self.K_num)
        ds_list = []
        pds_list = []
        score_list = []
        vol_ruiwa_train = []
        vol_log_train = []
        signal_all = []
        similarity = []
        count = 0
        #print(start_idx)
        #print(end_idx)
        for t in range(start_idx, end_idx):
            
            self.past = self._get_hankel(x, order = self.order,
                                       start = t,
                                       end = t + self.window_length)
            self.present = self._get_hankel(x, order = self.order,
                                     start = t + self.lag,
                                     end = t + self.window_length + self.lag)
            
            sm = SubspaceMethod(r = self.M)
            sm.fit(self.past.T)
            subspace1 = sm.components_
            s1_w = sm.coef_

            sm = SubspaceMethod(r = self.N)
            sm.fit(self.present.T)
            subspace2 = sm.components_
            s2_w = sm.coef_

            subspace1 = subspace1[:, self.rem:self.M]
            subspace2 = subspace2[:, self.rem:self.M]
            #print(s1_w, s2_w)
            _, S, _ = np.linalg.svd(subspace1.T@subspace2)
            #volume = np.sum(np.log(S))
            volume_ruiwa = np.sum(S)
            volume_log = np.sum(np.log(S))
            #print(S, volume_ruiwa, volume_log)
            vol_ruiwa_train.append(volume_ruiwa)
            vol_log_train.append(volume_log)

            G = subspace1 @ subspace1.T + subspace2 @ subspace2.T
            w, v = np.linalg.eigh(G)
            w, v = w[::-1], v[:, ::-1]
            rank = np.linalg.matrix_rank(G)
            w, v = w[:rank], v[:, :rank]

            #d = v[:, int(v.shape[1]/2):int(v.shape[1]/2)+self.DS_dim]
            d1 = v[:, self.DS_dim < w]
            #print("1", 0.0001 < w, d1.shape)
            w = w[0:d1.shape[1]]
            d2 = d1[:, 1 > w]

            X_matrix = np.zeros((self.present.shape[0], self.present.shape[1]))
            X_traject = np.zeros((self.present.shape[0], self.present.shape[1]))
            for j in range(subspace1.shape[1]):
                U_i = subspace1[:, j][:, np.newaxis]
                V_i = (self.past.T @ subspace1[:, j] / np.sqrt(s1_w[j]))[:, np.newaxis]
                X_matrix += np.sqrt(s1_w[j]) * U_i @ V_i.T
                
            for k in range((self.present.shape[0] - 1) + (self.present.shape[1] - 1) + 1):
                index_list = []
                diag_sum = 0
                for l in range(self.present.shape[0]):
                    for m in range(self.present.shape[1]):
                        if l + m == k:
                            diag_sum += X_matrix[l, m]
                            index_list.append(l)
                diag_ave = diag_sum / len(index_list)
                for index in index_list:
                    X_traject[index, k - index] = diag_ave
            signal1 = X_traject[0, :]
            signal2 = X_traject[:, self.present.shape[1] - 1][1:]
            signal_past = np.concatenate([signal1, signal2], 0)

            X_matrix = np.zeros((self.present.shape[0], self.present.shape[1]))
            X_traject = np.zeros((self.present.shape[0], self.present.shape[1]))
            for j in range(subspace2.shape[1]):
                U_i = subspace2[:, j][:, np.newaxis]
                V_i = (self.present.T @ subspace2[:, j] / np.sqrt(s2_w[j]))[:, np.newaxis]
                X_matrix += np.sqrt(s2_w[j]) * U_i @ V_i.T
                
            for k in range((self.present.shape[0] - 1) + (self.present.shape[1] - 1) + 1):
                index_list = []
                diag_sum = 0
                for l in range(self.present.shape[0]):
                    for m in range(self.present.shape[1]):
                        if l + m == k:
                            diag_sum += X_matrix[l, m]
                            index_list.append(l)
                diag_ave = diag_sum / len(index_list)
                for index in index_list:
                    X_traject[index, k - index] = diag_ave
            signal1 = X_traject[0, :]
            signal2 = X_traject[:, self.present.shape[1] - 1][1:]
            signal_present = np.concatenate([signal1, signal2], 0)


            '''
            X_matrix = np.zeros((self.present.shape[0], self.present.shape[1]))
            X_traject = np.zeros((self.present.shape[0], self.present.shape[1]))
            for j in range(d2.shape[1]):
                U_i = d2[:, j][:, np.newaxis]
                V_i = (self.present.T @ d2[:, j] / np.sqrt(w[j]))[:, np.newaxis]
                X_matrix += np.sqrt(w[j]) * U_i @ V_i.T
                
            for k in range((self.present.shape[0] - 1) + (self.present.shape[1] - 1) + 1):
                index_list = []
                diag_sum = 0
                for l in range(self.present.shape[0]):
                    for m in range(self.present.shape[1]):
                        if l + m == k:
                            diag_sum += X_matrix[l, m]
                            index_list.append(l)
                diag_ave = diag_sum / len(index_list)
                for index in index_list:
                    X_traject[index, k - index] = diag_ave
            signal1 = X_traject[0, :]
            signal2 = X_traject[:, self.present.shape[1] - 1][1:]
            signal = np.concatenate([signal1, signal2], 0)
            signal_all.append(signal)
            '''
            new_dir_path0 = f'data_{self.data_name}/{self.data_name}_{self.info}_{self.data_type}_{self.window_length}_{self.order}_img'
            new_dir_path1 = f'data_{self.data_name}/{self.data_name}_{self.info}_{self.data_type}_{self.window_length}_{self.order}_signal'
            os.makedirs(new_dir_path0, exist_ok=True)
            '''
            os.makedirs(new_dir_path1, exist_ok=True)
            plt.plot(signal)
            plt.savefig(f'{new_dir_path0}/{self.data_name}_{self.info}_{self.data_type}_{self.window_length}_{self.order}_{t}_img.png')
            plt.close()
            '''
            plt.plot(signal_past)
            plt.savefig(f'{new_dir_path0}/{self.data_name}_{self.info}_{self.data_type}_{self.window_length}_{self.order}_{t}_past_img.png')
            plt.close()

            plt.plot(signal_present)
            plt.savefig(f'{new_dir_path0}/{self.data_name}_{self.info}_{self.data_type}_{self.window_length}_{self.order}_{t}_present_img.png')
            plt.close()

            #np.savetxt(f'{new_dir_path1}/{self.data_name}_{self.info}_{self.data_type}_{self.window_length}_{self.order}_{t}_signal.csv', signal, delimiter=',')
            #this!
            #np.savetxt(f'{new_dir_path1}/{self.data_name}_{self.info}_{self.data_type}_{self.window_length}_{self.order}_{t}_signal_past.csv', signal_past, delimiter=',')
            #np.savetxt(f'{new_dir_path1}/{self.data_name}_{self.info}_{self.data_type}_{self.window_length}_{self.order}_{t}_signal_present.csv', signal_present, delimiter=',')

        #new_dir_path2 = f'data_{self.data_name}/{self.data_name}_{self.info}_{self.data_type}_{self.window_length}_{self.order}_signallist'
        #os.makedirs(new_dir_path2, exist_ok=True)
        #np.savetxt(f'{new_dir_path2}/{self.data_name}_{self.info}_{self.data_type}_{self.window_length}_{self.order}_signallist.csv', signal_all, delimiter=',')

        return 1
            
            
    def _get_hankel(self, x, order, start, end):
        return np.array([x[start+i:end+i] for i in range(order)]).T
