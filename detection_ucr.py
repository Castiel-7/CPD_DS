import sys
import os
import re
import os.path
import numpy as np
import pandas as pd
#from utils.load_data import load
from utils.detection_ds_size_research import DetectionDS
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.preprocessing import Binarizer
import matplotlib.pyplot as plt

def parentpath1(path=__file__, f=0):
    return str(os.path.abspath(""))

def main():
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #1. detect change point    
    for i in range(1):
        for i in range(1):
            data_list = ["chfdb_chf01_275_1", "chfdb_chf01_275_2", "mitdb__100_180_1", "mitdb__100_180_2", "nprs44", "stdb_308_0_1", "stdb_308_0_2"]
            #data_list = ["chfdb_chf01_275_1"]
            me1 = 'sizeruiwadif_standard'                   #method of calculataing the volume of DS by sum of eigenvalues
            me2 = 'sizelogdif_standard'                     #method of calculataing the volume of DS by logarithmic sum of eigenvalues
            window_list = [64, 128, 256]                    #windows_width
            order_list = [64, 128, 256]                     #the number of windos
            lag_list = [0.3, 0.5, 0.7, 0.9]                 #past and present overlap rate
            M_list = [30]                                   #dimensions of signal subspace
            d_list = [0.001, 0.0001, 0.00001, 0.000001]     #the range of eigenvalues(DS)
            p_list = [30, 50, 70, 90]                       #dimensions of principal subspace(normal subspace generated during training)
            for M_i in M_list:                                
                for DS_i in d_list:                    
                    for data_i in data_list:
                        for window_i in window_list:
                            for order_i in order_list:
                                for lag_i in lag_list:
                                    for PS_i in p_list:
                                        data_name = data_i
                                        window_length = window_i
                                        order = order_i
                                        lag = int((window_length+order-1)*(1-lag_i))
                                        M = M_i
                                        N = M_i
                                        DS_dim = DS_i
                                        PS_dim = PS_i
                                        #print(f'{data_name}, w{window_length}, o{order}, l{lag}, m{M}, n{N}, ds_dim{DS_dim}, ps_dim{PS_dim}')
 
                                        if data_name == "chfdb_chf01_275_1":
                                            train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[0:1000, 1]
                                            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 1]
                                        elif data_name == "chfdb_chf01_275_2":
                                            train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[0:1000, 2]
                                            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 2]
                                        elif data_name == "chfdb_chf13_45590_1":
                                            train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[0:1000, 1]
                                            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 1]
                                        elif data_name == "chfdb_chf13_45590_2":
                                            train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[0:1000, 2]
                                            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 2]
                                        elif data_name == "chfdbchf15_1":
                                            train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[0:1000, 1]
                                            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 1]
                                        elif data_name == "chfdbchf15_2":
                                            train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[0:1000, 2]
                                            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 2]
                                        elif data_name == "mitdb__100_180_1":
                                            train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[0:1000, 1]
                                            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 1]
                                        elif data_name == "mitdb__100_180_2":
                                            train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[0:1000, 2]
                                            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1000:3500, 2]
                                        elif data_name == "nprs44":
                                            train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[12700:15500]
                                            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[15500:22000]
                                        elif data_name == "stdb_308_0_1":
                                            train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[0:1500, 1]
                                            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1500:5000, 1]
                                        elif data_name == "stdb_308_0_2":
                                            train = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[0:1500, 2]
                                            test = np.loadtxt(f'{parentpath1(__file__, f=0)}/data_s2/{data_name}.txt')[1500:5000, 2]
                                        print(f'{data_name}_w{window_length}_o{order}_l{lag_i}_d{M}_dsdim{DS_dim}_psdim{PS_dim}')
                                        model = DetectionDS(window_length=window_length, order=order, lag = lag, M = M, N = N, DS_dim = DS_dim, PS_dim = PS_dim)
                                        score1 = model.fit(train)      #train
                                        score2 = model.predict(test)   #test
                                        print(score2.shape)
                                        #print(train.shape, test.shape, score2.shape)
                                        #print(int((score2.shape[1]-1)/2))
                                        #print(type(score2))
                                        cano_type = ["1", "5", "all"]      #number of canonical angles
                                        for i in range(3):
                                            new_dir_path4 = f'{me1}/dissim_{data_name}'
                                            new_dir_path6 = f'{me2}/dissim_{data_name}'
                                            os.makedirs(new_dir_path4, exist_ok=True)
                                            os.makedirs(new_dir_path6, exist_ok=True)
                                            np.savetxt(f'{new_dir_path4}/w{window_length}_o{order}_l{lag_i}_d{M}_dsdim{DS_dim}_psdim{PS_dim}_top{cano_type[i]}_v.csv', score2[:, 2*i] ,delimiter=',')
                                            np.savetxt(f'{new_dir_path6}/w{window_length}_o{order}_l{lag_i}_d{M}_dsdim{DS_dim}_psdim{PS_dim}_top{cano_type[i]}_v.csv', score2[:, 2*i+1] ,delimiter=',')

if __name__=='__main__':
    main()