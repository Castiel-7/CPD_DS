import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.detection_ds_difcut import DetectionDS
import scipy.io

def parentpath1(path=__file__, f=0):
    return str(os.path.abspath(""))

data_name = 'machine-1-1'

me2 = 'yahoodifcuttest_ruiwa3'
me3 = 'yahoodifcuttest_log3'

data_name = ['yahoo-7.mat', 'yahoo-8.mat', 'yahoo-16.mat', 'yahoo-22.mat', 'yahoo-27.mat', 'yahoo-33.mat', \
                  'yahoo-37.mat', 'yahoo-42.mat', 'yahoo-45.mat', 'yahoo-46.mat', 'yahoo-50.mat', 'yahoo-51.mat', \
                  'yahoo-54.mat', 'yahoo-55.mat', 'yahoo-56.mat']
for data_name_i in data_name:
    mat = scipy.io.loadmat(f'{parentpath1(__file__, f=0)}/yahoo/{data_name_i}') 
    data = np.array(mat)
    feature = np.array(mat['Y']).T
    label = np.array(mat['L'])
    one_list = []
    
    #print(label.shape)
    train_num = int(label.shape[0] * 0.6)
    val_num = int(label.shape[0] * 0.8)
    test_num = label.shape[0]
    #print(train_num, val_num, test_num)
    #print(val_num - train_num)
    train_x = feature[:, 0:train_num]
    train_y = label[0:train_num]
    val_x = feature[:, train_num:val_num]
    val_y = label[train_num:val_num]
    test_x = feature[:, val_num:test_num]
    test_y = label[val_num:test_num]

    for dim_i in [1]:
        for window_length_i in [16]:
            for lag_rate_i in [0.7]:
                for order_i in [8]:
                    for ds_i in [0.001]:
                        for ps_i in [5]:
                            for dif_cut_i in [0.9]:
                                data_name = data_name_i
                                window_length = window_length_i
                                order = order_i
                                lag_rate = lag_rate_i
                                lag = int((window_length+order-1)*(1-lag_rate))
                                M = dim_i
                                N = dim_i
                                DS_dim = ds_i
                                PS_dim = ps_i
                                dif_cut = dif_cut_i
                                print(f'{data_name}_{me2}_w{window_length}_o{order}_l{lag_rate}_d{M}_dsdim{DS_dim}_psdim{PS_dim}_difcut{dif_cut}')
                                model = DetectionDS(window_length=window_length, order=order, lag = lag, M = M, N = N, DS_dim = DS_dim, PS_dim = PS_dim, dif_cut = dif_cut)
                                score1 = model.fit(train_x)
                                score2 = model.predict(test_x)
                                print(score2.shape)
                                cano_type = ["1", "5", "all"]
                                
                                for i in range(3):
                                    new_dir_path4 = f'{me2}/dissim_{data_name}'
                                    new_dir_path6 = f'{me3}/dissim_{data_name}'
                                    os.makedirs(new_dir_path4, exist_ok=True)
                                    os.makedirs(new_dir_path6, exist_ok=True)
                                    np.savetxt(f'{new_dir_path4}/w{window_length}_o{order}_l{lag_rate}_d{M}_dsdim{DS_dim}_psdim{PS_dim}_top{cano_type[i]}_cut{dif_cut}_v.csv', score2[:, 2*i] ,delimiter=',')
                                    np.savetxt(f'{new_dir_path6}/w{window_length}_o{order}_l{lag_rate}_d{M}_dsdim{DS_dim}_psdim{PS_dim}_top{cano_type[i]}_cut{dif_cut}_v.csv', score2[:, 2*i+1] ,delimiter=',')