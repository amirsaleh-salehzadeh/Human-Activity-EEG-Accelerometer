# -*- coding: utf-8 -*-
"""
Created on Fri May  4 21:34:01 2018

@author: 大茄茄
"""

#对原始数据进行四阶巴特沃斯滤波

from scipy.signal import butter, lfilter  
import pandas as pd
import os 
from sklearn.externals.joblib import Parallel, delayed


SampFreq = 256
ChannelNum = 22

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):  
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)  
    y = lfilter(b, a, data)  
    return y  
    
  
def butter_bandpass(lowcut, highcut, fs, order=5):  
    nyq = 0.5 * fs  
    low = lowcut / nyq  
    high = highcut / nyq  
    b, a = butter(order, [low, high], btype='bandpass')  
    return b, a 

def filterX_onset(path, savepath):
    for per_file in os.listdir(path):
        filename, suffix = os.path.splitext(per_file)
        filepath = os.path.join(path, per_file)
        data = pd.read_csv(filepath)
        data = data.iloc[:, 1:]
        filterData = []
        for per_channel in range(ChannelNum):
            X = data.iloc[per_channel, :]
            filterX = butter_bandpass_filter(X, 0.01, 32, SampFreq, 4)
            filterData.append(filterX)
        filterData = pd.DataFrame(filterData)
        filterData.to_csv(savepath + '\\{}.csv'.format(filename))
        
        
        
def filterX_interItcal(file, path, savepath):
    filename, suffix = os.path.splitext(file)
    if suffix == '.csv':
        data = pd.read_csv(os.path.join(path, file))
        data = data.iloc[:, 1:]
        filterData = []
        for per_channel in range(ChannelNum):
            X = data.iloc[per_channel, :]
            filterX = butter_bandpass_filter(X, 0.01, 32, SampFreq, 4)
            filterData.append(filterX)
        filterData = pd.DataFrame(filterData)
        filterData.to_csv(savepath + '\\{}.csv'.format(filename))
    
            
def multiprocess(path, savepath):
    Parallel(n_jobs=1)(delayed(filterX_interItcal)(i, path, savepath) for i in os.listdir(path))
    
        
        
    
if __name__ == '__main__':
    pass