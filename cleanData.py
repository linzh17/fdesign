import scipy.io as sio 
 
from pylab import *
 
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os
import glob
os.chdir('bilibili_data/')
files = glob.glob('*.mat')
print(len(files))
#print(files)
i = 1
for file in files:
    #print(file)
    print(i)
    i = i + 1

    '''
    matfn='/home/lin/Desktop/课程作业/毕业论文/bilibili_data/'+file
    
    data=sio.loadmat(matfn)

    df_train = pd.DataFrame(data[file[:-4]])
    for i in range(0,df_train.shape[0]):
        df_train[0][i] = pd.to_datetime(df_train[0][i]-719529,unit='D').date()
    df_train = df_train.drop_duplicates(0,'last')
    df_train = df_train.reset_index(drop=True)
    print(df_train)
    csv_path = '/home/lin/Desktop/课程作业/毕业论文/cleanData/playNum/'+file[:-4]+'.csv'
    df_train.to_csv(csv_path)
    '''
    #break
print(i)
