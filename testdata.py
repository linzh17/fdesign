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
    matfn='/home/lin/Desktop/课程作业/毕业论文/bilibili_data/'+file
    
    data=sio.loadmat(matfn)

    '''print(data.keys())
    datakey = data.keys()
    print(type(datakey))

    print(list(data)[3])
    print(data[file][0])
    datanum = data[file][0][0]
    print(int(data[file][0][0]))
    time = pd.to_datetime(datanum-719529,unit='D').date()
    print(type(time))
    print(time)'''

    #Data = [[row.flat[0] for row in line] for line in data[file[:-4]]]
    df_train = pd.DataFrame(data[file[:-4]])
    print(df_train)
    break
    '''
    for i in range(0,df_train.shape[0]):
        df_train[0][i] = pd.to_datetime(df_train[0][i]-719529,unit='D').date()
    df_train = df_train.drop_duplicates(0,'last')
    df_train = df_train.reset_index(drop=True)
    #print(df_train)
    df_train = pd.Series(df_train[1].tolist(),index = df_train[0].tolist())
    df_train.plot()
    '''
    #print(df_train[1].diff()
    #plt.plot(df_train[0][1:],df_train[1].diff()[1:])

    

plt.show()