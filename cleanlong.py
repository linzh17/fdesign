import scipy.io as sio 
 
from pylab import *
 
import numpy as np
import pandas as pd
from pandas import Series
import datetime
import matplotlib.pyplot as plt
import os
import glob
from shutil import copyfile
from datetime import datetime


#处理长序列数据
os.chdir('/home/lin/Desktop/课程作业/毕业论文/cleanData/longSeries/addNumDay')
files = glob.glob('*.csv')
print(len(files))
print(files)
i = 1

timeList = pd.date_range(start='2016-08-01',end='2016-09-01')
print(timeList)

for file in files:
    print(file)
    print(i)
    i = i + 1
    csvfile ='/home/lin/Desktop/课程作业/毕业论文/cleanData/longSeries/addNumDay/'+file
    

    df_csv = pd.read_csv(csvfile)


    #print(df_csv)

    

    if df_csv.shape[0] > 540:


        df_csv = df_csv = pd.Series(df_csv['addnum'].tolist(),index  = df_csv['time'].tolist())
        df_csv = df_csv['2016-08-01':'2016-09-01']
        print(len(df_csv))
        print(len(timeList))
        
        if len(df_csv) == len(timeList):
            print('True')
            path = '/home/lin/Desktop/课程作业/毕业论文/cleanData/longSeries/cleanAddNumDay/'+file  
            print(df_csv)
            df_csv.to_csv(path)

            df_csv.plot()

        
    else:
        continue

    break

    #处理出每个月记录的播放量 存到文件playNumMonth
    '''for i in range(0,df_csv.shape[0]):
        df_csv['0'][i] = df_csv['0'][i][:-3]

    #print(df_csv)
    df_csv = df_csv.drop_duplicates('0','last')
    df_csv = df_csv.reset_index(drop=True)
    print(df_csv)
    path = '/home/lin/Desktop/课程作业/毕业论文/cleanData/longSeries/playNumMonth/'+file   
    df_csv.to_csv(path)'''
    
    
    #print(df_train[1].diff()
    #绘制每天的播放量
    #plt.plot(df_csv['0'],df_csv['1'])
    #print(type(df_csv))
    #print(df_csv['0'])

    #绘制每天增加的播放量
    '''df_csv = pd.Series(df_csv['1'].diff()[1:].tolist(),index = df_csv['0'][1:].tolist())
  
    #break
    #plt.plot(df_csv['0'][1:],df_csv['1'].diff()[1:])
    df_csv.plot()
    #break'''

    #生成每日增加播放量的数据到addNumDay文件夹
    '''df_csv = pd.DataFrame({'time':df_csv['0'][1:],'addnum':df_csv['1'].diff()[1:]})
    #print(df_csv)
    path = '/home/lin/Desktop/课程作业/毕业论文/cleanData/longSeries/addNumDay/'+file
    df_csv.to_csv(path)
    #break'''

    #生成每月增加播放量的数据到addNumMonth文件夹
    '''path = '/home/lin/Desktop/课程作业/毕业论文/cleanData/longSeries/playNumMonth/'+file
    df_csv = pd.read_csv(path)
    df_csv = pd.DataFrame({'time':df_csv['0'][1:],'addnum':df_csv['1'].diff()[1:]})
    print(df_csv)
    path = '/home/lin/Desktop/课程作业/毕业论文/cleanData/longSeries/addNumMonth/'+file
    df_csv.to_csv(path)
    #break'''

    #绘制每月增加播放量
    '''path = '/home/lin/Desktop/课程作业/毕业论文/cleanData/longSeries/addNumMonth/'+file
    df_csv = pd.read_csv(path)
    print(df_csv)
    df_csv = pd.Series(df_csv['addnum'].tolist(),index  = df_csv['time'].tolist())
    df_csv.plot()
    #break'''


plt.show()

