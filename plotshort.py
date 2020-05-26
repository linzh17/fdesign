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
import math
from sklearn.preprocessing import MinMaxScaler


#处理短序列数据
os.chdir('/home/lin/Desktop/课程作业/毕业论文/cleanData/shortSeries/')
files = glob.glob('*.csv')
#print(len(files))
#print(files)
i = 1
for file in files:
    #暂不执行以下操作
    break
    
    print(file)
    print(i)
    i = i + 1
    csvfile ='/home/lin/Desktop/课程作业/毕业论文/cleanData/shortSeries/'+file
    

    df_csv = pd.read_csv(csvfile)


    print(df_csv)

    #break
    
    #处理出每个月记录的播放量 存到文件playNumMonth
    '''for i in range(0,df_csv.shape[0]):
        df_csv['0'][i] = df_csv['0'][i][:-3]

    #print(df_csv)
    df_csv = df_csv.drop_duplicates('0','last')
    df_csv = df_csv.reset_index(drop=True)
    #print(df_csv)
    df_csv = pd.DataFrame({'time':df_csv['0'],'playnum':df_csv['1']})
    #print(df_csv)
    
    path = '/home/lin/Desktop/课程作业/毕业论文/cleanData/shortSeries/playNumMonth/'+file   
    df_csv.to_csv(path)
    #break'''
    
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
    print(df_csv)
    path = '/home/lin/Desktop/课程作业/毕业论文/cleanData/shortSeries/addNumDay/'+file
    df_csv.to_csv(path)
    #break'''
    
    '''
    #处理2016-07-01~2017-01-01播放量缺失值
    dates = pd.date_range('2016-07-01','2017-01-01')
    df_date = pd.DataFrame({'addnum':np.NaN},index = dates)
    print(df_date['2016-07-01':'2017-01-01'])
    print(len(df_date))
    #print(df_date.loc['2016-07-01','addnum'])
    path = '/home/lin/Desktop/课程作业/毕业论文/cleanData/shortSeries/addNumDay/'+file
    df_csv = pd.read_csv(path)
    df_csv = df_csv.set_index('time')

    
    
    if('2016-07-01' not in df_csv.index.tolist() and '2017-01-01' not in df_csv.index.tolist() ):
        print("不在区间范围内")
        continue
    df_csv = df_csv['2016-07-01':'2017-01-01']
    
    print(df_csv)
    for i in df_csv.index.tolist():
        df_date.loc[i,'addnum'] = df_csv.loc[i,'addnum']
    
    numlist = df_date['addnum'].tolist()
    print(numlist)
    flag = 0
    for i in range(0,len(numlist)):
       
        if math.isnan(numlist[i]):
            #print(i)
            flag = flag + 1
        elif numlist[i] != nan:
            temp = numlist[i]
            for j in range(0,flag+1):
                numlist[i-j] = float(int(temp/(flag+1)))
            flag = 0
    
    print(numlist)

    df_csv = pd.DataFrame({"time":dates,"playnum":numlist})
    path = '/home/lin/Desktop/课程作业/毕业论文/cleanData/shortSeries/playNumDay/'+file
    df_csv.to_csv(path)
    #break
    '''

    



    #生成每月增加播放量的数据到addNumMonth文件夹
    '''path = '/home/lin/Desktop/课程作业/毕业论文/cleanData/shortSeries/playNumMonth/'+file
    df_csv = pd.read_csv(path)
    df_csv = pd.DataFrame({'time':df_csv['time'][1:],'addnum':df_csv['playnum'].diff()[1:]})
    print(df_csv)
    
    path = '/home/lin/Desktop/课程作业/毕业论文/cleanData/shortSeries/addNumMonth/'+file
    df_csv.to_csv(path)
    #break'''

    #绘制每月增加播放量
    ''' path = '/home/lin/Desktop/课程作业/毕业论文/cleanData/shortSeries/addNumMonth/'+file
    df_csv = pd.read_csv(path)
    for i in range(0,df_csv['time'].shape[0]):
        df_csv['time'][i] = pd.to_datetime(df_csv['time'][i])
    print(df_csv)
    if df_csv.shape[0] == 0:
        continue
    df_csv = pd.Series(df_csv['addnum'].tolist(),index  = df_csv['time'].tolist())
    df_csv.plot()'''
    

#plt.xticks(pd.date_range('2017-05','2018-9',freq='1m'))
#plt.show()


#绘制处理好的部分短数据 20160701-20170101
'''os.chdir('/home/lin/Desktop/课程作业/毕业论文/cleanData/shortSeries/playNumDay/20160701-20170101/')
files = glob.glob('*.csv')
print(len(files))
print(files)
i = 0
for file in files:
    path = '/home/lin/Desktop/课程作业/毕业论文/cleanData/shortSeries/playNumDay/20160701-20170101/'+file
    df_csv = pd.read_csv(path)
    if len(df_csv[df_csv['playnum']<0]) > 0:
        continue
    
    i = i+1
    df_csv = pd.Series(df_csv['playnum'].tolist(),index  = df_csv['time'].tolist())
    df_csv.plot()
    
    #break
    
print(i)
plt.title("The amount of playing from 2016-07-01~2017-01-01  ")
plt.xlabel("Date  (2016-07-01~2017-01-01)")
plt.ylabel("Amount of playing")
plt.show()'''



#生成日播放量 numpy 的生成对象 训练数据 （2016-01-01-2017-01-01）
numpyList = []
fileList = []

os.chdir('/home/lin/Desktop/课程作业/毕业论文/cleanData/shortSeries/playNumDay/20160701-20170101/')
files = glob.glob('*.csv')
print(len(files))
print(files)
i = 0
for file in files:

    
    path = '/home/lin/Desktop/课程作业/毕业论文/cleanData/shortSeries/playNumDay/20160701-20170101/'+file
    df_csv = pd.read_csv(path)
    if len(df_csv[df_csv['playnum']<0]) > 0:
        continue
    i = i+1    
    #print(len(df_csv))
    print(i)
        
    numpyList.append(df_csv['playnum'].tolist())
    fileList.append(file)

    #print(numpyList)
        
    #break

#生成长序列 日播放量 numpy 对象

dataNumpy = np.array(numpyList)
print(dataNumpy)
print(dataNumpy.shape)
#归一化
scaler = MinMaxScaler()
dataNumpy = scaler.fit_transform(dataNumpy)
print(dataNumpy)
fileList  = pd.DataFrame({'name':fileList})
path = '/home/lin/Desktop/课程作业/毕业论文/cleanData/shortSeries/fileList/dayShortFileList.csv'
fileList.to_csv(path)
np.savetxt('/home/lin/Desktop/课程作业/毕业论文/cleanData/shortSeries/dataNumpy/dayShortNumpy.csv',dataNumpy)
