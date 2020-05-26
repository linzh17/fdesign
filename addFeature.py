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

# 用长序列视频数据 
os.chdir('/home/lin/Desktop/课程作业/毕业论文/cleanData/longSeries/')
files = glob.glob('*.csv')
print(len(files))
print(files)

num = 0
fileList = []
for file in files:

    
    num = num+1
    print(num)
#添加额外特征  
    csvfile ='/home/lin/Desktop/课程作业/毕业论文/cleanData/longSeries/'+file
    df_csv = pd.read_csv(csvfile)

    df_csv = pd.DataFrame({'time':df_csv['0'][1:],'addnum':df_csv['1'].diff()[1:],'comment':df_csv['2'].diff()[1:],'like':df_csv['3'].diff()[1:],'share':df_csv['4'].diff()[1:]})

    #print(df_csv)

    #break

    #统一时间序列长度 2016-07-01~2017-12-01 补充缺失值 并写入文件夹
    dates = pd.date_range('2016-07-01','2017-12-01')
    df_date = pd.DataFrame({'playnum':np.NaN,'comment':np.NaN,'like':np.NaN,'share':np.NaN},index = dates)
    #print(df_date['2016-07-01':'2017-12-01'])
    #print(len(df_date))
    #print(df_date.loc['2016-07-01','addnum'])
    df_csv = df_csv.set_index('time')

    
    
    if('2016-07-01' not in df_csv.index.tolist() and '2017-12-01' not in df_csv.index.tolist() ):
        print("不在区间范围内")
        continue
    df_csv = df_csv['2016-07-01':'2017-12-01']
    
    #print(df_csv)
    for i in df_csv.index.tolist():
        df_date.loc[i,'playnum'] = df_csv.loc[i,'addnum']
        df_date.loc[i,'comment'] = df_csv.loc[i,'comment']
        df_date.loc[i,'like'] = df_csv.loc[i,'like']
        df_date.loc[i,'share'] = df_csv.loc[i,'share']
    
    numList = df_date['playnum'].tolist()
    commentList = df_date['comment'].tolist()
    likeList = df_date['like'].tolist()
    shareList = df_date['share'].tolist()
    
    flag = 0
    for i in range(0,len(numList)):
       
        if math.isnan(numList[i]):
            #print(i)
            flag = flag + 1
        elif numList[i] != nan:
            temp = numList[i]
            for j in range(0,flag+1):
                numList[i-j] = float(int(temp/(flag+1)))
            flag = 0

    flag = 0
    for i in range(0,len(commentList)):
       
        if math.isnan(commentList[i]):
            #print(i)
            flag = flag + 1
        elif commentList[i] != nan:
            temp = commentList[i]
            for j in range(0,flag+1):
                commentList[i-j] = float(int(temp/(flag+1)))
            flag = 0

    flag = 0
    for i in range(0,len(likeList)):
       
        if math.isnan(likeList[i]):
            #print(i)
            flag = flag + 1
        elif likeList[i] != nan:
            temp = likeList[i]
            for j in range(0,flag+1):
                likeList[i-j] = float(int(temp/(flag+1)))
            flag = 0
    
    flag = 0
    for i in range(0,len(shareList)):
       
        if math.isnan(shareList[i]):
            #print(i)
            flag = flag + 1
        elif shareList[i] != nan:
            temp = shareList[i]
            for j in range(0,flag+1):
                shareList[i-j] = float(int(temp/(flag+1)))
            flag = 0

    df_csv = pd.DataFrame({"time":dates,'playnum':numList,'comment':commentList,'like':likeList,'share':shareList})
    path = '/home/lin/Desktop/课程作业/毕业论文/cleanData/longSeries/addFeature/'+file
    df_csv.to_csv(path)
    #break

print(num)
