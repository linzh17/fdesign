import scipy.io as sio 
 
from pylab import *
 
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os
import glob
from shutil import copyfile

os.chdir('/home/lin/Desktop/课程作业/毕业论文/cleanData/playNum/')
files = glob.glob('*.csv')
print(len(files))
print(files)
i = 1
for file in files:
    #print(file)
    print(i)
    i = i + 1
    csvfile ='/home/lin/Desktop/课程作业/毕业论文/cleanData/playNum/'+file
    longfile = '/home/lin/Desktop/课程作业/毕业论文/cleanData/longSeries/'+file
    shortfile = '/home/lin/Desktop/课程作业/毕业论文/cleanData/shortSeries/'+file
    

    df_csv = pd.read_csv(csvfile)
    #print(df_csv)
    if df_csv.shape[0] > 500:
        print('long')
        copyfile(csvfile,longfile)
    else:
        copyfile(csvfile,shortfile)
   
    


