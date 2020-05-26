import numpy as np

import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler


trainData = np.loadtxt('/home/lin/Desktop/课程作业/毕业论文/cleanData/longSeries/dataNumpy/dataNumpy.csv')
print(trainData.shape[1])
trainData = trainData.reshape(2009,21,1)
print(trainData)


seed = 0
np.random.seed(seed)

np.random.shuffle(trainData)


trainData = TimeSeriesScalerMeanVariance(mu=0.0, std=1.0).fit_transform(trainData[100:201])
#trainData = trainData[:100]

clunum=3#输入聚类的数目

'''

sz = trainData.shape[1]
print("Euclidean k-means")
km = TimeSeriesKMeans(n_clusters=clunum, verbose=True,random_state=seed)#聚类，输入聚类的数目。
y_pred = km.fit_predict(trainData)

print('各时间序列的所属类别=',y_pred)#输出各个时间序列对应的类
 
plt.figure(figsize=(16,9))
for yi in range(clunum):
    plt.subplot(clunum, 1, 1 + yi)
    for xx in trainData[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.title("Cluster %d" % (yi + 1))
 
plt.tight_layout()
plt.show()'''



# DBA-k-means
print("DBA k-means")
dba_km = TimeSeriesKMeans(n_clusters=clunum,
                          n_init=2,
                          metric="dtw",
                          verbose=True,
                          max_iter_barycenter=10,
                          random_state=seed)
y_pred = dba_km.fit_predict(trainData)

print('各时间序列的所属类别=',y_pred)#输出各个时间序列对应的类

plt.figure(figsize=(16,9))
for yi in range(clunum):
    plt.subplot(clunum, 1, 1 + yi)
    for xx in trainData[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
    plt.title("Cluster %d" % (yi + 1))
plt.show()
print(dba_km.predict(trainData[:1]))


'''
# Soft-DTW-k-means
print("Soft-DTW k-means")
sdtw_km = TimeSeriesKMeans(n_clusters=clunum,metric="softdtw",metric_params={"gamma": .01},verbose=True,random_state=seed)
y_pred = sdtw_km.fit_predict(trainData)
 
plt.figure(figsize=(16,9))
for yi in range(clunum):
    plt.subplot(clunum, 1, 1 + yi)
    for xx in trainData[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(sdtw_km.cluster_centers_[yi].ravel(), "r-")
    plt.title("Cluster %d" % (yi + 1))
 
plt.tight_layout()
plt.show()
'''
