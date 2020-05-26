from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold,train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import numpy as np
import pandas as pd
from pandas import Series
import datetime
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import os
import glob
import time
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import random

scaler = MinMaxScaler()
lists = []
fileLists = []
os.chdir('/home/lin/Desktop/课程作业/毕业论文/cleanData/longSeries/playNumDay/')
files = glob.glob('*.csv')
#print(files)
num = 0
for file in files:
    
    path = '/home/lin/Desktop/课程作业/毕业论文/cleanData/longSeries/addFeature/'+file
    df_data  = pd.read_csv(path)
    #要求视屏平均播放量大于5
    if df_data['playnum'].mean() < 5:
        continue

    num = num +1

    #随机时间跨度
    '''
    end = 518-15
    start = random.randint(0,end)
    end = start+15
    df_data = df_data.loc[start:200,:]
    '''

    #相同时间跨度
    df_data = df_data.loc[0:15,:]
    #修改数据类型
    df_data['playnum'] = df_data['playnum'].astype(int) 
    df_data['comment'] = df_data['comment'].astype(int) 
    df_data['like'] = df_data['like'].astype(int) 
    df_data['share'] = df_data['share'].astype(int) 
    
    
    #修改播放量异常值 将负值改成数据中的众数
    if len(df_data[df_data['playnum']<0]) > 0:
         #print(file)
         indexs = df_data[df_data['playnum']<0].index
         for index in indexs:
             df_data.loc[index,'playnum'] = int(df_data['playnum'].mode()[0])

    #修改评论异常值
    if len(df_data[df_data['comment']<0]) > 0:
         #print(file)
         indexs = df_data[df_data['comment']<0].index
         for index in indexs:
             df_data.loc[index,'comment'] = int(df_data['comment'].mode()[0])

    #修改收藏异常值
    if len(df_data[df_data['like']<0]) > 0:
         #print(file)
         indexs = df_data[df_data['like']<0].index
         for index in indexs:
             df_data.loc[index,'like'] = int(df_data['like'].mode()[0])
    
    #修改分享异常值
    if len(df_data[df_data['share']<0]) > 0:
         #print(file)
         indexs = df_data[df_data['share']<0].index
         for index in indexs:
             df_data.loc[index,'share'] = int(df_data['share'].mode()[0])

    #特征构建    
    #添加视频名称
    df_data["av_id"] = num
    fileLists.append(file)

    #添加一周时间，周末
    series = pd.Series(index=df_data['time'].tolist())
    pd_series = pd.DataFrame(series)
    pd_series .index= pd.to_datetime(pd_series.index)
    df_data["weekday"] = pd_series.index.weekday
    df_data["weekend"] = df_data.weekday.isin([5,6])*1

    #添加偏移量 偏移一周
    for i in range(1,8):
       df_data["shift_{}".format(i)] =df_data.playnum.shift(i)
       df_data["shift_comment{}".format(i)] =df_data.comment.shift(i)
       df_data["shift_like{}".format(i)] =df_data.like.shift(i)
       df_data["shift_share{}".format(i)] =df_data.share.shift(i)

    # 除去偏移的NaN 值
    df_data = df_data.dropna()

    #修改数据类型
    #归一化
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    for i in range(1,8):
        df_data["shift_{}".format(i)] = df_data["shift_{}".format(i)].astype(int)
        '''
        df_data["shift_{}".format(i)] = df_data[["shift_{}".format(i)]].apply(max_min_scaler).astype(float)
        df_data["shift_{}".format(i)] = df_data["shift_{}".format(i)].replace([np.inf, -np.inf], df_data["shift_{}".format(i)].mean())
        df_data["shift_{}".format(i)] = df_data["shift_{}".format(i)].fillna(df_data["shift_{}".format(i)].mean())
        pass
    '''
        
    #print(df_data[df_data.isnull().values==True])

    
    df_data.time = pd.to_datetime(df_data.time)
    df_data.time = df_data.time.map(dt.datetime.toordinal)
    print(df_data)
    lists.append(df_data)
    
    '''
    if num>0:
        break
    '''
    
    
#print(fileLists[29-1])

#整合数据集
data_new  = pd.concat(lists)
data_new = data_new.sort_values(by="time"  )
data_new = data_new.loc[ : , ~data_new.columns.str.contains("^Unnamed")]

print(data_new)

#onehot编码
df_week = pd.get_dummies(data_new.weekday)
df_week.columns = ['mon','tues','wed','thur','fri','sat','sun']
data_new = pd.concat([data_new,df_week],axis = 1)
'''
df_year = pd.get_dummies( data_new.year)
df_year.columns = ['2016','2017']
data_new = pd.concat([data_new,df_year],axis = 1)

df_month = pd.get_dummies(data_new.month)
df_month.columns=['jan','feb','mar','april',',may','jun','july','aug','sep','oct','nov','dec']
data_new = pd.concat([data_new,df_month],axis = 1)

df_day = pd.get_dummies(data_new.day)
df_day.columns = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
data_new = pd.concat([data_new,df_day],axis=1)
'''


data_x = data_new.drop(['playnum'], axis=1)

#除去额外特征
data_x = data_x.drop(['comment'], axis=1)
data_x = data_x.drop(['like'], axis=1)
data_x = data_x.drop(['share'], axis=1)

#除去时间参数
data_x = data_x.drop(['time'], axis=1)
data_x = data_x.drop(['weekend'],axis = 1)
data_x = data_x.drop(['weekday'],axis = 1)

data_y = data_new['playnum']

train_x,test_x,train_y,test_y=train_test_split(data_x,data_y,test_size=0.2)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
print(data_new.info())


#建立模型
lr_model=LinearRegression()
tree_model=DecisionTreeRegressor()
gbdt_model=GradientBoostingRegressor()
rfr_model=RandomForestRegressor()
xgb_model = XGBRegressor()

models=[lr_model,tree_model,rfr_model,gbdt_model,xgb_model]
model_names=['lr_model','tree_model','rfr_model','gbdt_model','xgb_model']

scores=[]
for model,model_name in zip(models,model_names):
    t5=time.time()
    score=cross_val_score(model,train_x,train_y,cv=StratifiedKFold(5))
    t6=time.time()
    print('{}运行时间：'.format(model_name),(t6-t5))
    scores.append(score)
score_matrix=pd.DataFrame(scores,index=model_names)
score_matrix['mean']=score_matrix.mean(axis=1)
score_matrix['std']=score_matrix.std(axis=1)
print('{:*^30}'.format('各模型分数矩阵'))
print(score_matrix)

'''
#model = tree_model
#model = gbdt_model
#model = xgb_model
#model =rfr_model
#model = lr_model

model.fit(train_x,train_y)
y_pred=model.predict(test_x)
mse=mean_squared_error(test_y.tolist(),y_pred)
r2=r2_score(test_y.tolist(),y_pred)

print(mse)
print(r2)

plt.plot(test_y[:50].tolist(),'r')
#plt.plot(test_y[:50],'r')
plt.plot(y_pred[:50],'g')
plt.show()'''


scores3=[]
for model,model_name in zip(models,model_names):
    model.fit(train_x,train_y)
    y_pred=model.predict(test_x)
    mae = mean_absolute_error(test_y.tolist(),y_pred)
    mse=mean_squared_error(test_y.tolist(),y_pred)
    rmse = np.sqrt(mse)
    r2=r2_score(test_y.tolist(),y_pred)
    scores3.append([mae,mse,rmse,r2])

#作图查看测试数据预测和实际拟合程度
y_lr_predict=lr_model.predict(test_x)
y_tree_predict=tree_model.predict(test_x)
y_rfr_predict=rfr_model.predict(test_x)
y_gbdt_predict=gbdt_model.predict(test_x)
y_xgb_predict = xgb_model.predict(test_x)

fig=plt.figure(figsize=(30,8))
plt.style.use('ggplot')
predicts=[y_lr_predict,y_tree_predict,y_rfr_predict,y_gbdt_predict,y_xgb_predict]
colors=['r','g','y','b','c']
for predict,model_name,color in zip(predicts,model_names,colors):
    plt.plot(predict[:30],color=color,label='predict with {}'.format(model_name))
plt.plot(test_y.tolist()[:30],color='k',label='actual playnum')    
plt.legend(loc=0)


metrics_matrix=pd.DataFrame(scores3,index=model_names,columns=['mae','mse','rmse','r2'])
print('{:*^30}'.format('各模型分数矩阵'))
print(metrics_matrix)

#查看重要程度
i=1
models = models[1:]
model_names = model_names[1:]
fig=plt.figure(figsize=(10,6))
for model,model_name in zip(models,model_names):
    importance_metrics=pd.Series(model.feature_importances_,index=train_x.columns).sort_values()
    plt.style.use('ggplot')
    ax=fig.add_subplot(1,4,i)
    importance_metrics.plot.barh(ax=ax)
    ax.set_xlabel('importance')
    ax.set_title('feature importances of '+model_name)
    i = i+1

plt.show()


