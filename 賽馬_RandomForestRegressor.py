# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 15:28:06 2021

@author: johnn

跑分38
"""

from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_train_1 = pd.read_csv('horse_train.csv')
df_test = pd.read_csv('horse_test.csv')


df_train = df_train_1[df_train_1['horse_id_1'] != df_train_1['horse_id_2']]
# df_test = df_test_1[df_test_1['horse_id_1']!=df_test_1['horse_id_2']]

df_train = df_train.drop(['horse_id_1','horse_id_2'],axis = 1)
df_test = df_test.drop(['horse_id_1','horse_id_2'],axis = 1)



#%%
# 把欄位的文字按順序轉換成0~n的數字
for h in df_train.keys():
    typeshort_1 =df_train[h]
    
    for i in typeshort_1:
        if i == 'G':
            row =  df_train[h] == 'G'
            df_train.loc[row,h] = 5
        elif i =='F':
            row =  df_train[h] == 'F'
            df_train.loc[row,h] = 10
        elif i == 'E':
            row =  df_train[h] == 'E'
            df_train.loc[row,h] = 20
        elif i == 'D':
            row =  df_train[h] == 'D'
            df_train.loc[row,h] = 40
        elif i == 'C':
            row =  df_train[h] == 'C'
            df_train.loc[row,h] = 60
        elif i =='B':
            row =  df_train[h] == 'B'
            df_train.loc[row,h] = 80
        elif i =='A':
            row =  df_train[h] == 'A'
            df_train.loc[row,h] = 100
        else:
            pass

df_train[['short_1','short_2',
            'mile_1','mile_2',
            'medium_1','medium_2',
            'long_1','long_2',
            'front_1','front_2',
            'with-pace_1','with-pace_2',
            'off-pace_1','off-pace_2',
            'stretch_1','stretch_2',
            'grass_1','grass_2',
            'dirt_1','dirt_2',
            ]] = df_train[['short_1','short_2',
            'mile_1','mile_2',
            'medium_1','medium_2',
            'long_1','long_2',
            'front_1','front_2',
            'with-pace_1','with-pace_2',
            'off-pace_1','off-pace_2',
            'stretch_1','stretch_2',
            'grass_1','grass_2',
            'dirt_1','dirt_2',]].astype(int)      
                          
#把各個屬性作加權
#屬性加成
df_train['speed_total'] = df_train['speed_1']+df_train['speed_2']
df_train['stamina_total'] = df_train['stamina_1']+df_train['stamina_2']
df_train['power_total'] = df_train['power_1']+df_train['power_2']
df_train['spirit_total'] = df_train['spirit_1']+df_train['spirit_2']
df_train['intelligence_total'] = df_train['intelligence_1']+df_train['intelligence_2']





df_train = df_train.drop(['speed_1','speed_2'
                ,'stamina_1','stamina_2'
                ,'power_1','power_2'
                ,'spirit_1','spirit_2'
                ,'intelligence_1','intelligence_2'],axis = 1)



#適性加成
df_train['short_total'] = df_train['short_1']+df_train['short_2']
df_train['mile_total'] = df_train['mile_1']+df_train['mile_2']
df_train['medium_total'] = df_train['medium_1']+df_train['medium_2']
df_train['long_total'] = df_train['long_1']+df_train['long_2']
df_train['front_total'] = df_train['front_1']+df_train['front_2']
df_train['with-pace_total'] = df_train['with-pace_1']+df_train['with-pace_2']
df_train['off-pace_total'] = df_train['off-pace_1']+df_train['off-pace_2']
df_train['stretch_total'] = df_train['stretch_1']+df_train['stretch_2']
df_train['grass_total'] = df_train['grass_1']+df_train['grass_2']
df_train['dirt_total'] = df_train['dirt_1']+df_train['dirt_2']

#特徵增加
df_train['short_total**2'] = (df_train['short_1']-df_train['short_2'])**2
df_train['mile_total**2'] = (df_train['mile_1']-df_train['mile_2'])**2
df_train['medium_total**2'] = (df_train['medium_1']-df_train['medium_2'])**2 
df_train['long_total**2'] = (df_train['long_1']-df_train['long_2'])**2
df_train['front_total**2'] = (df_train['front_1']-df_train['front_2'])**2
df_train['with-pace_total**2'] = (df_train['with-pace_1']-df_train['with-pace_2'])**2
df_train['off-pace_total**2'] = (df_train['off-pace_1']-df_train['off-pace_2'])**2
df_train['stretch_total**2'] = (df_train['stretch_1']-df_train['stretch_2'])**2
df_train['grass_total**2'] = (df_train['grass_1']-df_train['grass_2'])**2
df_train['dirt_total**2'] = (df_train['dirt_1']-df_train['dirt_2'])**2



#舊有特徵移除(雜訊
df_train = df_train.drop(['short_1','short_2'
                ,'mile_1','mile_2'
                ,'medium_1','medium_2'
                ,'long_1','long_2'
                ,'front_1','front_2'
                ,'with-pace_1','with-pace_2'
                ,'off-pace_1','off-pace_2'
                ,'stretch_1','stretch_2'
                ,'grass_1','grass_2'
                ,'dirt_1','dirt_2'],axis = 1)

#%%
for a in df_test.keys():
    typeshort_1 =df_test[a]
    for b in typeshort_1:
        if b == 'G':
            row =  df_test[a] == 'G'
            df_test.loc[row,a] = 5
        elif b =='F':
            row =  df_test[a] == 'F'
            df_test.loc[row,a] = 10
        elif b == 'E':
            row =  df_test[a] == 'E'
            df_test.loc[row,a] = 20
        elif b == 'D':
            row =  df_test[a] == 'D'
            df_test.loc[row,a] = 40
        elif b == 'C':
            row =  df_test[a] == 'C'
            df_test.loc[row,a] = 60
        elif b =='B':
            row =  df_test[a] == 'B'
            df_test.loc[row,a] = 80
        elif b =='A':
            row =  df_test[a] == 'A'
            df_test.loc[row,a] = 100
        else:
            pass

df_test[['short_1','short_2',
            'mile_1','mile_2',
            'medium_1','medium_2',
            'long_1','long_2',
            'front_1','front_2',
            'with-pace_1','with-pace_2',
            'off-pace_1','off-pace_2',
            'stretch_1','stretch_2',
            'grass_1','grass_2',
            'dirt_1','dirt_2',
            ]] = df_test[['short_1','short_2',
            'mile_1','mile_2',
            'medium_1','medium_2',
            'long_1','long_2',
            'front_1','front_2',
            'with-pace_1','with-pace_2',
            'off-pace_1','off-pace_2',
            'stretch_1','stretch_2',
            'grass_1','grass_2',
            'dirt_1','dirt_2',]].astype(int)       
# for k in df_test.keys():
#     df_test[k] = df_train[h].astype(int)


#把各個屬性作加權
#屬性加成
df_test['speed_total'] = df_test['speed_1']+df_test['speed_2']
df_test['stamina_total'] = df_test['stamina_1']+df_test['stamina_2']
df_test['power_total'] = df_test['power_1']+df_test['power_2']
df_test['spirit_total'] = df_test['spirit_1']+df_test['spirit_2']
df_test['intelligence_total'] = df_test['intelligence_1']+df_test['intelligence_2']



df_test = df_test.drop(['speed_1','speed_2'
                ,'stamina_1','stamina_2'
                ,'power_1','power_2'
                ,'spirit_1','spirit_2'
                ,'intelligence_1','intelligence_2'],axis = 1)
#適性加成
df_test['short_total'] = df_test['short_1']+df_test['short_2']
df_test['mile_total'] =df_test['mile_1']+df_test['mile_2']
df_test['medium_total'] = df_test['medium_1']+df_test['medium_2']
df_test['long_total'] = df_test['long_1']+df_test['long_2']
df_test['front_total'] = df_test['front_1']+df_test['front_2']
df_test['with-pace_total'] = df_test['with-pace_1']+df_test['with-pace_2']
df_test['off-pace_total'] = df_test['off-pace_1']+df_test['off-pace_2']
df_test['stretch_total'] = df_test['stretch_1']+df_test['stretch_2']
df_test['grass_total'] = df_test['grass_1']+df_test['grass_2']
df_test['dirt_total'] = df_test['dirt_1']+df_test['dirt_2']




#特徵增加
df_test['short_total**2'] = (df_test['short_1']-df_test['short_2'])**2
df_test['mile_total**2'] = (df_test['mile_1']-df_test['mile_2'])**2
df_test['medium_total**2'] = (df_test['medium_1']-df_test['medium_2'])**2 
df_test['long_total**2'] = (df_test['long_1']-df_test['long_2'])**2
df_test['front_total**2'] = (df_test['front_1']-df_test['front_2'])**2
df_test['with-pace_total**2'] = (df_test['with-pace_1']-df_test['with-pace_2'])**2
df_test['off-pace_total**2'] = (df_test['off-pace_1']-df_test['off-pace_2'])**2
df_test['stretch_total**2'] = (df_test['stretch_1']-df_test['stretch_2'])**2
df_test['grass_total**2'] = (df_test['grass_1']-df_test['grass_2'])**2
df_test['dirt_total**2'] = (df_test['dirt_1']-df_test['dirt_2'])**2

df_test = df_test.drop(['short_1','short_2'
                ,'mile_1','mile_2'
                ,'medium_1','medium_2'
                ,'long_1','long_2'
                ,'front_1','front_2'
                ,'with-pace_1','with-pace_2'
                ,'off-pace_1','off-pace_2'
                ,'stretch_1','stretch_2'
                ,'grass_1','grass_2'
                ,'dirt_1','dirt_2'],axis = 1)


#以上為參數，以下為模型類別
#%%  分割訓練&測試集
from sklearn.model_selection import train_test_split
# x_train = df_train.drop('value',axis = 1)
# y_train = df_train['value']

x= df_train.drop('value',axis = 1)
y= df_train['value']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1,random_state=42)
#%%    選擇關聯性高的特徵
# train_x_f4 = x_train[['short_total', 'medium_total', 'long_total', 'with-pace_total',
#        'grass_total', 'dirt_total']]
# test_x_f4 = x_test[['short_total', 'medium_total', 'long_total', 'with-pace_total',
#        'grass_total', 'dirt_total']]


#%%    隨機森林
rfModel = RandomForestRegressor(random_state=1000)
rfModel.fit(x_train,y_train)
predict_rfModel = rfModel.predict(df_test)


#%%    線性回歸

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
predict_lr = lr.predict(df_test)


#%%    xgboost
# from xgboost import XGBRegressor
# from xgboost import XGBClassifier
# from sklearn.model_selection import GridSearchCV

# xgb = XGBRegressor()
# xgb.fit(x_train,y_train)
# predict_xgb = xgb.predict(df_test)

#%% LightGBM
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier

gbm = LGBMRegressor()
gbm.fit(x_train,y_train)
predicts_gbm = gbm.predict(df_test)

#%%    catboost
# from catboost import CatBoostRegressor
# cat = CatBoostRegressor()
# cat.fit(x_train,y_train)
# predicts_cat = gbm.predict(x_test)
#%%    Stacking
from sklearn.ensemble import StackingRegressor
stacking = StackingRegressor(estimators=[('gbm', gbm),('lr', lr)], final_estimator=None)
stacking.fit(x_train,y_train)
predict_stacking = stacking.predict(df_test)

#%%    Bagging
from sklearn.ensemble import BaggingRegressor

bag = BaggingRegressor(base_estimator=rfModel, n_estimators=10)
bag.fit(x_train,y_train)
predict_bag = bag.predict(df_test)
#%% 跑分測試
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve, auc
import itertools
# print("線性回歸 R2 Score: {}".format(r2_score(y_test,predict_lr) * 100))
# print("stacking R2 Score: {}".format(r2_score(y_test,predict_stacking) * 100))
# print("隨機森林 R2 Score: {}".format(r2_score(y_test,predict_rfModel) * 100))
# #print("Xgboost R2 Score: {}".format(r2_score(y_test,predict_xgb) * 100))
# print("LightGBM R2 Score: {}".format(r2_score(y_test,predicts_gbm) * 100))
# #print("catboost R2 Score: {}".format(r2_score(y_test,predicts_cat) * 100))
# print("Bagging R2 Score: {}".format(r2_score(y_test,predict_bag) * 100))
# print('---------------------------------------------------------------------')
# print('線性回歸mean_squared_error: {}'.format(mean_squared_error(y_test, predict_lr)))
# print('stacking mean_squared_error: {}'.format(mean_squared_error(y_test, predict_stacking)))
# print('隨機森林 mean_squared_error: {}'.format(mean_squared_error(y_test, predict_rfModel)))
# print('LightGBM mean_squared_error: {}'.format(mean_squared_error(y_test,predicts_gbm)))

#%%
df_sub = pd.read_csv("sample_submission.csv")

#%%

df_sub['value'] = predicts_gbm

#%%
df_sub.to_csv("./predicts_gbm-07_submission.csv", index=False)

#%%

# plt.scatter(y_test,predicts_gbm)
# plt.plot([0,40],[0,40],marker = '',color='r')
# plt.xlabel('Real')
# plt.ylabel('predicts_gbm')




