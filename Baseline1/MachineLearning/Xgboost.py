from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import pandas as pd
import numpy
from pandas import DataFrame
import numpy as np
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)

def xgboost_forecast(trainx, trainy):

    model = XGBRegressor(objective='reg:squarederror', n_estimators=100,max_depth=1)
    model.fit(trainx, trainy)
    return model

def Model(train,test,n_lag,n_seq):
    trainX = train[:,:n_lag]
    trainY = train[:,n_lag:]
    #print(trainY,trainY.shape,"trainY")
    a_Y = np.mean(trainY, axis=1)
    #print(a_Y,a_Y.shape,"ay")
    model = xgboost_forecast(trainX, a_Y)

    testx = test[:,:n_lag]
    pre = model.predict(testx)
    #print(pre,pre.shape,"pre")
    pre = np.array(np.transpose(np.mat(pre))) #(389, 1) pre2
    pre = pre.repeat(n_seq ,axis=1) #(389, 3) pre3
    str = 'XGboost'
    return pre,str

# for i in range(3):
#     d = testy[:,i]
#     r = pre[:,i]
#     MAE = mean_absolute_error(r,d)
#     RMSE = sqrt(mean_squared_error(r,d))
#     print('MAE: %.3f' % MAE)
#     print('RMSE: %.3f' % RMSE)







