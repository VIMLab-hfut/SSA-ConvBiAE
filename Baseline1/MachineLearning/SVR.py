import os
import numpy as np
from matplotlib.font_manager import FontProperties
from sklearn.svm import SVR
def svr_model(trainx,trainy):
    model=SVR(kernel='linear')
    model.fit(trainx,trainy)
    return model


def Model(train,test,n_lag,n_seq):
    trainX = train[:,:n_lag]
    trainY = train[:,n_lag:]
    #print(trainY,trainY.shape,"trainY")
    a_Y = np.mean(trainY, axis=1)
    #print(a_Y,a_Y.shape,"ay")
    model = svr_model(trainX, a_Y)

    testx = test[:,:n_lag]
    pre = model.predict(testx)
    #print(pre,pre.shape,"pre")
    pre = np.array(np.transpose(np.mat(pre))) #(389, 1) pre2
    pre = pre.repeat(n_seq ,axis=1) #(389, 3) pre3

    file_name = os.path.basename(__file__)  # 获取数据的文件名
    name1, name2 = os.path.splitext(file_name)
    return pre,name1









