import numpy
from pandas import DataFrame
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from matplotlib import pyplot
from numpy import array
from math import fabs

# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = [], []
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

import os
def predic_to_file(dataFrame_real, dataFrame_predic,Mlname,readme):
    # 去掉文件后缀，只要文件名称
    #name = os.path.basename(__file__).split(".")[0]
    name = Mlname
    file_name = os.path.basename(file_path)  # 获取数据的文件名
    name1, name2 = os.path.splitext(file_name)  # 分离文件名与扩展名
    if not os.path.exists('result\\' + name + '\\' + name1):
        os.makedirs('result\\' + name + '\\' + name1)
    dataFrame_real.to_csv('result\\' + name + '\\' + name1 + '\\dataFrame_real.csv', index=False)
    dataFrame_predic.to_csv('result\\' + name + '\\' + name1 + '\\dataFrame_predic.csv', index=False)

    file = open('result\\' + name + '\\' + name1 + '\\readme.txt', mode='a', encoding='utf-8')
    # file.write('\n')
    file.write("数据集：" + name1 + '\n')
    file.write("data_length: %d  n_test: %d\n" % (data_length, n_test))
    file.write("skipfooter: %d\n" % (skipfooter))
    for i in readme:
        file.write(i + '\n')
    file.write('\n')
    file.close()

def prepare_data2(series, n_test, n_lag, n_seq):
    raw_values = series
    diff_values = raw_values.reshape(len(raw_values), 1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(diff_values, n_lag, n_seq)
    supervised_values = supervised.values
    #train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return supervised_values

# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_seq):
    list = []
    for i in range(n_seq):
        actual = [row[i] for row in test]
        predicted = [forecast[i] for forecast in forecasts]
        n = len(actual)
        mape = numpy.sum(numpy.abs((array(actual) - array(predicted)) / actual)) / n * 100
        rmse = sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        print('t+%d MAE: %.4f  RMSE: %.4f  MAPE: %.4f%%  ' % ((i + 1), mae, rmse, mape))
        a = 't+%d MAE: %.4f  RMSE: %.4f  MAPE: %.4f%% ' % ((i + 1), mae, rmse, mape)
        list.append(a)
    return list



file_path = 'data\\Milan_air.csv'

skipfooter = 0
series = read_csv(file_path, encoding='utf-8', header=0, parse_dates=[0], index_col=0,
                  squeeze=True, engine='python', skipfooter=skipfooter)

series = series.dropna()
data = series.values.astype('float32')
#print(data)
max_value = max(data)
#print(max_value)
data = data/max_value

#exit()


data_length = len(data)
# configure
n_lag = 18
n_seq = 4

n_test = int(len(data) * 0.2)
print(n_test)
data_train = data[:-n_test]
data_test = data[-n_test:]

# prepare data
train, test = prepare_data2(data_train,n_test,n_lag,n_seq),prepare_data2(data_test,n_test,n_lag,n_seq)




#from MachineLearning.SVR import Model
from MachineLearning.Xgboost import Model
#prediction
pre, ML_name = Model(train,test,n_lag,n_seq)
pre = pre*max_value

# real data
actual = [row[n_lag:] for row in test]
actual = array(actual)
actual = actual*max_value

#evaluation
list = evaluate_forecasts(actual, pre, n_seq)

df_actual = DataFrame(actual)
df_pre = DataFrame(pre)
#write file
predic_to_file(df_actual,df_pre,ML_name,list)







