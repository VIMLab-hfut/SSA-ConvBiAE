import numpy
from pandas import DataFrame

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
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
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

import os
def predic_to_file2(real_dataframe,pre_add_dataframe,
                    AE_name,ML_name,predict_value,file_path,data_length,n_test):
    # 去掉文件后缀，只要文件名称
    #name = os.path.basename(__file__).split(".")[0]
    file_name = os.path.basename(file_path)  # 获取数据的文件名
    name1, name2 = os.path.splitext(file_name)  # 分离文件名与扩展名

    if not os.path.exists('step3_ssa_pre\\'+AE_name + '\\'+ML_name + '\\' + name1):
        os.makedirs('step3_ssa_pre\\'+AE_name + '\\'+ML_name + '\\' + name1)
    real_dataframe.to_csv('step3_ssa_pre\\'+AE_name + '\\'+ML_name + '\\' + name1 + '\\real_dataframe.csv', index=False)
    pre_add_dataframe.to_csv('step3_ssa_pre\\'+AE_name + '\\'+ML_name + '\\' + name1 + '\\pre_add_dataframe.csv', index=False)

    file = open('step3_ssa_pre\\'+AE_name + '\\'+ML_name + '\\' + name1 + '\\predict_value.txt', mode='a', encoding='utf-8')
    # file.write('\n')
    file.write("dataset：" + name1 + '\n')
    file.write("data_length: %d  n_test: %d\n" % (data_length, n_test))
    for i in predict_value:
        file.write(i + '\n')
    file.write('\n')
    file.close()


def prepare_data2(series, n_test, n_lag, n_seq):
    # extract raw values
    #raw_values = series.values
    raw_values = series
    diff_values = raw_values.reshape(len(raw_values), 1)
    scaled_values = diff_values.reshape(len(diff_values), 1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
    #print(supervised_values,supervised_values.shape,"supervised")
    value = supervised_values
    return value

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
        print('t+%d MAE: %.4f  RMSE: %.4f  MAPE：%.4f%%  ' % ((i + 1), mae, rmse, mape))
        a = 't+%d MAE: %.4f  RMSE: %.4f  MAPE：%.4f%% ' % ((i + 1), mae, rmse, mape)
        list.append(a)
    return list

skipfooter = 0
def series5(file_path):
    series = read_csv(file_path, encoding='utf-8', header=0, parse_dates=[0], index_col=0,
                      squeeze=True,engine='python', skipfooter=skipfooter)
    series = series.dropna()
    data = series.values.astype('float32')
    print(data,len(data),"data")
    data_max_value = numpy.max(data)
    data = data/data_max_value

    data_length = len(data)
    n_test = int(len(data)*0.2)
    data_train = data[:-n_test]
    data_test = data[-n_test:]
    #print(data_train, len(data_train), "data")
    #print(data_test, len(data_test), "data")
    #exit()


    return data_train,data_test,data_length,n_test,data_max_value
#re1_data,re2_data = return_data(data, windowLen, sum)#data 是原数据，windowLen 是滑窗大小，sum 是用多少个奇异值来还原原始数据

# prepare data


import importlib
def main(file_path,c):
    #seqtoseq prediction
    data_train,data_test,data_length, n_test,data_max_value = series5(file_path)
    train, test = prepare_data2(data_train, n_test, n_lag, n_seq), prepare_data2(data_test, n_test, n_lag, n_seq)

    test_real = prepare_data2(data_test, n_test, n_lag, n_seq)
    print(test_real,"test_real")
    print(test_real*data_max_value,"real")
    test_real = test_real*data_max_value

    a = importlib.import_module(c,"baseLine_method")
    fit_lstm, make_forecasts = a.fit_lstm,a.make_forecasts


    model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
    forecasts,AE_name = make_forecasts(model, n_batch, test, n_lag)
    #forecasts = inverse_transform(forecasts, scaler)

    # real data
    actual = [row[n_lag:] for row in test_real]
    real = numpy.array(actual)

    seqtoseq_pre = numpy.array(forecasts)

    #pre_add
    pre_add = seqtoseq_pre
    pre_add = pre_add*data_max_value
    #exit()

    #evaluation
    list2 = evaluate_forecasts(real,pre_add,n_seq)

    real_dataframe = DataFrame(real)
    pre_add_dataframe = DataFrame(pre_add)

    #prediction_real data to file
    predic_to_file2(real_dataframe,pre_add_dataframe,AE_name,AE_name,list2,file_path,data_length,n_test)
    #readme(list,AE_name,ML_name)
    print("over!!!!!")
    #exit()


if __name__ == '__main__':

    n_lag = 18
    n_seq = 4
    n_epochs = 100
    n_batch = 32
    n_neurons = 128

    file_path1 = 'data\\TH-reservoir.csv'  # skipfooter = 0
    file_path2 = 'data\\Milan_air.csv'
    l = [file_path1] #
    l2 = [".GRU"]  #".ConvLSTMAE",".Bi-LSTM",".GRU",".DLSTM"
    for k in range(len(l2)):
        for i in range(len(l)):
            for j in range(1):
                main(l[i],l2[k])









