import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,RepeatVector,TimeDistributed, Activation
from tensorflow.keras.layers import ConvLSTM2D,LSTM,GRU,Flatten,Bidirectional


features = 1
seq = 6
steps = 3

#n_lag = 18
#n_seq = 5
# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    #print(X.shape,"x1")
    X = X.reshape((X.shape[0], seq, 1, steps, features))
    # print(X.shape,"X")
    # print(y.shape)
    # design network
    model = Sequential()
    model.fit(X, y, epochs=nb_epoch, batch_size=n_batch, verbose=2, shuffle=True)
    return model

# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
    X = X.reshape((1, seq, 1, steps, features))
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    return [x for x in forecast[0, :]]

# evaluate the persistence model
def make_forecasts(model, n_batch, test, n_lag):
    forecasts = []
    for i in range(len(test)):
        X, y = test[i, 0:n_lag], test[i, n_lag:]
        # make forecast
        forecast = forecast_lstm(model, X, n_batch)
        forecasts.append(forecast)
    file_name = os.path.basename(__file__)  # 获取数据的文件名
    name1, name2 = os.path.splitext(file_name)
    return forecasts, name1