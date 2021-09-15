import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,RepeatVector,TimeDistributed, Activation
from tensorflow.keras.layers import LSTM,GRU,Bidirectional
from tensorflow.keras import optimizers
from math import sqrt
from matplotlib import pyplot
from numpy import array

# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]

    X = X.reshape(X.shape[0], 1, X.shape[1])
    #y = y.reshape(y.shape[0], y.shape[1], 1)

    # design network
    model = Sequential()#,activation='tanh'
    model.add(Bidirectional(GRU(n_neurons, input_shape=(X.shape[1], X.shape[2]), stateful=False, return_sequences=False)))
    model.add(Activation('tanh'))
    model.add(Dense(y.shape[1]))
    #model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam')
    #optimizers.Adam(learning_rate=0.0001)
    model.fit(X, y, epochs=nb_epoch, batch_size=n_batch, verbose=2)
    return model

# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, 1, len(X))
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
    return forecasts,name1



