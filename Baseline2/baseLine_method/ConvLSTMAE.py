import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,RepeatVector,TimeDistributed, Activation
from tensorflow.keras.layers import ConvLSTM2D,LSTM,GRU,Flatten,Bidirectional


features = 1
seq = 3
steps = 6

# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    print(X.shape,"x1")

    X = X.reshape((X.shape[0], seq, 1, steps, features))
    #y = y.reshape(y.shape[0], y.shape[1], 1)
    #exit()
    # design network
    model = Sequential()
    model.add(ConvLSTM2D(filters=128, kernel_size=(1, 4), activation='tanh',
                         input_shape=(seq, 1, steps, features),return_sequences=True))
    model.add(Dropout(0.1))
    model.add(ConvLSTM2D(filters=64, kernel_size=(1, 3), activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(RepeatVector(n_seq))
    #model.add(GRU(100, stateful=False, return_sequences=False))
    model.add(Bidirectional(LSTM(100, stateful=False, return_sequences=True)))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(100, stateful=False, return_sequences=False)))
    model.add(Dropout(0.1))
    #model.add(Activation('tanh'))
    model.add(Dense(n_seq))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    # fit model

    model.fit(X, y, epochs=nb_epoch, batch_size=n_batch, verbose=2, shuffle=True)
    return model

# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    #X = X.reshape(1, 1, len(X))
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