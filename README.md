# SSA-ConvBiAE
time series prediction
## Abstract
Time series forecasting currently has a wide range of applications in many fields. It can help people make important decisions if they can accurately estimate the future development of events or indicators. However, modeling and accurately predicting time series with different features has become one of the most challenging applications. Therefore, we propose a novel hybrid multi-step prediction model, which is combined with singular spectrum analysis (SSA) and end-to-end model, called SSA-ConvBiAE. Specifically, SSA is used to decompose the original data into different trending components, and the autoencoder model by using Convolutional LSTM (ConvLSTM) and Bidirectional GRU (BiGRU) as encoder and decoder units, which enables the ability to model complex data. Finally, the different components are input to the corresponding models for training and prediction and summing the prediction results. To evaluate the predictive performance of our model, we conducted experiments on two real water supply datasets and two publicly available time series datasets. Experimental results show that our proposed model achieves better performance than other baseline methods.


## Data

### (1) TH-reservoir. The dataset is the water level value of a reservoir in Huangshan Scenic Area. This dataset contains daily water level values from January 1, 2017 to December 31, 2019.
### (2) XH-waterworks. The dataset is the water supply of a water plant in Huangshan Scenic Area. This dataset contains daily water supply from January 1, 2017 to December 31, 2019.
### (3) Milan-air. This dataset is the Milan air PM2.5 concentration data. This dataset contains the average of hourly PM2.5 concentrations from July 24, 2020 to September 20, 2020.
### (4) Delhi-meantemp. The dataset is weather temperature data for the city of Delhi, India. This dataset contains the average of daily temperatures from January 1, 2013 to January 1, 2017.



## Usage

python main_prediction.py

