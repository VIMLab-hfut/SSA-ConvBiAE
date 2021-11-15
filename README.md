# SSA-ConvBiAE
time series prediction
## Abstract
Time series forecasting currently has a wide range of applications in many fields. It can help people make important decisions if they can accurately estimate the future development of events or indicators. However, modeling and accurately predicting time series with different features has become one of the most challenging applications. Therefore, we propose a novel hybrid multi-step prediction model, which is combined with singular spectrum analysis (SSA) and end-to-end model, called SSA-ConvBiAE.


## Data

TH-reservoir and XH-waterworks are real water supply datasets obtained from Huangshan Scenic Area, China. milan-air and delhi-meantemp are publicly available datasets obtained from Kaggle.

### 
(1) TH-reservoir. The dataset is the water level value of a reservoir in Huangshan Scenic Area. This dataset contains daily water level values from January 1, 2017 to December 31, 2019.
### 
(2) XH-waterworks. The dataset is the water supply of a water plant in Huangshan Scenic Area. This dataset contains daily water supply from January 1, 2017 to December 31, 2019.
### 
(3) Milan-air. This dataset is the Milan air PM2.5 concentration data. This dataset contains the average of hourly PM2.5 concentrations from July 24, 2020 to September 20, 2020. milan-air is publicly available datasets obtained from Kaggle.https://www.kaggle.com/
### 
(4) Delhi-meantemp. The dataset is weather temperature data for the city of Delhi, India. This dataset contains the average of daily temperatures from January 1, 2013 to January 1, 2017. delhi-meantemp is publicly available datasets obtained from Kaggle. https://www.kaggle.com/
###
（5）
Global-power. This dataset is the electricity consumption data of a household over a four-year period. The data is collected every minute and the daily consumption is calculated, the dataset contains the daily consumption values from December 16, 2006 to November 26, 2010.


## Usage

python main_prediction.py

