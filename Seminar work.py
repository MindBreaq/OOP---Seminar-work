import math
import numpy as np
from datetime import datetime
import yfinance as yf
from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

class StockPredictionModel:
    ## Definating initial variables and output variables of functions
    def __init__(self, stock, y_symbols, startdate, enddate, data_column, days_in, days_to_predict, epochs):
        self.stock = stock
        self.y_symbols = y_symbols
        self.startdate = startdate
        self.enddate = enddate
        self.data_column = data_column
        self.days_in = days_in
        self.days_to_predict = days_to_predict
        self.epochs = epochs
        self.data = self.download_stock_data()
        self.dataset, self.scaler, self.scaled_data = self.preprocess_data()
        self.training_data_len, self.train_in, self.train_out = self.create_training_data()
        self.model = self.build_model()
        self.train_model()
        self.test_in_by_one, self.test_out = self.create_testing_data()
        self.predictions = self.predict_by_one()
        self.plot = self.plot_predictions()

    ## Downloading datas from yahoo finance
    def download_stock_data(self):
        yf.pdr_override()
        data = pdr.get_data_yahoo(self.y_symbols, start=self.startdate, end=self.enddate)
        return data

    ## Preparing datas for model training
    def preprocess_data(self):
        dataset = self.data.filter([self.data_column]).values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        return dataset, scaler, scaled_data

    ## Creating training dataset
    def create_training_data(self):
        training_data_len = math.ceil(len(self.scaled_data) - self.days_to_predict)
        training_data = self.scaled_data[0:training_data_len, :]
        train_in = []
        train_out = []
        for i in range(self.days_in, len(training_data)):
            train_in.append(training_data[i-self.days_in:i,0])
            train_out.append(training_data[i,0])
        train_in = np.array(train_in)
        train_out = np.array(train_out)
        train_in = np.reshape(train_in, (train_in.shape[0], train_in.shape[1], 1))
        return training_data_len, train_in, train_out

    ## Model building
    def build_model(self):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(self.train_in.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    ## Model training
    def train_model(self):
        self.model.fit(self.train_in, self.train_out, batch_size=1, epochs=self.epochs)
    
    ## Creating testing dataset
    def create_testing_data(self):
        test_data = self.scaled_data[self.training_data_len-self.days_in:,:]
        test_out = self.dataset[self.training_data_len:, :]
        test_in_by_one = []
        for i in range(self.days_in, len(test_data)):
            test_in_by_one.append(test_data[i-self.days_in:i, 0])
        test_in_by_one = np.array(test_in_by_one)
        test_in_by_one = np.reshape(test_in_by_one, (test_in_by_one.shape[0], test_in_by_one.shape[1], 1))
        return test_in_by_one, test_out
    
    ## Testing
    def predict_by_one(self):
        predictions = self.model.predict(self.test_in_by_one)
        predictions = self.scaler.inverse_transform(predictions)    
        return predictions 

    ## Output plot
    def plot_predictions(self):
        plt.style.use('ggplot')
        plt.figure(figsize=(20,10))
        plt.plot(self.test_out, color='red', label='Original')
        plt.plot(self.predictions, color='blue', label='Prediction')
        plt.title(self.stock, fontsize=30)
        plt.xlabel('Date', fontsize=20)
        plt.ylabel('Close Price', fontsize=20)
        plt.legend(loc='lower left')
        plt.show()

## Inputs
stock = 'AAPL'  # Other examples: S&P 500 = '^GSPC', Gold = 'GC=F', BTC = 'BTC-USD'
model = StockPredictionModel(stock, [stock], datetime(2010,1,1),    # Start date
                                             datetime(2022,12,27),  # End date
                                             'Close',
                                             5,     # number of days, from which model predicts
                                             60,    # Days to be predicted
                                             2      # Epochs
)