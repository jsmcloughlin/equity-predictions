import requests_html
import sys
#sys.path.append("/home/ubuntu/model_test")
#sys.path.append('/tmp/pycharm_project_765/venv/lib/python3.8/site-packages')
#sys.path.append('/home/ubuntu/.local/lib/python3.7/site-packages')
from math import sqrt
from numpy import split, array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from equity_classes import classes
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, date
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import classification_report, balanced_accuracy_score
from keras.utils import np_utils
import subprocess
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import os
os.getcwd()

import sys
#print(sys.path)


class parent_rnn:

    def __init__(self, ticker):
        self.ticker = ticker
        if __name__ == '__main__':
            parent_rnn().print_ticker()
        pass

    def print_ticker(self):
        print("Ticker is: " +str(self.ticker))


    def reshape_dataset(self, data, timesteps):
        '''timesteps here should be set to 1 - for simplicity
        returns the input but in 3D form in equal spaced timesteps'''

        leftover = data.shape[0] % timesteps  # Reduce the data to a number divisible by 5
        data_sub = data[leftover:]
        data_sub = np.array(np.split(data_sub, len(data_sub) / timesteps))

        # If univariate input, returns reshaped from 2d to 3d - otherwise, returns 3d
        if data_sub.ndim == 2:
            return data_sub.reshape(data_sub.shape[0], data_sub.shape[1], 1)
        else:
            return data_sub


    def get_difference(self, data, interval=1):
        '''Takes in a 2d dataset and an interval (default is 1), then returns
        a differenced value'''

        diff = list()
        for i in range(interval, len(data)):
            value = data[i] - data[i - interval]
            diff.append(value)

        return diff

    def get_difference_pct(self, data, interval=1):
        '''Takes in a 2d dataset and an interval (default is 1), then returns
        a differenced value'''

        diff = list()
        for i in range(interval, len(data)):
            value = (data[i] - data[i - interval]) / data[i - interval]
            diff.append(value)

        return diff

    def invert_scale(self, scaler_y, yhat):
        inverted = scaler_y.inverse_transform(yhat)
        return inverted[0, :]

    def split_dataset(self, data, train_pct):

        '''performs the splitting of the already reshaped dataset'''

        train_weeks = int(data.shape[0] * train_pct)
        train, test = data[0:train_weeks, :], data[train_weeks:, :]

        return train, test

    # convert history into inputs and outputs
    def to_supervised(self, data_in, n_input, n_out):
        # reshaping has to be this way, because when preparing multivariate data,
        # the shape is as follows: 1500, 14, 1 versus 1500, 1, 1 for univariate.
        # For univariate, we can reshape as follows: data_in.reshape((data_in.shape[0] * data_in.shape[1], data_in.shape[2]))
        # However, for mvariate, this will not work because of the second dim
        #data = data_in.reshape((data_in.shape[0] * data_in.shape[1], data_in.shape[2]))
        data = data_in
        X, y = list(), list()
        in_start = 0
        # step over the entire history one time step at a time
        for _ in range(len(data)):
            # define the end of the input sequence
            in_end = in_start + n_input
            out_end = in_end + n_out
            # ensure we have enough data for this instance
            if out_end <= len(data):
                X.append(data[in_start:in_end, :])
                y.append(data[in_end:out_end, 0])
            # move along one time step
            in_start += 1
        return array(X), array(y)


    def create_scaler(self, train):
        '''Takes a 2d data set regardless of uni or multivariate features
        e.g. 1500, 1 or 1500, 15'''

        s0, s1 = train.shape[0], train.shape[1]
        scaler, scaler_y = RobustScaler(), RobustScaler()

        train_y = train[:, 0]
        train_y = train_y.reshape(train_y.shape[0], 1)

        scaler = scaler.fit(train)
        scaler_y = scaler_y.fit(train_y)  # scale the first column for the close

        train_scale = scaler.fit_transform(train)

        return train_scale, scaler, scaler_y

    def apply_scaler(self, test, scaler, scaler_y):
        s0, s1 = test.shape[0], test.shape[1]

        test_y = test[:, 0]
        test_y = test_y.reshape(test_y.shape[0], 1)

        # test_x = test.reshape(s0 * s1, s2)
        test_scale = scaler.transform(test)
        # test_scale = test_x.reshape(s0, 1, s1)
        return test_scale

    def timer(self, start_time=None):
        if not start_time:
            start_time = datetime.now()
            return start_time
        elif start_time:
            thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
            tmin, tsec = divmod(temp_sec, 60)
            print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

    def forecast_d(self, model, history, n_input, scaler, scaler_y):
        '''The only difference with function: forecast_c is that there is
        NO allowance for differencing (removal of the trend) - interval input var is unneeded'''

        data = array(history)

        # If not differencing
        data = self.apply_scaler(np.array(data), scaler,
                                   scaler_y)  # difference the history data to prepare next test sample

        # retrieve last observations for input data
        input_x = data[-n_input:, :]
        input_x = input_x.reshape(1, input_x.shape[0], input_x.shape[1])

        # forecast the next week on the differenced data
        yhat = model.predict(input_x, verbose=0)

        return self.invert_scale(scaler_y, yhat)

    def forecast_c(self, model, history, n_input, scaler, scaler_y, interval):
        '''The only difference with function: forecast_d is that there is
        an allowance for differencing (removal of the trend)'''

        data = array(history)

        #If differencing
        data = self.apply_scaler(np.array(self.get_difference(data, interval)), scaler,
                                   scaler_y)  # difference the history data to prepare next test sample

        # retrieve last observations for input data
        input_x = data[-n_input:, :]
        input_x = input_x.reshape(1, input_x.shape[0], input_x.shape[1])

        # forecast the next week on the differenced data
        yhat = model.predict(input_x, verbose=0)

        return self.invert_scale(scaler_y, yhat)


    def forecast_mh_cnn(self, model, history, n_input, scaler, scaler_y, interval):
        #This differs from forecast - since the model is multi headed,
        #each feature is fed into the cnn individually - which requires a
        #key difference: input_x = [input_x[:, i].reshape((1, input_x.shape[0], 1)) for i in range(input_x.shape[1])]
        # flatten data
        data = array(history)
        data = self.apply_scaler(np.array(self.get_difference(data, interval)), scaler, scaler_y)
        data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))

        if data.shape[1] == 1:  # univariate versus multi
            input_x = data[-n_input:, 0]
            input_x = input_x.reshape((1, len(input_x), 1))
        else:
            input_x = data[-n_input:, :]
            input_x = [input_x[:, i].reshape((1, input_x.shape[0], 1)) for i in range(input_x.shape[1])]
        # forecast the next week
        yhat = model.predict(input_x, verbose=0)

        return self.invert_scale(scaler_y, yhat)

    def get_prepare_stock_data(self):

        subprocess.call(['python3', '/home/ubuntu/model_test/web_scraping/prepare_stock_data.py'])

        headers = pd.read_csv(r'/home/ubuntu/model_test/export_files/headers.csv')
        df = pd.read_csv(r'/home/ubuntu/model_test/export_files/stock_history.csv', header=None, names=list(headers))
        return df

    def prepare_variate(self, data, ndim):
        if ndim == 1:
            arr = np.array(data.iloc[:, 0])
            arr = arr.reshape(arr.shape[0], 1)
        else:
            data = data[['close', 'high', 'low', 'volume']]
            #data = data.drop(['adj close', 'day', 'ticker', 'volume_delta', 'prev_close_ch', 'prev_volume_ch',
             #                 'macds', 'macd', 'dma', 'macdh', 'ma200', 'day_1', 'day_2', 'day_3', 'day_4'], axis=1)
            #arr = np.array(data)
            arr = data

        return arr

    def process_data(self, df):

        df.index.name = 'date'

        df.reset_index(inplace=True)  # temporarily reset the index to get the week day for OHE
        df['date'] = pd.to_datetime(df['date'])
        df.drop_duplicates(['date', 'ticker', 'close'], inplace=True)
        df['day'] = list(map(lambda x: datetime.weekday(x), df['date']))  # adds the numeric day for OHE
        df.set_index('date', inplace=True)  # set the index back to the date field

        # use pd.concat to join the new columns with your original dataframe
        df = pd.concat([df, pd.get_dummies(df['day'], prefix='day', drop_first=True)], axis=1)

        #print("Ticker is (pd): " + str(self.ticker))
        df_close = df[df['ticker'] == self.ticker].sort_index(ascending=True)

        df_close = df_close.sort_index(ascending=True, axis=0)

        # Move the target variable to the end of the dataset so that it can be split into X and Y for Train and Test
        cols = list(df_close.columns.values)  # Make a list of all of the columns in the df
        cols.pop(cols.index('close'))  # Remove outcome from list
        df_close = df_close[['close'] + cols]  # Create new dataframe with columns in correct order

        df_close = df_close.dropna()

        #fig, axes = plt.subplots(figsize=(16, 8))
        # Define the date format
        #axes.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # to display ticks every 3 months
        #axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # to set how dates are displayed
        #axes.set_title(self.ticker)
        #axes.plot(df_close.index, df_close['close'], linewidth=3)
        #plt.show()

        return df_close

    def model_configs(self):
        # define scope of configs
        n_input = [10, 15, 25]
        n_nodes = [25, 50, 100]
        n_epochs = [25, 50, 100, 500, 700]
        n_batch = [8, 16, 32, 64]
        n_diff = [1]
        n_out = [5]
        n_lr = [0.001, 0.005, 0.01, 0.05, 0.09, 0.15, 0.20]
        #n_lr = [0.001, 0.005, 0.01, 0.05, 0.09]
        n_actfn = ['tanh', 'relu']
        n_dropout = [0.5, 0.2, 0.1, 0]

        # create configs
        configs = list()
        for i in n_input:
            for j in n_nodes:
                for k in n_epochs:
                    for l in n_batch:
                        for m in n_diff:
                            for n in n_out:
                                for o in n_lr:
                                    for p in n_actfn:
                                        for q in n_dropout:
                                            cfg = [i, j, k, l, m, n, o, p, q]
                                            configs.append(cfg)
        print('Total configs: %d' % len(configs))
        return configs

    def get_exp_charts(self, data):
        '''plots 3 charts:
        Plot 1: histogram of the % change distribution across all categories
        Plot 2: Bar chart counting each category
        Plot 3: histogram grouped by each category'''
        sns.distplot(data.iloc[:, 0], kde=True, bins=100).set_title(data.columns[0])
        plt.show()

        sns.countplot(x=data.iloc[:, 1], data=data).set_title(data.columns[0])
        plt.show()

        fig = plt.figure(figsize=(8, 4))

        labels = data.iloc[:, 1].unique()
        for i in labels:
            sns.distplot(data[data.iloc[:, 1] == i].iloc[:, 0], kde=False, bins=25)
        fig.legend(labels=labels)
        plt.show()

    def encode_y(self, data):
        y = data.iloc[:, 1]
        encoder = LabelEncoder()
        encoder.fit(y)
        encoded_y = encoder.transform(y)  # would be sufficient for a binary classifier
        # convert integers to dummy variables (i.e. one hot encoded) - because of multi-classification
        dummy_y = np_utils.to_categorical(encoded_y)
        return dummy_y, encoded_y, encoder

    def decode_y(self, y, encoder):
        '''Takes the dummy_y and returns the original label'''
        #inv_encode = encoder.inverse_transform(np.argmax(y, axis=-1))
        inv_encode = encoder.inverse_transform(y)
        return pd.DataFrame({'y_label': inv_encode})

    def decode_yhat(self, yhat, yhat_proba, encoder):
        '''Takes the y and returns the original label and the max probability for the classifier
        Can be applied to yhat output or the dummy_y - columns will require renaming'''
        # inv_encode = encoder.inverse_transform(np.argmax(yhat, axis=-1))
        inv_encode = encoder.inverse_transform(yhat)
        inv_proba = np.max(yhat_proba, axis=-1)
        return pd.DataFrame({'yhat_proba': inv_proba, 'yhat_label': inv_encode})

    def get_classifier_results(self, model, X_test, y_test, encoder):
        print('The parameters of the best model are: ')
        print(model.best_params_)
        results = model.cv_results_

        best_model = model.best_estimator_
        print(best_model)

        yhat = best_model.predict(X_test)
        yhat_proba = best_model.predict_proba(X_test)

        yhat_inv = self.decode_yhat(yhat, yhat_proba, encoder)

        y_inv = self.decode_y(y_test, encoder)
        df_inv = pd.concat([yhat_inv, y_inv], axis=1)

        df_inv['rank'] = df_inv.groupby(['y_label'])['yhat_proba'].transform(
            lambda x: pd.qcut(x, 5, labels=range(1, 6)))

        df_inv['rank'] = df_inv['rank'].astype('int32')

        print(
            'Balanced Accuracy Score (Overall): \n' + str(
                balanced_accuracy_score(df_inv['y_label'], df_inv['yhat_label'])))
        print('Balanced Crosstab Rank (Overall): \n' + str(
            pd.crosstab(df_inv['y_label'], df_inv['yhat_label'], rownames=['Actual'], colnames=['Predicted'])))

        rcount = list(df_inv['rank'].unique())
        for i in range(1, len(rcount) + 1):
            df = df_inv[df_inv['rank'] == i]
            print('Balanced Accuracy Score Rank \n' + str(i) + ' ' + str(
                balanced_accuracy_score(df['y_label'], df['yhat_label'])))
            print('Balanced Crosstab Rank \n' + str(i) + ' ' + str(
                pd.crosstab(df['y_label'], df['yhat_label'], rownames=['Actual'], colnames=['Predicted'])))

    def get_regressor_results(self, model, X_test, y_test, encoder):
        print('The parameters of the best model are: ')
        print(model.best_params_)
        results = model.cv_results_

        best_model = model.best_estimator_
        print(best_model)

        yhat = best_model.predict(X_test)
        yhat_proba = best_model.predict_proba(X_test)

        yhat_inv = self.decode_yhat(yhat, yhat_proba, encoder)

        y_inv = self.decode_y(y_test, encoder)
        df_inv = pd.concat([yhat_inv, y_inv], axis=1)

        df_inv['rank'] = df_inv.groupby(['y_label'])['yhat_proba'].transform(
            lambda x: pd.qcut(x, 5, labels=range(1, 6)))

        df_inv['rank'] = df_inv['rank'].astype('int32')

        print(
            'Balanced Accuracy Score (Overall): \n' + str(
                balanced_accuracy_score(df_inv['y_label'], df_inv['yhat_label'])))
        print('Balanced Crosstab Rank (Overall): \n' + str(
            pd.crosstab(df_inv['y_label'], df_inv['yhat_label'], rownames=['Actual'], colnames=['Predicted'])))

        rcount = list(df_inv['rank'].unique())
        for i in range(1, len(rcount) + 1):
            df = df_inv[df_inv['rank'] == i]
            print('Balanced Accuracy Score Rank \n' + str(i) + ' ' + str(
                balanced_accuracy_score(df['y_label'], df['yhat_label'])))
            print('Balanced Crosstab Rank \n' + str(i) + ' ' + str(
                pd.crosstab(df['y_label'], df['yhat_label'], rownames=['Actual'], colnames=['Predicted'])))


class prepare_classical(parent_rnn):

    def __init__(self, ticker):
        self.ticker = ticker
        super(prepare_classical).__init__()


    # convert history into inputs and outputs - includes previous day
    def to_supervised_classical(self, train, n_input, n_out):
        # flatten data
        data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
        X, y = list(), list()
        in_start = 0
        # step over the entire history one time step at a time
        for _ in range(len(data)):
            # define the end of the input sequence
            in_end = in_start + n_input
            out_end = in_end + n_out
            # ensure we have enough data for this instance
            if out_end <= len(data):
                X.append(data[in_start:in_end, :])
                y.append(data[in_end - 1:out_end, 0])  # slightly different behavior here than the equivalent
            # classes function. This includes the previous day as well.
            # move along one time step
            in_start += 1
        return array(X), array(y)

    def reshape_X_classical(self, data):
        return data.reshape(data.shape[0], data.shape[1] * data.shape[2])

    def reshape_y_classical(self, data, n_out=5):
        '''This function takes the y output from to_supervised_classical
        as well as the number of steps. It then calculates the
        % change cumulating from day 1 to day n (usually 5). The
        first day's % change is calculated from the previous day'''
        pct_cume = []
        df = np.array(pd.DataFrame(data).pct_change(axis=1).iloc[:, 1:])

        for i in range(len(df)):
            pct_cume.append([sum(df[i, 0:x:1]) for x in range(0, n_out + 1)])

        return np.array(pct_cume)[:, 1:]

    def get_chg_pc(self, data):

        '''Calculates the intraday max across n days as well as
        the end of week % change (not necessarily max for the week)
        returns 2 lists'''

        nday_chg, intraday_max = [], []

        for i in range(len(data)):
            nday_chg.append(data[i, -1])  # get the final day change over the n days
            intraday_max.append(data[i, :].max())  # get the max cumulative change over the n days

        return nday_chg, intraday_max

    def get_chg_pc_label(self, data):
        '''Adds label based on the changes - will be used for classification and regression attempts'''
        add_label = []
        for i in range(len(data)):
            if data[i] >= 0.04:
                add_label.append([data[i], '4+% Up'])
            elif 0.01 <= data[i] < 0.04:
                add_label.append([data[i], '1-4% Up'])
            elif -0.01 <= data[i] < 0.01:
                add_label.append([data[i], '+/-1%'])
            elif -0.04 <= data[i] < -0.01:
                add_label.append([data[i], '1-4% Down'])
            elif data[i] < -0.04:
                add_label.append([data[i], '4+% Down'])
            else:
                add_label.append([data[i], 'Other'])
        return np.array(add_label)

    def get_4pct_accuracy(self, df):
        '''Calculate the TP, FP and FN of the current classification model.
        df consists of a two column df: titled actual and pred. From there
        the tp, fp and fn are calculated for precision and recall'''

        df_4pct = df [ (df.actual == '4+% Up') | (df.pred == '4+% Up') ]
        c = {'tp': 0, 'fn': 0, 'fp': 0}
        df_4pct = df_4pct.assign (**c)

        for i in range (len (df_4pct)):
            df_4pct [ 'tp' ].iloc [ i ] = (
                1 if df_4pct [ 'actual' ].iloc [ i ] == df_4pct [ 'pred' ].iloc [ i ] else 0)
            df_4pct [ 'fn' ].iloc [ i ] = (
                1 if df_4pct [ 'actual' ].iloc [ i ] == '4+% Up' and df_4pct [ 'pred' ].iloc [
                    i ] != '4+% Up' else 0)
            df_4pct [ 'fp' ].iloc [ i ] = (
                1 if df_4pct [ 'actual' ].iloc [ i ] != '4+% Up' and df_4pct [ 'pred' ].iloc [
                    i ] == '4+% Up' else 0)

        return df_4pct

    def precision_recall(self, df):
        '''takes the output from get_4pct_accuracy(self, df)
        to calculate the actual Precision and Recall.'''
        tp = np.sum (df [ 'tp' ])
        fp = np.sum (df [ 'fp' ])
        fn = np.sum (df [ 'fn' ])
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        return precision, recall

    # Start Actuals....
    def get_backtest(self, data, days):
        '''This function will take in close, and secondly dates. The close
        will go back 6 days in order to determine the % change over the
        5 days. The date range only needs 5 days to get the date range only.'''

        X = list ()
        in_start = 0
        # step over the entire history one time step at a time
        for _ in range (len (data)):
            in_end = in_start + days
            if in_end <= len (data):
                X.append (data [ in_start:in_end ])
            in_start += 1
        return np.array (X)

class predict_classical(prepare_classical):

    def __init__(self, ticker):
        self.ticker = ticker
        super(predict_classical).__init__()

    def get_stock_recent(self):
        subprocess.call(['python3', '/home/ubuntu/model_test/web_scraping/scrape_stock_data.py'])
        headers = pd.read_csv(r'/home/ubuntu/model_test/export_files/headers.csv')
        return pd.read_csv(r'/home/ubuntu/model_test/export_files/stock_recent.csv', header=None, names=list(headers))
