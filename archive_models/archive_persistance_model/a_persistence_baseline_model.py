from math import sqrt
from numpy import mean, median
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from warnings import catch_warnings, filterwarnings
from sklearn.metrics import mean_squared_error
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
from yahoo_fin import stock_info
# from yahoo_fin.options import *
import stockstats
from stockstats import StockDataFrame as sdf
from pandas_datareader import data as pdr
import requests_html
#python -m pip install requests-html

'''Develop a baseline univariate model, using only daily close to determine the
next day's stock price.

For example, if n=1 and offset=3, then the avg is calculated from the single value
at n x offset or 1 x 3 = -3. If n = 2, then it would 2 x 3 = -6.

By having the persist, mean and median baselines in the one function, we can
then test a number of different configurations.'''


# one-step naive forecast
def naive_forecast(history, n):
    return history[-n]# test naive forecast

# one-step simple forecast
def simple_forecast(history, config):
    n, offset, avg_type = config
# persist value, ignore other config
    if avg_type == 'persist':
        return history[-n]
# collect values to average
    values = list()
    if offset == 1:
        values = history[-n:]
    else:
    # skip bad configs
        if n*offset > len(history):
            raise Exception('Config beyond end of data: %d %d' % (n,offset))
    # try and collect n values using offset
    for i in range(1, n+1):
        ix = i * offset
        values.append(history[-ix])
  # check if we can average
    if len(values) < 2:
        raise Exception('Cannot calculate average')
  # mean of last n values
    if avg_type == 'mean':
        return mean(values)
  # median of last n values
    return median(values)

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]

# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
    predictions = list()
# split dataset
    train, test = train_test_split(data, n_test) # seed history with training dataset
    history = [x for x in train]
  # step over each time step in the test set
    for i in range(len(test)):
    # fit model and make forecast for history
        yhat = simple_forecast(history, cfg)
    # store forecast in list of predictions
        predictions.append(yhat)
    # add actual observation to history for the next loop
        history.append(test[i])
  # estimate prediction error
    error = measure_rmse(test, predictions)
    return error

# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation (data, n_test, cfg)
    else:
    # one failure during model validation suggests an unstable config
        try:
        # never show warnings when grid searching, too noisy
            with catch_warnings ():
                filterwarnings ("ignore")
                result = walk_forward_validation (data, n_test, cfg)
        except:
            error = None
    # check for an interesting result
    if result is not None:
        print (' > Model[%s] %.3f' % (key, result))
    return (key, result)

# define executor
executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')

# define list of tasks
tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)

# execute list of tasks
scores = executor(tasks)

# execute list of tasks sequentially
scores = [score_model(data, n_test, cfg) for cfg in cfg_list]

# order scores
scores = [r for r in scores if r[1] != None]

# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
    scores = None
    if parallel:
    # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
  # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores

# create a set of simple configs to try
def simple_configs(max_length, offsets=[1]):
    configs = list()
    for i in range(1, max_length+1):
        for o in offsets:
            for t in ['persist', 'mean', 'median']:
                cfg = [i, o, t]
                configs.append(cfg)
    return configs

headers = pd.read_csv (r'/home/ubuntu/stock_lstm/export_files/headers.csv')
df = pd.read_csv (r'/home/ubuntu/stock_lstm/export_files/stock_history.csv', header=None, names=list(headers))

# Extract the close for 'AAPL' only
history = df[df['ticker'] == 'AAPL']['close'].sort_index(ascending=True)
history.index.name = 'date'
n_test_pct = 0.1
n_test = int(len(history) * n_test_pct)
# model configs
max_length = (len (history) - n_test)
cfg_list = simple_configs (max_length)
# grid search
scores = grid_search (history, cfg_list, n_test)
print ('done')
# list top 3 configs
for cfg, error in scores [ :5 ]:
    print (cfg, error)
