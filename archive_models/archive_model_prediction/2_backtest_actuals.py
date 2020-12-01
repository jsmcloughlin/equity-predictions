import site;

site.getsitepackages ()
import subprocess
import sys

sys.path.append ("/home/ubuntu/model_test")
from equity_classes import classes as cl
import numpy as np
import pandas as pd
from datetime import datetime, date
import pickle
import joblib
from dateutil.relativedelta import relativedelta
from numpy import loadtxt
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

n_diff = 1
n_out = 6 #set n_out to 6 in order to include day0-day5


def get_backtest(data, days):
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


def concat_predictions(ticker, close, low, dates):
    '''Calculates a % change over 5 days - using day0 as the anchor day.
    Merges with day 1 and day 5 and returns actuals'''

    close [ 'actual_pct_change' ] = ((close.iloc [ :, -1 ] - close.iloc [ :, 0 ]) / close.iloc [ :, 0 ])
    #close = close.iloc [ :, 1: ]

    # Get the 'actual_pct_change' field from the close_df, converts it to array,
    # then calls the label. Then drop the actual_pct_change, keeping only the actual_pct_change_label
    # It then converts the actual_pct_change_label to df and names the field. This is then concatenated
    # back to the main dataframe
    actual_pct_chg_label = \
        pd.DataFrame (aapl.get_chg_pc_label (close.loc [ :, 'actual_pct_change' ].values) [ :, -1 ],
                      columns=[ 'actual_pct_chg_label' ])


    close['low'] = low.iloc [ :, 0 ]

    #low_pct_chg should be the 5 day low subtracted from day0 close - to see the drop
    close ['low_pct_change'] = ((low.iloc [ :, 0 ] - close.iloc [ :, 0 ]) / close.iloc [ :, 0 ])

    close [ 'ticker' ] = ticker

    #close = close.drop ('close0', axis=1)

    return pd.concat ([ dates [ [ 'day1', 'day5' ] ], close, actual_pct_chg_label ], axis=1, sort=False)


def export_actuals(actuals):
    pathname = r'/home/ubuntu/model_test/model_output/'
    filedupe = 'actuals.csv'

    actuals = actuals.dropna (thresh=4)

    actuals.to_csv (pathname + filedupe, index=False, header=True, float_format='%.3f')

aapl = cl.predict_classical ('AAPL')
stock_recent = aapl.get_prepare_stock_data ()  # Pulls source data for all models
backtest_ds = pd.DataFrame (aapl.process_data (stock_recent).loc [ :, ['close', 'low'] ])  # keep only the index (date) and the close
backtest_ds [ 'date' ] = backtest_ds.index

close = get_backtest (backtest_ds.loc [ :, 'close' ].values, n_out) #set to n_out+1 to account for day0 to determine % chg
low = get_backtest (backtest_ds.loc [ :, 'low' ].values, n_out) #[:, 1:] #drop the first row because close takes day0
dates = get_backtest (backtest_ds.loc [ :, 'date' ].values, n_out) #[:, 1:]

close_df = pd.DataFrame (close, columns=[ 'close' + str (x) for x in range (n_out) ])
low_df = pd.DataFrame ([min(row) for row in low[:, 1:]], columns=['5day_low']) #5 day low from day 1 - which is column 2
dates_df = pd.DataFrame (dates, columns=[ 'day' + str (x) for x in range (n_out) ])

actuals = concat_predictions ('AAPL', close_df, low_df, dates_df)
export_actuals (actuals)

