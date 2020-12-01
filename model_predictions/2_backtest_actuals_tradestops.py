import site
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
from keras.models import load_model
import os
from datetime import datetime, timedelta

sys.path.append ("/home/ubuntu/model_test")
from equity_classes import classes as cl
import joblib
import pickle
from equity_classes import class_prepare_stock as cps
from stockstats import StockDataFrame as sdf
import scipy.stats as stats

os.environ [ 'TF_CPP_MIN_LOG_LEVEL' ] = '2'

n_input_cls = 15
n_diff = 1
n_out = 6  # has to be set to 6
n_input_seq = 10


def instantiate(ticker):
    '''instantiate 2 class objects for a specific ticker'''
    return cps.yfinance_scrape (ticker), cl.prepare_classical (ticker)


def get_data():
    '''Pulls recent stock data for a specific equity'''
    return scraper.run_cps ()


# Start Actuals....
'''
def get_backtest(data, days):
    This function will take in close, and secondly dates. The close
    will go back 6 days in order to determine the % change over the
    5 days. The date range only needs 5 days to get the date range only.

    X = list ()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range (len (data)):
        in_end = in_start + days
        if in_end <= len (data):
            X.append (data [ in_start:in_end ])
        in_start += 1
    return np.array (X)
'''


def concat_actuals(ticker, close, low, dates):
    '''Calculates a % change over 5 days - using day0 as the anchor day.
    Merges with day 1 and day 5 and returns actuals'''

    close [ 'actual_pct_change' ] = ((close.iloc [ :, -1 ] - close.iloc [ :, 0 ]) / close.iloc [ :, 0 ])

    # Get the 'actual_pct_change' field from the close_df, converts it to array,
    # then calls the label. Then drop the actual_pct_change, keeping only the actual_pct_change_label
    # It then converts the actual_pct_change_label to df and names the field. This is then concatenated
    # back to the main dataframe
    actual_pct_chg_label = \
        pd.DataFrame (processor.get_chg_pc_label (close.loc [ :, 'actual_pct_change' ].values) [ :, -1 ],
                      columns=[ 'actual_pct_chg_label' ])
    close [ 'low' ] = low.iloc [ :, 0 ]

    # low_pct_chg should be the 5 day low subtracted from day0 close - to see the drop
    close [ 'low_pct_change' ] = ((low.iloc [ :, 0 ] - close.iloc [ :, 0 ]) / close.iloc [ :, 0 ])
    close [ 'ticker' ] = ticker

    actuals = pd.concat ([ dates [ [ 'day1', 'day5' ] ], close, actual_pct_chg_label ], axis=1, sort=False)

    floats = [ 'close0', 'close1', 'close2', 'close3', 'close4', 'close5', 'actual_pct_change', 'low',
               'low_pct_change' ]
    for colname in floats:
        actuals [ colname ] = pd.to_numeric (actuals [ colname ], errors='coerce')

    return actuals


def get_zscore(pct_chg, movement, percentile):
    '''Takes in a list of pct_chg history, calculates the zscore based on the
    distribution of the pct_chg. For example, a zscore of -0.5 translates to a half of one sd below the
    average. Any percentile below 0.5 or 50% results in a negative zscore since it is sd's below average.
    A percentile of 0.5 translates to a zscore of 0.
    A list of percentiles then looked up. A standard zscore for a p-value is determined. It is then
    matched against the z_dict for the pct_chg, and the pct_chg according to that percentile, is returned.'''
    pct_chg_arr = np.array (pct_chg)
    pct_chg_arr = pct_chg_arr [ ~np.isnan (pct_chg_arr) ]
    sd = np.std (pct_chg_arr)
    mean = np.mean (pct_chg_arr)
    zscore = np.divide (np.subtract (pct_chg_arr, mean), sd)  # x - xbar / sd

    z_dict = {}  # zscore as key, pct_chg as value
    for i in range (len (zscore)):
        z_dict.update ([ (np.round (zscore [ i ], 2), np.round (pct_chg_arr [ i ], 2)) ])

    perc_dict = {}
    for p in percentile:
        z = np.round (stats.norm.ppf (p), 2)
        perc_dict [ p ] = z_dict.get (z)
        # print(movement + ' pct chg: ' + str(z_dict.get(z, 0.0)) +'%' + ', p = ' + str(p) + ', z = ' + str(z))

    return perc_dict


def export_trade_stops(trade_stops):
    print ("Setting up pickle...")
    with open ('/home/ubuntu/model_test/model_output/predictions/trade_stops.json', 'wb') as f:
        pickle.dump (trade_stops, f)
    print ("Pickle complete...")


# Import the names of the (sector, equity) and volume dictionary
with open ('/home/ubuntu/model_test/export_files/best_tickers.json', 'rb') as f:
    best_tickers = pickle.load (f)

n_items = {k: best_tickers [ k ] for k in list (best_tickers) [ 0: 5 ]}

percentile_list = [ 0.05, 0.1, 0.2, 0.3, 0.4, 0.5 ]
parameters, trade_stops = {}, {}
count = 1

for (sector, ticker) in best_tickers.keys ():
    try:
        t0 = datetime.now ()
        print ("Start ticker " + str (count) + ": " + ticker)
        scraper, processor = instantiate (ticker)
        stock_data = get_data ()

        close = processor.get_backtest (stock_data.loc [ :, 'close' ].values, n_out)
        low = processor.get_backtest (stock_data.loc [ :, 'low' ].values, n_out)
        dates = processor.get_backtest (stock_data.index.values, n_out)

        close_df = pd.DataFrame (close, columns=[ 'close' + str (x) for x in range (n_out) ])
        low_df = pd.DataFrame ([ min (row) for row in low [ :, 1: ] ],
                               columns=[ '5day_low' ])  # 5 day low from day 1 - which is column 2
        dates_df = pd.DataFrame (dates, columns=[ 'day' + str (x) for x in range (n_out) ])

        actuals = concat_actuals (ticker, close_df, low_df, dates_df)

        pct_chg_low = actuals [ 'low_pct_change' ]
        pct_chg_low_positive = actuals [ actuals.loc [ :, 'actual_pct_change' ] >= 0 ].loc [ :, 'low_pct_change' ]

        z_percentiles = {}
        z_percentiles [ ticker ] = ({'z_low': get_zscore (pct_chg_low, 'Lows Week', percentile_list),
                                     'z_low_positive': get_zscore (pct_chg_low_positive, 'Lows Pos Week',
                                                                   percentile_list)})
        trade_stops.update (
            z_percentiles)  # runs the same function twice for week lows and week lows off a positive week
        print ("End ticker " + str (count) + ": " + ticker)
        count += 1
    except:
        pass

export_trade_stops (trade_stops)
print ("Time Taken:", datetime.now () - t0)

# Import the names of the (sector, equity) and volume dictionary
with open ('/home/ubuntu/model_test/model_output/predictions/trade_stops.json', 'rb') as f:
    trade_stops = pickle.load (f)
