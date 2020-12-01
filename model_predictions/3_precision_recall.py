import sys
from sklearn.metrics import balanced_accuracy_score

sys.path.append ("/home/ubuntu/model_test")
from equity_classes import classes as cl
import numpy as np
import pandas as pd
from datetime import datetime, date
import pickle
import joblib
from dateutil.relativedelta import relativedelta
import os
from datetime import datetime, timedelta

sys.path.append ("/home/ubuntu/model_test")
from equity_classes import classes as cl
import pickle
from equity_classes import class_prepare_stock as cps

from pathlib import Path

os.environ [ 'TF_CPP_MIN_LOG_LEVEL' ] = '2'

n_input_cls = 15
n_diff = 1
n_out = 6  # has to be set to 6
n_input_seq = 10


def get_50day_preds():
    filename = Path (r'/home/ubuntu/model_test/model_output/predictions/enhanced_models_yhat.csv')

    try:
        print ('enhanced_models_yhat exists. Importing....')
        df = pd.read_csv (filename, parse_dates=[ 'start', 'end', 'updated_end' ])
        df.rename (columns={'start': 'day1', 'end': 'day5'}, inplace=True)
    except:
        print ('enhanced_models_yhat does not exist')

    return df [ df [ 'ticker' ] == ticker ]


'''
def get_preds_todict(row):
    i = str (row.ticker)
    j = str (row.start)
    k = str (row.end)
    preds_dict[ i, j, k ] = row.yhat_label
df_preds.apply (get_preds_todict, axis=1)
'''


def instantiate(ticker):
    '''instantiate 2 class objects for a specific ticker'''
    return cps.yfinance_scrape (ticker), cl.prepare_classical (ticker)


def get_data():
    '''Pulls recent stock data for a specific equity'''
    return scraper.run_cps ()


def get_temporal_subset(df, days):
    '''Takes the input from get_data() and returns a subset based on recency. It uses the last 50 trading days
    (70 calendar) to calculate current Precision and Recall - which was held out from the model'''

    end = date.today ()
    end = datetime (end.year, end.month, end.day)  # to midnight of that day

    start = (end - relativedelta (days=days))

    return df [ df.index > start ]


def concat_actuals(ticker, close, dates):
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

    close [ 'ticker' ] = ticker

    actuals = pd.concat ([ dates [ [ 'day1', 'day5' ] ], close, actual_pct_chg_label ], axis=1, sort=False)

    floats = [ 'close0', 'close1', 'close2', 'close3', 'close4', 'close5', 'actual_pct_change' ]
    for colname in floats:
        actuals [ colname ] = pd.to_numeric (actuals [ colname ], errors='coerce')

    actuals = actuals.drop ([ 'close0', 'close1', 'close2', 'close3', 'close4', 'close5' ], axis=1)

    return actuals


def prepare_comparison(ticker, preds, actuals):
    df = pd.merge (preds, actuals, on=[ 'day1', 'day5' ])
    df [ 'ticker' ] = ticker
    df = df [ [ 'ticker', 'day1', 'day5', 'yhat_label', 'actual_pct_chg_label' ] ]
    df.rename (columns={'yhat_label': 'pred', 'actual_pct_chg_label': 'actual'}, inplace=True)

    # Soft Precision and Recall - pred is '4+% Up' and the actual is '1-4% Up', this is soft precision.
    # It is a safe False Positive since it still represents a positive 5 day stretch on the market.
    df_soft = df.copy ()
    df_soft [ 'actual' ] = np.where ((df_soft.actual == '1-4% Up'), '4+% Up', df_soft.actual)

    # Calculate Precision Recall for '4+% Up' (Actual) against '4+% Up' (Pred) and Soft Precision
    precision_recall_dict = {}

    precision_recall_dict [ str (ticker)] = \
        processor.precision_recall (processor.get_4pct_accuracy (df)), \
        processor.precision_recall (processor.get_4pct_accuracy (df_soft)) [ 0 ]

    # Round the Precision, Recall and Soft Precision numbers
    for i, ((p, r), r_s) in precision_recall_dict.items ():
        precision_recall_dict [ i ] = ((round (p, 2), round (r, 2)), round (r_s, 2))

    print ('Balanced Accuracy Score (Overall): \n' + str (
        round (balanced_accuracy_score (df [ 'actual' ], df [ 'pred' ]), 2)))
    print (precision_recall_dict)
    print ('Ticker: ' + str (ticker) + '\n' + str (
        pd.crosstab (df [ 'actual' ], df [ 'pred' ], rownames=[ 'Actual' ], colnames=[ 'Predicted' ])))

    return precision_recall_dict

def export_output(recent_prec_recall):
    print ("Setting up pickle...")
    with open ('/home/ubuntu/model_test/model_output/predictions/recent_prec_recall.json', 'wb') as f:
        pickle.dump (recent_prec_recall, f)
    print ("Pickle complete...")


# Import the names of the (sector, equity) and volume dictionary
with open ('/home/ubuntu/model_test/export_files/best_tickers.json', 'rb') as f:
    best_tickers = pickle.load (f)

n_items = {k: best_tickers [ k ] for k in list (best_tickers) [ 0: 5 ]}

count = 1
recent_prec_recall = [ ]
for (sector, ticker) in best_tickers.keys ():
    try:
        t0 = datetime.now ()

        print ("Start ticker " + str (count) + ": " + ticker)
        df_preds = get_50day_preds ()
        scraper, processor = instantiate (ticker)

        df = get_temporal_subset (get_data (), 100) #this can be set to whatever number
        close_df = pd.DataFrame (processor.get_backtest (df.loc [ :, 'close' ].values, n_out),
                                 columns=[ 'close' + str (x) for x in range (n_out) ])
        dates_df = pd.DataFrame (processor.get_backtest (df.index.values, n_out),
                                 columns=[ 'day' + str (x) for x in range (n_out) ])
        df_actuals = concat_actuals (ticker, close_df, dates_df)

        recent_prec_recall.append (prepare_comparison (ticker, df_preds, df_actuals))

        print ("End ticker " + str (count) + ": " + ticker)
        count += 1
    except:
        pass
export_output (recent_prec_recall)
print ("Time Taken:", datetime.now () - t0)
