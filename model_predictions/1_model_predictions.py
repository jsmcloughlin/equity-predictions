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
from pathlib import Path

os.environ [ 'TF_CPP_MIN_LOG_LEVEL' ] = '2'

n_input_cls = 15
n_diff = 1
n_out = 5
n_input_seq = 10


def get_models(ticker):
    '''Imports a specific model and encoder
    '''
    try:
        model = joblib.load ('/home/ubuntu/model_test/model_saved/grid_models/classifier_' + str (ticker) + '.mod')
        # reload the encoder
        pkl_file = open ('/home/ubuntu/model_test/model_saved/grid_encoders/encoder_' + str (ticker) + '.pkl', 'rb')
        encoder = pickle.load (pkl_file)
        pkl_file.close ()

    except:
        pass

    return model, encoder


def instantiate(ticker):
    '''instantiate 2 class objects for a specific ticker'''
    return cps.yfinance_scrape (ticker), cl.prepare_classical (ticker)


def get_data():
    '''Pulls recent stock data for a specific equity'''
    return scraper.run_cps ()

def get_filter_data(df, days):
    '''Filters the output from get_data() for a specific number of days'''

    df.index = pd.to_datetime (df.index)

    end_date = df.index.max () - timedelta (days=days)
    df = df [ df.index <= end_date ]

    print ("Running Cycle: " + str (days + 1))
    print ("Last Date was: " + str (end_date))

    return df

def process_data(df, n_input_cls=15, n_diff=1, n_out=5, n_input_seq=10):
    '''Takes stock_recent as an input as well as the other static hyperparameters'''

    cls_ds = processor.process_data (df).drop ([ 'adj close', 'day', 'ticker' ], axis=1).iloc [ -n_input_cls:, : ]
    last_close = pd.Timestamp (cls_ds.index.max ())
    last_close_amount = cls_ds.iloc [ -1:, 0 ]
    cls_reshape = processor.reshape_dataset (np.array (cls_ds), 1)
    cls_data = cls_reshape.reshape ((cls_reshape.shape [ 0 ] * cls_reshape.shape [ 1 ], cls_reshape.shape [ 2 ]))

    # Finalize the input data
    cls_input_x = cls_data.reshape (1, cls_data.shape [ 0 ] * cls_data.shape [ 1 ])

    # Pulls last 3 years of stock data - then flushes it through get_process() for standard cleaning
    # prepare_variate then prepares a multivariate set X (0), or univariate set X (1)
    # -input-1 to ensure we get the earlier record. One record is lost with differencing.
    seq_ds = processor.prepare_variate (processor.process_data (df).iloc [ -n_input_seq - 1:, : ], 0)

    seq_ds_diff = np.array (processor.get_difference_pct (np.array (seq_ds), n_diff))

    seq_input_x = seq_ds_diff.reshape ((1, seq_ds_diff.shape [ 0 ], seq_ds_diff.shape [ 1 ]))

    return cls_ds, cls_input_x, seq_ds, seq_input_x, last_close, last_close_amount

#yhat = classifier.predict (cls_input_x)
#yhat_proba = classifier.predict_proba (cls_input_x)
#yhat_clf = processor.decode_yhat (yhat, yhat_proba, encoder)

def get_clf_predictions(model, X, encoder):
    yhat = model.predict (X)
    yhat_proba = model.predict_proba (X)
    yhat_clf = processor.decode_yhat (yhat, yhat_proba, encoder)
    return yhat_clf


def get_date_range(df, n_out):
    '''
        Takes in the processed dataset and returns a list of n_out
        business days, beginning from tomorrow (given the next 5 days
        of prediction
    '''
    start = (pd.Timestamp (df.index.max ()) + relativedelta (days=1))
    date = set ()
    for i in range (
            n_out + 1):  # add an extra day to the set to allow for duplicates caused by Sat and Sun - then remove
        date.add (start + pd.tseries.offsets.BusinessDay (n=i))

    dates = list (date)
    dates.sort ()

    return dates [ :5 ]  # in case 6 are returned - n_out + 1 is necessary for weekly duplicating issue.


'''
def concat_predictions(yhat_dict, sector, ticker, yhat_clf, date_range, last_close, last_close_amount):
    yhat_dict[(sector, ticker, min(date_range), max(date_range))] = \
        {
        'yhat_proba': yhat_clf.yhat_proba,
        'yhat_label': yhat_clf.yhat_label,
        'last_close': last_close,
        'last_close_amount': last_close_amount,
        'update_dt': str (datetime.now ())
        }
    return yhat_dict
'''


def concat_predictions(sector, ticker, yhat_clf, date_range, last_close, last_close_amount):
    yhat_clf [ 'sector' ] = sector
    yhat_clf [ 'ticker' ] = ticker
    yhat_clf [ 'start' ] = min (date_range)
    yhat_clf [ 'end' ] = max (date_range)
    yhat_clf [ 'last_close' ] = last_close
    yhat_clf [ 'last_close_amount' ] = last_close_amount.values

    yhat_clf [ 'update_dt' ] = str (datetime.now ())
    yhat_clf.set_index ('update_dt', inplace=True)
    return yhat_clf


def export_predictions(predictions, ticker):
    # Export and Append to model_preds. This file will be created if it
    # does not exist, or will be appended to if it does
    # Creates 2 versions: model_preds.csv and model_preds_deduped.csv
    # filename = Path (r'/home/ubuntu/model_test/model_output/predictions/' + str (ticker) + '_model_preds.csv')
    filename = Path (r'/home/ubuntu/model_test/model_output/predictions/models_yhat.csv')

    if filename.is_file ():
        print (str (ticker) + '_model_preds.csv exists. Appending....')
        df = pd.read_csv (filename, keep_default_na=False)
        df.drop('updated_end', axis=1, inplace=True)
        df = df.append (predictions, ignore_index=True)
    else:
        print (str (ticker) + '_model_preds.csv does not currently exist. Creating....')
        df = predictions
        #df = df.reindex (columns=df.columns.tolist () + [ 'updated_end' ])

    df [ 'updated_end' ] = df [ 'end' ]
    dates = [ 'start', 'end', 'last_close', 'updated_end' ]
    for d in dates:
        df [ d ] = pd.to_datetime (df [ d ]).dt.date

    df.drop_duplicates (subset=[ 'ticker', 'last_close' ], inplace=True, keep='first')
    df.to_csv (filename, float_format='%.3f', index=False)  # rounded to two decimals

    return df

def update_update_end():
    # This logic is not working right now....
    #The goal is to update the update_end field if signals (4%+ persists)

    import_file = Path (r'/home/ubuntu/model_test/model_output/predictions/models_yhat.csv')
    export_file = Path (r'/home/ubuntu/model_test/model_output/predictions/enhanced_models_yhat.csv')
    df_import = pd.read_csv (import_file, keep_default_na=False)

    df = pd.DataFrame ()
    grouped = df_import.groupby ('ticker')
    for name, group in grouped:
        group [ 'updated_end' ] = group [ 'end' ]
        group = group.sort_values (by=[ 'ticker', 'last_close', 'start' ])
        group = group.reset_index (drop=True)

        for i in range (len (group)):
            for j in range (1, len (group)):
                if (group.loc [ j, 'yhat_label' ] == '4+% Up') & (group.loc [ j - 1, 'yhat_label' ] == '4+% Up'):
                    group.loc [ j - 1, 'updated_end' ] = group.loc [ j, 'end' ]
        df = df.append (group, ignore_index=True)

    df.to_csv (export_file, float_format='%.3f', index=False)  # rounded to two decimals
    return df

# End Predictions

# Import the names of the (sector, equity) and volume dictionary
with open ('/home/ubuntu/model_test/export_files/best_tickers.json', 'rb') as f:
    best_tickers = pickle.load (f)

n_items = {k: best_tickers [ k ] for k in list (best_tickers) [ -1: ]}


for (sector, ticker) in best_tickers.keys ():
#for (sector, ticker) in n_items.keys ():
    preds = pd.DataFrame ()
    for days in range(1): #only needs to have a range of 1 after the initial set up of 10 days or so
        print(ticker + " " + str(days))
        try:
            t0 = datetime.now ()
            classifier, encoder = get_models (ticker)
            print("Start Pass " + str(days) + ": " + ticker)
            scraper, processor = instantiate (ticker)
            stock_filter = get_filter_data (get_data (), days) #filters output from get_data()
            cls_ds, cls_input_x, seq_ds, seq_input_x, last_close, last_close_amount = process_data (stock_filter)
            yhat = get_clf_predictions (classifier, cls_input_x, encoder)
            date_range = get_date_range (cls_ds, n_out)
            predictions = concat_predictions (sector, ticker, yhat, date_range, last_close, last_close_amount)
            preds = preds.append(predictions)
            preds.drop_duplicates (subset=[ 'ticker', 'last_close' ], inplace=True, keep='first')
            _ = export_predictions (preds, ticker)
        except:
            pass
print ("Time Taken:", datetime.now () - t0)
_ = update_update_end() #run this after everything else
