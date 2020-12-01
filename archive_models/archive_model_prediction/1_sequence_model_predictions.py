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

os.environ [ 'TF_CPP_MIN_LOG_LEVEL' ] = '2'


n_input_cls = 15
n_diff = 1
n_out = 5
n_input_seq = 10


def get_models():
    '''Loads the models and the encoder'''
    load_regressor = joblib.load ('/home/ubuntu/model_test/saved_models/rf_regressor.mod')
    load_classifier = joblib.load ('/home/ubuntu/model_test/saved_models/rf_multi.mod')

    # reload the encoder
    pkl_file = open ('/home/ubuntu/model_test/saved_models/rf_multi_encoder.pkl', 'rb')
    rf_multi_encoder = pickle.load (pkl_file)
    pkl_file.close ()

    # Load the sequence model
    cnn_seq_model = load_model ('/home/ubuntu/model_test/saved_models/cnn_seq_model.h5')
    return load_regressor, load_classifier, rf_multi_encoder, cnn_seq_model

def get_data(days):
        #Pulls recent stock data for a specific equity - set up to allow
        #back testing on predictions - by setting the days parameter, the end date
        #will step back one day at a time to fill in missed prediction days
    stock_recent = aapl.get_stock_recent ()
    stock_recent.index = pd.to_datetime (stock_recent.index)

    end_date = stock_recent.index.max () - timedelta (days=days)
    stock_recent = stock_recent [ stock_recent.index <= end_date ]

    print ("Running Cycle: " + str (days + 1))
    print ("Last Date was: " + str (end_date))

    return stock_recent


def process_data(df, n_input_cls=15, n_diff=1, n_out=5, n_input_seq=10):
        '''Takes stock_recent as an input as well as the other static hyperparameters'''
        cls_ds = aapl.process_data (df).drop ([ 'adj close', 'day', 'ticker' ], axis=1).iloc [ -n_input_cls:, : ]
        last_close = pd.Timestamp (cls_ds.index.max ())
        last_close_amount = cls_ds.iloc [ -1:, 0 ]
        cls_reshape = aapl.reshape_dataset (np.array (cls_ds), 1)
        cls_data = cls_reshape.reshape ((cls_reshape.shape [ 0 ] * cls_reshape.shape [ 1 ], cls_reshape.shape [ 2 ]))

        # Finalize the input data
        cls_input_x = cls_data.reshape (1, cls_data.shape [ 0 ] * cls_data.shape [ 1 ])

        # Pulls last 3 years of stock data - then flushes it through get_process() for standard cleaning
        # prepare_variate then prepares a multivariate set X (0), or univariate set X (1)
        # -input-1 to ensure we get the earlier record. One record is lost with differencing.
        seq_ds = aapl.prepare_variate (aapl.process_data (df).iloc [ -n_input_seq - 1:, : ], 0)

        seq_ds_diff = np.array (aapl.get_difference_pct (np.array (seq_ds), n_diff))

        seq_input_x = seq_ds_diff.reshape ((1, seq_ds_diff.shape [ 0 ], seq_ds_diff.shape [ 1 ]))

        return cls_ds, cls_input_x, seq_ds, seq_input_x, last_close, last_close_amount


def get_date_range(df, n_out):
    '''
        Takes in the processed dataset and returns a list of n_out
        business days, beginning from tomorrow (given the next 5 days
        of prediction
        '''
    start = (pd.Timestamp (df.index.max ()) + relativedelta (days=1))
    date = set ()
    for i in range (n_out + 1):  # add an extra day to the set to allow for duplicates caused by Sat and Sun - then remove
        date.add (start + pd.tseries.offsets.BusinessDay (n=i))

    dates = list (date)
    dates.sort ()

    return dates [ :5 ]  # in case 6 are returned - n_out + 1 is necessary for weekly duplicating issue.

'''
aapl = cl.predict_classical ('AAPL')
stock_recent = get_data (1)
end_date = stock_recent.index.max () - timedelta (days=12)
df = stock_recent
df = df [ df.index <= end_date ]
dates = get_date_range(df, 5)
dates
'''

def get_reg_predictions(model, X):
    return model.predict (X)


def get_clf_predictions(model, X, encoder):
    yhat = model.predict (X)
    yhat_proba = model.predict_proba (X)
    yhat_clf = aapl.decode_yhat (yhat, yhat_proba, encoder)
    return yhat_clf


def get_seq_predictions(model, X, X_raw):
    '''
        :param model:
        :param X:
        :param X_raw:
        In order to invert the prediction, the first close is needed from the raw input_x.
        This represent the day before the predictions and the source of the % changes.
        '''

    yhat_sequence = cnn_seq_model.predict (seq_input_x)
    yhat_sequence_n = [ ]  # instantiate each time a new sequence is generated (inside for loop)

    nday_pct_chg = yhat_sequence.sum ()  # As a first value, add the 5 day predicted % change
    day0 = seq_ds.iloc [ 0, 0 ]  # Get day0 value in order to make the week's predictions

    for yhat_pct in yhat_sequence.flatten ():  # extract each % prediction - day 1 - day n_out
        yhat = (yhat_pct * day0) + day0  # add the predicted % change to day0
        yhat_sequence_n.append (yhat)  # add to the undifferenced yhat_sequence_undifferenced
        day0 = yhat  # update day0 to the next predicted day along
    yhat_sequence_n = np.array (yhat_sequence_n)
    yhat_sequence_n = yhat_sequence_n.reshape (1, yhat_sequence_n.shape [ 0 ])

    return yhat_sequence_n, nday_pct_chg


def concat_predictions(ticker, yhat_reg, yhat_clf, yhat_seq, pct_chg_seq, date_range, last_close, last_close_amount):
    yhat_clf [ 'ticker' ] = ticker
    yhat_clf [ 'yhat_reg' ] = yhat_reg
    yhat_clf [ 'start' ] = min (date_range)
    yhat_clf [ 'end' ] = max (date_range)
    yhat_clf [ 'last_close' ] = last_close
    yhat_clf [ 'last_close_amount' ] = last_close_amount.values

    yhat_clf [ 'pct_chg_seq' ] = pct_chg_seq  # n_out days % change
    yhat_seq_df = pd.DataFrame (yhat_seq, columns=[ 'yhat_seq' + str (x) for x in range (1, 6) ])  # n_out days yhat
    yhat_clf = pd.concat ([ yhat_clf, yhat_seq_df ], axis=1)
    yhat_clf [ 'update_dt' ] = str (datetime.now ())
    yhat_clf.set_index ('update_dt', inplace=True)
    return yhat_clf


def export_predictions(preds):
    # Export and Append to model_preds. This file will be created if it
    # does not exist, or will be appended to if it does
    # Creates 2 versions: model_preds.csv and model_preds_deduped.csv
    pathname = r'/home/ubuntu/model_test/model_output/'
    filedupe = 'model_preds.csv'
    filededupe = 'model_preds_deduped.csv'

    with open (pathname + filedupe, 'a') as f:
        preds.to_csv (f, header=f.tell () == 0, float_format='%.3f')

    d = pd.read_csv (pathname + filedupe, keep_default_na=False)
    d.drop (d.columns [ 0 ], axis=1, inplace=True)
    d.drop_duplicates (subset=[ 'start', 'end', 'ticker' ], inplace=True, keep='first')

    df = d
    df = df.reindex (columns=df.columns.tolist () + [ 'updated_end' ])  # add placeholder field
    df.reset_index (drop=True, inplace=True)

    # create an updated end field - if the 4%+ signal persists, the trade length can be extended
    df.updated_end = df.end
    for i in range (1, len (df)):
        for j in range (i, len (df)):
            if (df.loc [ j, 'yhat_label' ] == '4+% Up') & (df.loc [ j - 1, 'yhat_label' ] == '4+% Up'):
                df.loc [ i - 1, 'updated_end' ] = df.loc [ j, 'end' ]

    df = df.sort_values (by=[ 'ticker', 'last_close', 'start' ])
    df.to_csv (pathname + filededupe, float_format='%.3f')  # rounded to two decimals

    return df

load_regressor, load_classifier, rf_multi_encoder, cnn_seq_model = get_models ()

#default days should be 1, in order to capture the most recent day
for days in range (1):
        aapl = cl.predict_classical ('AAPL')
        stock_recent = get_data (days)
        cls_ds, cls_input_x, seq_ds, seq_input_x, last_close, last_close_amount = process_data (stock_recent)
        #cls_ds, cls_input_x, seq_ds, seq_input_x, last_close, last_close_amount = process_data (stock_recent, 15, 1, 5, 10)
        yhat_reg = get_reg_predictions (load_regressor, cls_input_x)
        yhat_clf = get_clf_predictions (load_classifier, cls_input_x, rf_multi_encoder)
        yhat_seq, pct_chg_seq = get_seq_predictions (cnn_seq_model, seq_input_x, seq_ds)
        date_range = get_date_range (cls_ds, n_out)
        predictions = concat_predictions ('AAPL', yhat_reg, yhat_clf, yhat_seq, pct_chg_seq, date_range, last_close,
                                  last_close_amount)
        preds_deduped = export_predictions (predictions)

