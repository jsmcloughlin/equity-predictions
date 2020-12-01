from subprocess import call
import sys
#sys.path.append("/home/ubuntu/model_testing")
#sys.path.append('/tmp/pycharm_project_765/venv/lib/python3.8/site-packages')
#sys.path.append('/home/ubuntu/.local/lib/python3.7/site-packages')
from equity_classes import classes as cl
import numpy as np
import pandas as pd
from datetime import datetime, date
import pickle
import joblib
from dateutil.relativedelta import relativedelta
from numpy import loadtxt
from keras.models import load_model

import sys
print('\n'.join(sys.path))


n_input = 15
n_out = 5

def get_data():

        stock_recent = aapl.get_stock_recent()
        #stock = stock_recent[stock_recent.loc[:, 'ticker'] == ticker]

        # Process, drop columns and retain the last n_input days
        dataset = aapl.process_data(stock_recent).drop(['adj close', 'day', 'ticker'], axis=1).iloc[-n_input:, :]
        df_reshape = aapl.reshape_dataset(np.array(dataset), 1)
        data = df_reshape.reshape((df_reshape.shape[0] * df_reshape.shape[1], df_reshape.shape[2]))

        # Finalize the input data
        input_x = data.reshape(1, data.shape[0] * data.shape[1])
        return dataset, input_x


def get_date_range(df, ndays=5):
        '''Takes in the processed dataset and returns the prediction
        date range start date (yesterday + 1 (today)) and end date
        (today + 5 business days).'''
        start = (pd.Timestamp(df.index.max()) + relativedelta (days=1))
        end = start + pd.tseries.offsets.BusinessDay(n=5)

        return start, end

def get_models():
        '''Loads the models and the encoder'''
        load_regressor = joblib.load('/home/ubuntu/stock_lstm/saved_models/rf_regressor.mod')
        load_classifier = joblib.load('/home/ubuntu/stock_lstm/saved_models/rf_multi.mod')

        # reload the encoder
        pkl_file = open('/home/ubuntu/model_testing/saved_models/rf_multi_encoder.pkl', 'rb')
        rf_multi_encoder = pickle.load(pkl_file)
        pkl_file.close()

        # Load the sequence model
        #ccnn_seq_model = load_model('/home/ubuntu/model_testing/saved_models/cnn_seq_model.h5')
        return load_regressor, load_classifier, rf_multi_encoder #, cnn_seq_model

def get_reg_predictions(model, X):
        return model.predict(X)

def get_clf_predictions(model, X, encoder):
        yhat = model.predict(X)
        yhat_proba = model.predict_proba(X)
        yhat_clf = aapl.decode_yhat(yhat, yhat_proba, encoder)
        return yhat_clf

def concat_predictions(ticker, yhat_reg, yhat_clf, start, end):
        yhat_clf['yhat_reg'] = yhat_reg
        yhat_clf['start'] = start
        yhat_clf['end'] = end
        yhat_clf['ticker'] = ticker
        yhat_clf['update_date'] = str(datetime.now())
        return yhat_clf

def export_predictions(preds):
        # Export and Append to model_preds. This file will be created if it
        # does not exist, or will be appended to if it does
        #Creates 2 versions: model_preds.csv and model_preds_deduped.csv
        pathname = '/home/ubuntu/model_testing/model_output/'
        filedupe = 'model_preds.csv'
        filededupe = 'model_preds_deduped.csv'

        with open(pathname + filedupe, 'a') as f:
                preds.to_csv(f, header=f.tell() == 0)

        d = pd.read_csv(pathname + filedupe, keep_default_na=False)
        d.drop(d.columns[0], axis=1, inplace=True)
        d.drop_duplicates(subset=['start', 'end', 'ticker'], inplace=True, keep='first')

        d.to_csv(pathname + filededupe, float_format='%.4f')  # rounded to two decimals

        return d


aapl = cl.predict_classical('AAPL')  # instantiate the object
dataset, input_x = get_data()
load_regressor, load_classifier, rf_multi_encoder = get_models()
yhat_reg = get_reg_predictions(load_regressor, input_x)
yhat_clf = get_clf_predictions(load_classifier, input_x, rf_multi_encoder)
start, end = get_date_range(dataset)
predictions = concat_predictions('AAPL', yhat_reg, yhat_clf, start, end)

