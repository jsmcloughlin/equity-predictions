import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from dateutil.relativedelta import relativedelta
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta, date
import sys

sys.path.append ("/home/ubuntu/model_test")
from equity_classes import classes as cl
import joblib
import pickle
from equity_classes import class_prepare_stock as cps
from stockstats import StockDataFrame as sdf

pd.options.mode.chained_assignment = None  # turn of chain warning
from datetime import datetime
from collections import ChainMap
from pathlib import Path

def get_imports():
    '''Cleans a sector and stock tuple from best_tickers and extracts the params
    '''
    prec_rec = open ('/home/ubuntu/model_test/model_output/predictions/recent_prec_recall.json', 'rb')
    prec_rec_import = pickle.load (prec_rec)

    trade_stops = open ('/home/ubuntu/model_test/model_output/predictions/trade_stops.json', 'rb')
    tstops_import = pickle.load (trade_stops)

    preds = Path (r'/home/ubuntu/model_test/model_output/predictions/enhanced_models_yhat.csv')
    preds_import = pd.read_csv (preds, keep_default_na=False)

    with open ('/home/ubuntu/model_test/export_files/best_tickers.json', 'rb') as f:
        best_tickers = pickle.load (f)

    return prec_rec_import, \
           tstops_import, \
           preds_import.set_index ('ticker'), \
           best_tickers


def get_precision_recall(prec_rec_import):
    '''Takes the imported precision-recall dictionary
    and ultimately transforms it to a dataframe - where
    the ticker finally becomes the index.'''
    new_prec_rec = {}
    df = pd.DataFrame()
    x = dict (ChainMap (*prec_rec_import [ ::-1 ]))
    for k, v in x.items():
        new_prec_rec = {'ticker': k,
                        'precision': v[0][0],
                        'recall': v[0][1],
                        'soft_precision': v[1]
                        }
        temp_df = pd.DataFrame([new_prec_rec])
        df = df.append(temp_df)

    return df.set_index ('ticker')



def get_trade_stops(tstops_import):
    chain = ChainMap (tstops_import)
    df = pd.DataFrame()

    for sec, ticker in best_tickers:
        zlow = pd.DataFrame ([ chain.get(ticker)['z_low']])
        zlow.columns = [ 'zl_' + str (col) for col in zlow.columns ]

        zlow_pos = pd.DataFrame ([ chain.get (ticker) [ 'z_low_positive' ] ])
        zlow_pos.columns = [ 'zlp_' + str (col) for col in zlow_pos.columns ]

        result = pd.concat ([ zlow, zlow_pos ], axis=1, sort=False)
        result['ticker'] = ticker
        df = df.append (result)

    return df.set_index ('ticker')

def get_daily_report(preds, prec_rec, tstops):

    preds = preds.sort_values ([ 'ticker', 'end' ]).groupby ('ticker').tail (1)
    prec_rec = prec_rec.sort_values('ticker')
    tstops = tstops.sort_values ('ticker')
    daily_all_report = pd.concat([preds, tstops, prec_rec], axis=1)
    daily_4pc_report = daily_all_report[daily_all_report['yhat_label'] == '4+% Up']

    #daily_all_report = daily_all_report.reset_index (inplace=True)
    #daily_4pc_report = daily_4pc_report.reset_index (inplace=True)

    export_all = Path (r'/home/ubuntu/model_test/model_output/daily_all_report.csv')
    export_4pc = Path (r'/home/ubuntu/model_test/model_output/daily_4pc_report.csv')

    daily_all_report.to_csv (export_all, float_format='%.3f', index=True)  # rounded to two decimals
    daily_4pc_report.to_csv (export_4pc, float_format='%.3f', index=True)

    return daily_all_report, daily_4pc_report


prec_rec_import, tstops_import, preds_import, best_tickers = get_imports()
prec_rec_df = get_precision_recall(prec_rec_import)
tstops_df = get_trade_stops(tstops_import)
daily_all_report, daily_4pc_report = get_daily_report(preds_import, prec_rec_df, tstops_df)

