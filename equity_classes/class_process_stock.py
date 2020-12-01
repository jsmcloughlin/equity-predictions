import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix, balanced_accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, train_test_split, RandomizedSearchCV, GridSearchCV
import sys
sys.path.append("/home/ubuntu/model_test")
from equity_classes import classes as cl
import joblib
import pickle
from equity_classes import class_prepare_stock as cps
from stockstats import StockDataFrame as sdf
pd.options.mode.chained_assignment = None  # turn of chain warning
from math import isnan
from datetime import datetime

class yfinance_process(cps.yfinance_scrape):

    def __init__(self, ticker):
        self.ticker = ticker
        super(cps.yfinance_scrape).__init__()

    def instantiate(self, ticker):
        '''instantiate 2 class objects for a specific ticker'''
        return cps.yfinance_scrape (ticker), cl.prepare_classical (ticker)

    def scrape_and_process(self, scraper, processor):
        df_scraped = scraper.run_cps ()
        df_processed = processor.process_data (df_scraped)
        df_processed = df_processed.drop ([ 'adj close', 'day', 'ticker' ], axis=1)

        # df_reshape = processor.reshape_dataset(np.array(df_processed), 1)

        return processor.reshape_dataset (np.array (df_processed), 1)

    def prepare_data(df, n_input, n_out):
        X, y = processor.to_supervised_classical (df, 15, 5)

        X_classical = pd.DataFrame (
            processor.reshape_X_classical (X))  # Reshapes X into 1 row and all columns for the features
        y_classical = processor.reshape_y_classical (y, n_out=5)  # Reshapes y to calculate % change

        # work off nday_chg - since it is the % change over the course of the 5 days, not the max % change over the course
        nday_chg, intraday_max = processor.get_chg_pc (y_classical)

        nday_chg_label = pd.DataFrame.from_records (processor.get_chg_pc_label (nday_chg),
                                                    columns=[ 'nday_chg', 'nday_chg_label' ])

        dummy_y, encoded_y, encoder = processor.encode_y (nday_chg_label)

        X_train, X_test, y_train, y_test = train_test_split (X_classical, encoded_y, test_size=0.3, random_state=101)

        parameters = {
            "X_classical": X_classical,
            "y_classical": y_classical,
            "nday_chg_label": nday_chg_label,
            "dummy_y": dummy_y,
            "encoded_y": encoded_y,
            "encoder": encoder,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }

        return parameters

