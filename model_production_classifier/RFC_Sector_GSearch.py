import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix, \
    balanced_accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, train_test_split, RandomizedSearchCV, \
    GridSearchCV
import sys

sys.path.append ("/home/ubuntu/model_test")
from equity_classes import classes as cl
import joblib
import pickle
from equity_classes import class_prepare_stock as cps
from stockstats import StockDataFrame as sdf

pd.options.mode.chained_assignment = None  # turn of chain warning
from math import isnan
from datetime import datetime

'''The purpose of this note is to grid search the best performing stocks
from the RFC_Sector_CV - to see if we can improve on the performance'''


def instantiate(ticker):
    '''instantiate 2 class objects for a specific ticker'''
    return cps.yfinance_scrape (ticker), cl.prepare_classical (ticker)


def scrape_and_process():
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


def run_model(parameters):
    X_classical = parameters [ 'X_classical' ]
    y_classical = parameters [ 'y_classical' ]
    nday_chg_label = parameters [ 'nday_chg_label' ]
    dummy_y = parameters [ 'dummy_y' ]
    encoded_y = parameters [ 'encoded_y' ]
    encoder = parameters [ 'encoder' ]
    X_train = parameters [ 'X_train' ]
    X_test = parameters [ 'X_test' ]
    y_train = parameters [ 'y_train' ]
    y_test = parameters [ 'y_test' ]

    # Number of trees in random forest
    n_estimators = [ int (x) for x in np.linspace (start=200, stop=2000, num=10) ]
    # Number of features to consider at every split
    max_features = [ 'auto', 'sqrt' ]
    # Maximum number of levels in tree
    max_depth = [ int (x) for x in np.linspace (10, 110, num=11) ]
    max_depth.append (None)
    # Minimum number of samples required to split a node
    min_samples_split = [ 2, 5, 10 ]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [ 1, 2, 4 ]
    # Method of selecting samples for training each tree
    bootstrap = [ True, False ]
    # Create the random grid

    n_components = [ 20, 30, 50, 100, 300,
                     330 ]  # this is for PCA() - best course of action appears to be normalization of the data


    class_weights = compute_class_weight ('balanced', np.unique (y_train), y_train)
    #class_weights = compute_class_weight ('balanced', np.unique (encoded_y), encoded_y)
    weights_dict = dict (enumerate (class_weights))
    # class_weights = compute_class_weight ('balanced', np.unique (encoded_y), encoded_y)
    # weights_dict = dict (enumerate (class_weights))

    rf_classifier = RandomForestClassifier (criterion='entropy',
                                            random_state=42, class_weight=weights_dict)

    rf_pipeline = Pipeline (steps=
    [
        ('scaler', RobustScaler ()),
        # ('pca', PCA()),
        ('kc', rf_classifier)
    ])

    random_grid = {
        'kc__n_estimators': n_estimators,
        'kc__max_features': max_features,
        'kc__max_depth': max_depth,
        'kc__min_samples_split': min_samples_split,
        'kc__min_samples_leaf': min_samples_leaf,
        'kc__bootstrap': bootstrap}

    rf_grid = RandomizedSearchCV (estimator=rf_pipeline,
                                  cv=3,
                                  param_distributions=random_grid,
                                  n_iter=50,
                                  verbose=0,
                                  random_state=42,
                                  n_jobs=1)

    #rf_grid.fit (X_classical, encoded_y)
    rf_grid.fit (X_train, y_train)
    # processor.get_results (rf_grid, X_test, y_test, encoder)

    print ("Best: %f using %s" % (rf_grid.best_score_, rf_grid.best_params_))

    return rf_grid.best_score_, rf_grid.best_params_


def remove_nans(dict, cutoff):
    new_dict = {}
    for key, values in dict.items ():
        i, j = values
        if not isnan (i) and i > cutoff:
            new_dict [ key ] = values

    return new_dict


# Import the names of the (sector, equity) and volume dictionary
with open ('/home/ubuntu/model_test/export_files/best_tickers.json', 'rb') as f:
    best_tickers = pickle.load (f)

# sectors = np.unique(list([s for s, t in best_tickers.keys()]))
#import gc
#collected = gc.collect()

n_items = {k: best_tickers [ k ] for k in list (best_tickers) [ -1: ]}


#ticker = 'AAPL'
#sector = 'technology'

#Set Up AAPL grid search
#best_tickers[('technology', 'AAPL')] = (55.12055091303003, 2.146965390375466)
#n_items = {(sector, ticker): acc_sd for (sector, ticker), acc_sd in best_tickers.items() if ticker == 'AAPL'}
count = 1
#for (sector, ticker) in n_items.keys():
for (sector, ticker) in best_tickers.keys ():
    try:
        grid_dict = {}
        t0 = datetime.now ()
        print ("Start ticker " + str (count) + ": " + ticker)
        scraper, processor = instantiate (ticker)
        df_reshape = scrape_and_process ()
        parameters = prepare_data (df_reshape, 15, 5)
        grid_dict [ (sector, ticker) ] = run_model (parameters)
        with open ('/home/ubuntu/model_test/export_files/grid_results/' + str (ticker) + '_grid_results.json',
                   'wb') as f:
            pickle.dump (grid_dict, f)
        print ("End ticker " + str (count) + ": " + ticker)
        print ("Time Taken:", datetime.now () - t0)
        count += 1
    except:
        pass

# with open('/home/ubuntu/model_test/export_files/grid_dict_results.json', 'wb') as f:
#  pickle.dump(grid_dict, f)

# with open('/home/ubuntu/model_test/export_files/grid_dict_results' + str(count) + '.json', 'wb') as f:
# Import the names of the (sector, equity) and volume dictionary

# with open ('/home/ubuntu/model_test/export_files/grid_results/grid_dict_results2.json', 'rb') as f:
#    grid_tickers = pickle.load (f)
#    grid_tickers

