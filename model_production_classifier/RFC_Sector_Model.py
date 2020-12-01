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


# After testing the Precision, Recall and Accuracy, this code needs to be recompiled
# and applied to all of the training data.

def get_grid_results(ticker, sec):
    '''Cleans a sector and stock tuple from best_tickers and extracts the params
    '''
    grid_results, model_params = {}, {}
    # for (sec, ticker), (acc, std) in best_tickers.items():
    try:
        file = open ('/home/ubuntu/model_test/export_files/grid_results/' + str (ticker) + '_grid_results.json', 'rb')
        grid_results [ ticker ] = pickle.load (file)

        model_params [ (sec, ticker) ] = (grid_results [ ticker ] [ sec, ticker ] [ 1 ] [ 'kc__n_estimators' ],
                                          grid_results [ ticker ] [ sec, ticker ] [ 1 ] [ 'kc__min_samples_split' ],
                                          grid_results [ ticker ] [ sec, ticker ] [ 1 ] [ 'kc__min_samples_leaf' ],
                                          grid_results [ ticker ] [ sec, ticker ] [ 1 ] [ 'kc__max_features' ],
                                          grid_results [ ticker ] [ sec, ticker ] [ 1 ] [ 'kc__max_depth' ],
                                          grid_results [ ticker ] [ sec, ticker ] [ 1 ] [ 'kc__bootstrap' ])
    except:
        pass

    return model_params, grid_results


def instantiate(ticker):
    '''instantiate 2 class objects for a specific ticker'''
    return cps.yfinance_scrape (ticker), cl.prepare_classical (ticker)


def get_data():
    '''Pulls recent stock data for a specific equity'''
    return scraper.run_cps ()


def get_temporal_subset(df, days):
    '''Final Model Training is done on the last 8 years up to 50 trading days ago (70 calendar
    days). The reason is to have a hold out sample for estimating unseen Precision and Recall'''
    today = date.today ()
    today = datetime (today.year, today.month, today.day)  # to midnight of that day

    end = (today - relativedelta (days=days))

    return df [ df.index < end ]

def get_process(df):

    df_processed = processor.process_data (df)
    df_processed = df_processed.drop ([ 'adj close', 'day', 'ticker' ], axis=1)

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

def run_model(parameters, model_params):
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

    class_weights = compute_class_weight ('balanced', np.unique (encoded_y), encoded_y)
    # class_weights = compute_class_weight ('balanced', np.unique (encoded_y), encoded_y)
    weights_dict = dict (enumerate (class_weights))


    for (sec, ticker), params in model_params.items ():
        n_estimators, min_samples_split, min_samples_leaf, max_features, max_depth, bootstrap = params

    final_rf_classifier = RandomForestClassifier \
            (
            bootstrap=bootstrap,
            criterion='entropy',
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            n_estimators=n_estimators,
            random_state=42,
            class_weight=weights_dict
        )

    final_rf_pipeline = Pipeline ([
        ('scaler', RobustScaler ()),
        ('kc', final_rf_classifier)
    ], verbose=True)

    # Fit the final model
    return final_rf_pipeline.fit (X_classical, encoded_y), encoder


def plot_results(model, parameters, cls, ticker, sector):
    X_classical = parameters [ 'X_classical' ]
    y_classical = parameters [ 'y_classical' ]
    encoder = parameters [ 'encoder' ]
    X_test = parameters [ 'X_test' ]
    y_test = parameters [ 'y_test' ]

    yhat_inv = cls.decode_yhat (model.predict (X_test), model.predict_proba (X_test), encoder)  # Predictions
    y_inv = cls.decode_y (y_test, encoder)  # Actuals
    df = pd.concat ([ yhat_inv, y_inv ], axis=1).drop ([ 'yhat_proba' ], axis=1)
    df.rename (columns={'yhat_label': 'pred', 'y_label': 'actual'}, inplace=True)

    # Soft Precision and Recall - pred is '4+% Up' and the actual is '1-4% Up', this is soft precision.
    # It is a safe False Positive since it still represents a positive 5 day stretch on the market.
    df_soft = df.copy ()
    df_soft [ 'actual' ] = np.where ((df_soft.actual == '1-4% Up'), '4+% Up', df_soft.actual)

    # Calculate Precision Recall for '4+% Up' (Actual) against '4+% Up' (Pred) and Soft Precision
    precision_recall_dict = {}
    #precision_recall_dict [ str (sector) + ' ' + str (ticker) + ' prec-recall' ] = \
    #    precision_recall (get_4pct_accuracy (df)), precision_recall (get_4pct_accuracy (df_soft)) [ 0 ]

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


def export_results(ticker, model, encoder):
    # Dump the saved model and the encoder
    joblib.dump (model, '/home/ubuntu/model_test/saved_models/grid_models/classifier_' + str (ticker) + '.mod')

    # exporting the departure encoder
    output = open ('/home/ubuntu/model_test/saved_models/grid_encoders/encoder_' + str (ticker) + '.pkl', 'wb')
    pickle.dump (encoder, output)
    output.close ()

    # Import the names of the (sector, equity) and volume dictionary


with open ('/home/ubuntu/model_test/export_files/best_tickers.json', 'rb') as f:
    best_tickers = pickle.load (f)


n_items = {k: best_tickers [ k ] for k in list (best_tickers) [ -1: ]}
precision_recall_list = [ ]
count = 1
for (sector, ticker) in best_tickers.keys ():
#for (sector, ticker) in n_items.keys ():
    print (ticker)
    #try:
        t0 = datetime.now ()
        model_params, _ = get_grid_results (ticker, sector)
        print ("Start ticker " + str (count) + ": " + ticker)
        scraper, processor = instantiate (ticker)
        df = get_data()
        df = get_temporal_subset(df, 1) #trains up to last n calendar (trading) days for holdout
        df_reshape = get_process (df)
        parameters = prepare_data (df_reshape, 15, 5)
        model, encoder = run_model (parameters, model_params)
        precision_recall_list.append (plot_results (model, parameters, processor, ticker, sector))
        export_results (ticker, model, encoder)
        print ("End ticker " + str (count) + ": " + ticker)
        count += 1
    #except:
        #pass
print ("Time Taken:", datetime.now () - t0)

# with open ('/home/ubuntu/model_test/saved_models/grid_models/precision_recall_list.pkl', 'wb') as f:
#    pickle.dump (precision_recall_list, f)

# with open('/home/ubuntu/model_test/saved_models/grid_models/precision_recall_list.pkl', 'rb') as f:
#    precision_recall_list = pickle.load(f)


# removal_tickers = {(sector, ticker): acc_sd for (sector, ticker), acc_sd in best_tickers.items() if ticker not in win_ticker_list}
# removal_tickers
