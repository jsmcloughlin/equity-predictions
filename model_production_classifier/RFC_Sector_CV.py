import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix, balanced_accuracy_score, make_scorer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, train_test_split, RandomizedSearchCV, GridSearchCV
import sys
sys.path.append("/home/ubuntu/model_testing")
from equity_classes import classes as cl
import joblib
import pickle
from equity_classes import class_prepare_stock as cps
from stockstats import StockDataFrame as sdf
pd.options.mode.chained_assignment = None  # turn of chain warning
from sortedcontainers import SortedDict
from math import isnan

'''The purpose of this note is to cross fold validate stocks with > 1m
trading volume per day average over the past month.
The best performers from each sector will then have a grid search performed on
them - and a prediction component to determine Precision and Recall of the +4% weeks.
Finally, followed by a final model build for production.

Code will pull tickers and sectors and run through an nfold cv search'''


def instantiate(ticker):
    '''instantiate 2 class objects for a specific ticker'''
    return cps.yfinance_scrape(ticker), cl.prepare_classical(ticker)

def scrape_and_process():

    df_scraped = scraper.run_cps()
    df_processed = processor.process_data(df_scraped)
    df_processed = df_processed.drop(['adj close', 'day', 'ticker'], axis=1)

    #df_reshape = processor.reshape_dataset(np.array(df_processed), 1)

    return processor.reshape_dataset(np.array(df_processed), 1)

def prepare_data(df, n_input, n_out):
    X, y = processor.to_supervised_classical(df, 15, 5)

    X_classical = pd.DataFrame(processor.reshape_X_classical(X)) #Reshapes X into 1 row and all columns for the features
    y_classical = processor.reshape_y_classical(y, n_out=5) #Reshapes y to calculate % change

    #work off nday_chg - since it is the % change over the course of the 5 days, not the max % change over the course
    nday_chg, intraday_max = processor.get_chg_pc(y_classical)

    nday_chg_label = pd.DataFrame.from_records(processor.get_chg_pc_label(nday_chg), columns = ['nday_chg', 'nday_chg_label'])

    dummy_y, encoded_y, encoder = processor.encode_y(nday_chg_label)

    X_train, X_test, y_train, y_test = train_test_split(X_classical, encoded_y, test_size=0.3, random_state=101)


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
    X_classical = parameters['X_classical']
    y_classical = parameters['y_classical']
    nday_chg_label = parameters['nday_chg_label']
    dummy_y = parameters['dummy_y']
    encoded_y = parameters['encoded_y']
    encoder = parameters['encoder']

    class_weights = compute_class_weight('balanced', np.unique(encoded_y), encoded_y)
    weights_dict = dict(enumerate(class_weights))

    # wrap the model using the function you created
    rfc = RandomForestClassifier(class_weight=weights_dict)
    scaler = RobustScaler ()

    rfc_pipeline = Pipeline ([
        ('preprocess', scaler),
        ('rfc', rfc)
    ])

    kfold = KFold(n_splits=5, shuffle=True)
    results = cross_val_score(rfc_pipeline, X_classical, encoded_y, cv=kfold)
    print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

    return results.mean()*100, results.std()*100

def remove_nans(dict, cutoff):
    '''Remove nans from the output'''
    new_dict = {}
    for key, values in results_dict.items ():
        i, j = values
        if not isnan (i) and i > cutoff:
            new_dict [ key ] = values

    return new_dict


#Import the names of the (sector, equity) and volume dictionary
#Important here that there is enough training data for each stock - minimum of 5 years
with open('/home/ubuntu/model_test/export_files/subset_stock_sectors.json', 'rb') as f:
  stock_sectors = pickle.load(f)

#Import full set
with open('/home/ubuntu/model_test/export_files/all_stock_sectors.json', 'rb') as f:
  all_stock_sectors = pickle.load(f)

#n_items = {k: stock_sectors[k] for k in list(stock_sectors)[0:5]}

results_dict = {}
count=1
for (sector, ticker) in all_stock_sectors.keys():
    try:
        print("Start ticker " + str(count) + ": " + ticker)
        scraper, processor = instantiate (ticker)
        df_reshape = scrape_and_process()
        parameters = prepare_data(df_reshape, 15, 5)
        results_dict[(sector, ticker)] = run_model(parameters)
        print("End ticker " + str(count) + ": " + ticker)
        count+=1
    except:
        pass

best_tickers = remove_nans(results_dict, 54)

# Import the names of the (sector, equity) and volume dictionary
#with open ('/home/ubuntu/model_test/export_files/best_tickers.json', 'rb') as f:
#    best_tickers = pickle.load (f)
with open('/home/ubuntu/model_test/export_files/best_tickers.json', 'wb') as f:
  pickle.dump(best_tickers, f)
