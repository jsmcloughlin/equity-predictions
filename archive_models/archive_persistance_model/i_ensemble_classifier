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

    kfold = KFold(n_splits=8, shuffle=True)
    results = cross_val_score(rfc_pipeline, X_classical, encoded_y, cv=kfold)
    print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

    return results.mean()*100, results.std()*100

def remove_nans(dict):
    new_dict = {}
    for key, values in results_dict.items ():
        i, j = values
        if not isnan (i):
            new_dict [ key ] = values

    return new_dict


#Import the names of the (sector, equity) and volume dictionary
with open('/home/ubuntu/model_test/export_files/best_tickers.json', 'rb') as f:
    best_tickers = pickle.load(f)



#n_items = {k: stock_sectors[k] for k in list(stock_sectors)[0:5]}

results_dict = {}
count=1
for (sector, ticker) in stock_sectors.keys():
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

best_tickers = remove_nans(results_dict)

with open('/home/ubuntu/model_test/export_files/best_tickers.json', 'wb') as f:
  pickle.dump(best_tickers, f)







final_rf_classifier = RandomForestClassifier\
    (
    bootstrap=False,
    criterion='entropy',
    max_depth=20,
    min_samples_split=2,
    max_features='sqrt',
    min_samples_leaf=1,
    n_estimators=800,
    random_state=42,
    class_weight=weights_dict
    )




#Fit the final model
rf_multi = final_rf_pipeline.fit(X_classical, encoded_y)















###################################################################################
'''
def baseline_model():
    model = Sequential()
    model.add(Dense(30, input_dim=330, activation='relu')) #X.shape[1]
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=2, shuffle=True)
results = cross_val_score(pipeline, X_classical, dummy_y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
'''

def create_model(optimizer='rmsprop', kernel_initializer='glorot_uniform', dropout=0.2, activation='relu'):
    model = Sequential ()
    model.add (Dense (30, input_dim=330, kernel_initializer=kernel_initializer, activation=activation))
    model.add (Dropout (dropout))
    model.add (Dense (15, kernel_initializer=kernel_initializer, activation=activation))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# wrap the model using the function you created
clf = KerasClassifier(build_fn=create_model,verbose=0)
scaler = StandardScaler()

kc_pipeline = Pipeline([
    ('preprocess',scaler),
    ('clf',clf)
], verbose=1)


# create parameter grid, as usual, but note that you can
# vary other model parameters such as 'epochs' (and others
# such as 'batch_size' too)
param_grid = {
    'clf__optimizer':['rmsprop','adam','adagrad'],
    'clf__epochs':[50, 100, 200],
    'clf__dropout':[0.0, 0.2, 0.4, 0.5],
    'clf__kernel_initializer':['glorot_uniform','normal','uniform'],
    'clf__activation':['relu', 'tanh'],
    'clf__batch_size':[8, 16, 32]
}

# if you're not using a GPU, you can set n_jobs to something other than 1
grid = RandomizedSearchCV(estimator = kc_pipeline, cv=3, param_distributions=param_grid, n_iter=20, verbose=1, random_state=42, n_jobs=-1)
grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid.best_score_, grid.best_params_))
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
params = grid.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


#Best: 0.271705 using {'clf__optimizer': 'adagrad', 'clf__kernel_initializer': 'glorot_uniform', 'clf__epochs': 100,
#                     'clf__dropout': 0.2, 'clf__batch_size': 8, 'clf__activation': 'tanh'}
#
processor.get_classifier_results(grid, X_test, y_test, encoder)
















        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid

        n_components = [20, 30, 50, 100, 300, 330] #this is for PCA() - best course of action appears to be normalization of the data

        class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
        weights_dict = dict(enumerate(class_weights))

        rf_classifier = RandomForestClassifier(criterion = 'entropy', random_state = 42, class_weight=weights_dict)



        rf_pipeline = Pipeline(steps=
        [
            ('scaler',RobustScaler()),
            #('pca', PCA()),
            ('kc', rf_classifier)
        ])

        random_grid = {
                        'kc__n_estimators': n_estimators,
                        'kc__max_features': max_features,
                        'kc__max_depth': max_depth,
                        'kc__min_samples_split': min_samples_split,
                        'kc__min_samples_leaf': min_samples_leaf,
                        'kc__bootstrap': bootstrap}


        rf_validator = RandomizedSearchCV(estimator = rf_pipeline,
                                          cv=3,
                                          param_distributions=random_grid,
                                          n_iter = 30,
                                          verbose=0,
                                          random_state=42,
                                          n_jobs=1)

        rf_validator.fit(X_train, y_train)
        processor.get_results(rf_validator, X_test, y_test, encoder)

'''
{'kc__n_estimators': 800, 'kc__min_samples_split': 2, 'kc__min_samples_leaf': 1, 'kc__max_features': 'sqrt', 
 'kc__max_depth': 20, 'kc__bootstrap': False}
Pipeline(steps=[('scaler', RobustScaler()),
                ('kc',
                 RandomForestClassifier(bootstrap=False,
                                        class_weight={0: 0.7943217665615142,
                                                      1: 0.9291512915129151,
                                                      2: 0.5159836065573771,
                                                      3: 3.3131578947368423,
                                                      4: 2.353271028037383},
                                        criterion='entropy', max_depth=20,
                                        max_features='sqrt', n_estimators=800,
                                        random_state=42))])

Balanced Accuracy Score (Overall): 
0.5281114056513132
Balanced Crosstab Rank (Overall): 
Predicted  +/-1%  1-5% Down  1-5% Up  5+% Down  5+% Up
Actual                                                
+/-1%         57         18       51         1       2
1-5% Down     26         69       27         7       0
1-5% Up       28         12      157         1       7
5+% Down       1         12        3        12       0
5+% Up         0          0       26         0      23
'''



#results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))



#Now re-fit the classifier using the best parameters and against all of the data
#Make this as up to date as possible with data

class_weights = compute_class_weight('balanced', np.unique(encoded_y), encoded_y)
weights_dict = dict(enumerate(class_weights))

final_rf_classifier = RandomForestClassifier(class_weight=weights_dict)


final_rf_classifier = RandomForestClassifier\
    (
    bootstrap=False,
    criterion='entropy',
    max_depth=20,
    min_samples_split=2,
    max_features='sqrt',
    min_samples_leaf=1,
    n_estimators=800,
    random_state=42,
    class_weight=weights_dict
    )



final_rf_pipeline = Pipeline([
    ('scaler',RobustScaler()),
    ('kc', final_rf_classifier)
])


#Fit the final model
rf_multi = final_rf_pipeline.fit(X_classical, encoded_y)


#Dump the saved model and the encoder
joblib.dump(rf_multi, '/home/ubuntu/model_test/saved_models/rf_multi.mod')

#exporting the departure encoder
output = open('/home/ubuntu/model_test/saved_models/rf_multi_encoder.pkl', 'wb')
pickle.dump(encoder, output)
output.close()



#Load and predict from saved model
load_classifier = joblib.load('/home/ubuntu/model_test/saved_models/rf_multi.mod')

#Testing final model on X_Classifier
        yhat = rf_multi.predict(X_classical)
        yhat_proba = rf_multi.predict_proba(X_classical)

        yhat_inv = processor.decode_yhat(yhat, yhat_proba, encoder)

        y_inv = processor.decode_y(encoded_y, encoder)
        df_inv = pd.concat([yhat_inv, y_inv], axis=1)

        df_inv['rank'] = df_inv.groupby(['y_label'])['yhat_proba'].transform(lambda x: pd.qcut(x, 5, labels=range(1, 6)))

        df_inv['rank'] = df_inv['rank'].astype('int32')

        print(
            'Balanced Accuracy Score (Overall): \n' + str(
                balanced_accuracy_score(df_inv['y_label'], df_inv['yhat_label'])))
        print('Balanced Crosstab Rank (Overall): \n' + str(
            pd.crosstab(df_inv['y_label'], df_inv['yhat_label'], rownames=['Actual'], colnames=['Predicted'])))

        rcount = list(df_inv['rank'].unique())
        for i in range(1, len(rcount) + 1):
            df = df_inv[df_inv['rank'] == i]
            print('Balanced Accuracy Score Rank \n' + str(i) + ' ' + str(
                balanced_accuracy_score(df['y_label'], df['yhat_label'])))
            print('Balanced Crosstab Rank \n' + str(i) + ' ' + str(
                pd.crosstab(df['y_label'], df['yhat_label'], rownames=['Actual'], colnames=['Predicted'])))
