from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, date
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from keras.models import load_model
import joblib
import scipy.stats as stats
import math
import sys
sys.path.append("/home/ubuntu/model_testing")
from equity_classes import classes as cl
from sklearn.metrics import make_scorer

'''Wall time: 5h 1min 20s
The parameters of the best model are:
{'kc__n_estimators': 2000,
 'kc__min_samples_split': 2,
 'kc__min_samples_leaf': 2,
 'kc__max_features': 'sqrt',
 'kc__max_depth': 80,
 'kc__bootstrap': False}

RMSE Train: 0.003170801088337121
Score Train: 0.9931152229245923
RMSE Test: 0.02163670994577538
Score Train: 0.5918003821956077
'''



aapl_reg = cl.prepare_classical('AAPL') #instantiate the object
import_df = aapl_reg.get_prepare_stock_data()
dataset = aapl_reg.process_data(import_df)
dataset = dataset.drop(['adj close', 'day', 'ticker'], axis=1)
df_reshape = aapl_reg.reshape_dataset(np.array(dataset), 1)

X, y = aapl_reg.to_supervised_classical(df_reshape, 15, 5)


X_classical = pd.DataFrame(aapl_reg.reshape_X_classical(X)) #Reshapes X into 1 row and all columns for the features
y_classical = aapl_reg.reshape_y_classical(y, n_out=5) #Reshapes y to calculate % change


nday_chg, intraday_max = aapl_reg.get_chg_pc(y_classical)

nday_chg_label = pd.DataFrame.from_records(aapl_reg.get_chg_pc_label(nday_chg), columns = ['nday_chg', 'nday_chg_label'])

unencoded_y = nday_chg_label['nday_chg']

X_train, X_test, y_train, y_test = train_test_split(X_classical, unencoded_y, test_size=0.3, random_state=101)
y_train =  y_train.astype(float)
y_test =  y_test.astype(float)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#Vanilla
#vanilla = RandomForestRegressor(n_jobs=-1)
#%time vanilla.fit(X_train, y_train)
#print_score(vanilla)


#30 Tree
#tirtytree = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)
#%time tirtytree.fit(X_train, y_train)
#print_score(tirtytree)


#y_test.values[0]
#yhat = np.stack([t.predict(X_test) for t in vanilla.estimators_])
#yhat[:,0], np.mean(yhat[:,0]) #Calculate the prediction of each estimateor - then the mean of all estimators
#the bagging part (avg of all estimators)
#known value:           0.03216
#avg / bagged value:    0.01801

#yhat.shape
#plt.plot([metrics.r2_score(y_test, np.mean(yhat[:i+1], axis=0)) for i in range(yhat.shape[0])]);
#plt.show() #clearly, increasing number of trees is not very helpful - not much gain after 20


def rmse(x,y):
    return math.sqrt(((x-y)**2).mean())

def get_score(model):
    print("RMSE Train: " + str(rmse(model.predict(X_train), y_train)))
    print("Score Train: " + str(model.score(X_train, y_train)))
    print("RMSE Test: " + str(rmse(model.predict(X_test), y_test)))
    print("Score Train: " + str(model.score(X_test, y_test)))


def measure_rmse(actual, predicted):
	return np.sqrt(mean_squared_error(actual, predicted))


def get_predictions(X, y, model):
    return pd.DataFrame({'actual': y, 'pred': model.predict(X)})

def get_regressor_charts(X, y, model):

    df = get_predictions(X, y, model)
    labels = df.columns

    fig = plt.figure(figsize=(8, 4))
    j = sns.jointplot(x='pred', y='actual', kind='reg', data=df, height=8)
    j.annotate(stats.pearsonr)
    plt.show()



'''
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 1, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 20]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 10, 20]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid



rf_regressor = RandomForestRegressor(criterion = 'mse', random_state = 42)

rf_pipeline = Pipeline(steps=
[
    ('scaler',RobustScaler()),
	#('pca', PCA()),
    ('kc', rf_regressor)
])

random_grid = {
                'kc__n_estimators': n_estimators,
                'kc__max_features': max_features,
                'kc__max_depth': max_depth,
                'kc__min_samples_split': min_samples_split,
                'kc__min_samples_leaf': min_samples_leaf,
                'kc__bootstrap': bootstrap}


scorer = make_scorer(measure_rmse, greater_is_better=False)

rf_validator = RandomizedSearchCV(estimator = rf_pipeline,
								  cv=3,
								  param_distributions=random_grid,
								  n_iter = 50,
								  verbose=1,
								  random_state=42,
								  n_jobs=1,
                                  scoring=scorer)

%time rf_validator.fit(X_train, y_train)

print('The parameters of the best model are: ')
print(rf_validator.best_params_)
best_model = rf_validator.best_estimator_
print(best_model)

get_score(best_model)
get_regressor_charts(X_train, y_train, best_model)
get_regressor_charts(X_test, y_test, best_model)



############################################################
Wall time: 5h 1min 20s
The parameters of the best model are:
{'kc__n_estimators': 2000,
 'kc__min_samples_split': 2,
 'kc__min_samples_leaf': 2,
 'kc__max_features': 'sqrt',
 'kc__max_depth': 80,
 'kc__bootstrap': False}

RMSE Train: 0.003170801088337121
Score Train: 0.9931152229245923
RMSE Test: 0.02163670994577538
Score Train: 0.5918003821956077
'''



#Final Saved Model###################
final_rf_classifier = RandomForestRegressor()


final_rf_regressor = RandomForestRegressor\
    (
    bootstrap=False,
    criterion='mse',
    max_depth=80,
    min_samples_split=2,
    max_features='sqrt',
    min_samples_leaf=2,
    n_estimators=2000,
    random_state=42
    )



final_rf_pipeline = Pipeline([
    ('scaler',RobustScaler()),
    ('kc', final_rf_regressor)
], verbose=True)


#Fit the final model
rf_regressor = final_rf_pipeline.fit(X_classical, unencoded_y)

#Dump the saved model
joblib.dump(rf_regressor, '/home/ubuntu/model_test/saved_models/rf_regressor.mod')

#Load and predict from saved model
load_regressor = joblib.load('/home/ubuntu/model_test/saved_models/rf_regressor.mod')


get_score(rf_regressor)
get_regressor_charts(X_classical, unencoded_y, rf_regressor)
get_regressor_charts(X_classical, unencoded_y, rf_regressor)

df = get_predictions(X_classical, unencoded_y, rf_regressor)

sns.distplot(df['actual'].astype(float), kde=False, bins=50)
sns.distplot(df['pred'], kde=False, bins=50)
plt.show()