from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, date
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import classification_report, multilabel_confusion_matrix, balanced_accuracy_score
import sys
#sys.path.append("/home/ubuntu/model_test")
#sys.path.append("/Users/jamesm/.local/lib/python3.6/site-packages")
from equity_classes import classes as cl
import joblib
import pickle
from sklearn.metrics import make_scorer
import requests_html

import os
print(os.getcwd())
print(sys.path)


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
intraday_max_label = pd.DataFrame.from_records(aapl_reg.get_chg_pc_label(intraday_max), columns = ['intraday_max', 'intraday_max_label'])

dummy_y, encoded_y, encoder = aapl_reg.encode_y(nday_chg_label)

aapl_reg.get_exp_charts(nday_chg_label)
aapl_reg.get_exp_charts(intraday_max_label)

X_train, X_test, y_train, y_test = train_test_split(X_classical, encoded_y, test_size=0.3, random_state=101)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

'''
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
aapl_reg.get_results(rf_validator, X_test, y_test, encoder)

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







#Now re-fit the classifier using the best parameters and against all of the data
#Make this as up to date as possible with data

class_weights = compute_class_weight('balanced', np.unique(encoded_y), encoded_y)
weights_dict = dict(enumerate(class_weights))

#final_rf_classifier = RandomForestClassifier(class_weight=weights_dict)


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
], verbose = True)


#Fit the final model
rf_multi = final_rf_pipeline.fit(X_classical, encoded_y)


#Dump the saved model and the encoder
joblib.dump(rf_multi, '/home/ubuntu/model_test/saved_models/rf_multi.mod')

#exporting the departure encoder
output = open('/home/ubuntu/model_test/saved_models/rf_multi_encoder.pkl', 'wb')
pickle.dump(encoder, output)
output.close()


#'/home/ubuntu/model_test/saved_models/rf_regressor.mod'

#Load and predict from saved model
load_classifier = joblib.load('/home/ubuntu/model_test/saved_models/rf_multi.mod')

#Testing final model on X_Classifier
'''
        yhat = rf_multi.predict(X_classical)
        yhat_proba = rf_multi.predict_proba(X_classical)

        import seaborn as sns
        sns.distplot(yhat_proba)
        plt.show()

        yhat_inv = aapl_reg.decode_yhat(yhat, yhat_proba, encoder)
        sns.countplot(x='yhat_label', data=yhat_inv)
        plt.show()

        #y_inv = aapl_reg.decode_y(y_test, encoder)
        #df_inv = pd.concat([yhat_inv, y_inv], axis=1)

        #df_inv['rank'] = df_inv.groupby(['y_label'])['yhat_proba'].transform(lambda x: pd.qcut(x, 5, labels=range(1, 6)))

        #df_inv['rank'] = df_inv['rank'].astype('int32')

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
'''