#Review more moving to class then begin hyperp tuning
from math import sqrt
from numpy import split, array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from keras.constraints import maxnorm
from sklearn.preprocessing import  RobustScaler
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, date
from importlib import reload
import sys
sys.path.append("/home/ubuntu/model_testing")
from equity_classes import classes as cl
import random

'''
Top 5 results from random cross validation
Univariate
[25, 25, 1500, 128, 1, 5, 0.09, 'tanh', 0.5] 2.2645897418214096
[25, 25, 1000, 128, 1, 5, 0.09, 'tanh', 0.5] 2.5328209138445774
[10, 50, 750, 32, 1, 5, 0.05, 'tanh', 0.5] 2.6944125223186033
[10, 100, 500, 16, 1, 5, 0.09, 'tanh', 0.5] 2.79446775362913
[10, 25, 750, 32, 1, 5, 0.01, 'tanh', 0.5] 3.1772392884733187


Multivariate
[15, 25, 1000, 128, 1, 5, 0.09, 'tanh', 0.5] 3.028238186123635
[15, 25, 100, 64, 1, 5, 0.05, 'tanh', 0.5] 3.2549929566733717
[10, 50, 1000, 128, 1, 5, 0.01, 'tanh', 0.5] 3.327859974925378
[10, 50, 750, 64, 1, 5, 0.05, 'tanh', 0.5] 3.4921379237493975
[10, 25, 1000, 16, 1, 5, 0.09, 'tanh', 0.1] 3.7721380685336974
'''



def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[1]):
		# calculate mse
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores

# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))



	# train the model
def build_model(train, test, config):

	n_input, n_nodes, n_epochs, n_batch, n_diff, n_out, n_lr, n_actfn, n_dropout = config

	train_scale, scaler, scaler_y = aapl.create_scaler(np.array(train))  # creates the scaler on train
	test_scale = aapl.apply_scaler(np.array(test), scaler, scaler_y)  # applies the scaler to test

	train_x, train_y = aapl.to_supervised(train_scale, n_input, n_out)
	test_x, test_y = aapl.to_supervised(test_scale, n_input, n_out)

	# define parameters
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# define model
	model = Sequential()
	model.add(Conv1D(16, 3, activation=n_actfn, input_shape=(n_timesteps,n_features)))
	model.add(MaxPooling1D())
	model.add(Flatten())
	model.add(Dense(n_nodes, activation=n_actfn))
	model.add(Dropout(n_dropout))
	model.add(Dense(n_nodes//2, activation=n_actfn, kernel_constraint=maxnorm(3))) #impose constraint on the layer weights
	model.add(Dense(n_outputs))
	decay_rate = n_lr / n_epochs
	ADAM = Adam(lr=n_lr, decay=decay_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
	model.compile(loss='mse', optimizer=ADAM)
	# fit network
	model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)

	return model, scaler, scaler_y

	# evaluate a single model
def evaluate_model(data, n_train, config):
	n_input, n_nodes, n_epochs, n_batch, n_diff, n_out, n_lr, n_actfn, n_dropout = config
	train, test = aapl.split_dataset(data, n_train)
	model, scaler, scaler_y = build_model(train, test, config)

	history = train

	# walk-forward validation over each week
	predictions, actuals, predictions_ff = list(), list(), list()
	n_start = n_input
	for i in range(len(test)):
		# predict the week
		# predict the week - note forecast_d() does not allow for differencing and excludes n_diff
		yhat_sequence = aapl.forecast_d(model, history, n_input, scaler, scaler_y)

		# store the predictions
		# get real observation and add to history for predicting the next n days using the forecast() function
		history = np.vstack((history, test[i, :]))

		n_end = n_start + n_out
		if n_end <= len(test): #Allows for the EoF
			actuals.append(test[n_start:n_end, 0])
			predictions.append(yhat_sequence)

	# evaluate predictions days for each week
	actuals = array(actuals)
	actuals = actuals.reshape(actuals.shape[0], actuals.shape[1], 1)
	predictions = array(predictions)
	predictions = predictions.reshape(predictions.shape[0], predictions.shape[1], 1)

	score, scores = evaluate_forecasts(actuals, predictions)
	return scores

# score a model, return None on failure
def repeat_evaluate(data, config, n_train, n_repeats=2):
	# convert config to a key
	key = str(config)
	# fit and evaluate the model n times
	#scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]
	scores = [evaluate_model(data, n_train, config)  for _ in range(n_repeats)]
	# summarize score from the repeats of each config
	result = np.mean(scores)
	print('> Model[%s] %.3f' % (key, result))
	return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_train):
	# evaluate configs
	scores = [repeat_evaluate(data, cfg, n_train) for cfg in cfg_list]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores


#data=ds
aapl = cl.parent_rnn('AAPL') #instantiate the object
import_df = aapl.get_prepare_stock_data()
ds = aapl.process_data(import_df)

#When calling this function, set ndim=1 for univariate, and anything else for multivariate
data = aapl.prepare_variate(ds, 0)

n_train = 0.8
#Current best config:
#s> Model[[10, 100, 50, 16, 1, 5, 0.01, 'tanh']] 1.682

cfg_list = random.choices(aapl.model_configs(), k=25)

scores = grid_search(data, cfg_list, n_train)
print('done')

#list top configs

for cfg, error in scores[:5]:
	print(cfg, error)


'''
# evaluate model and get scores
n_input = 15
n_out = 5
train_pct = 0.7
interval = 1


#score, scores, actuals, predictions, history, predictions_ff = evaluate_model(df, n_input, n_out, train_pct, interval)
# summarize scores
summarize_scores('cnn', score, scores)
# plot scores
days = ['day1', 'day2', 'day3', 'day4', 'day5']
plt.plot(days, scores, marker='o', label='cnn')
plt.show()


#In plotting the actuals v predictions below, it is possible that the model is
#simply learning a persistance - that is, using the most recent value to make
#the prediction.
act = np.array(actuals)

pred = np.array(predictions)
pred_ff = np.array(predictions_ff)

for i in range(0, n_out):
	plt.plot (act[-25:, i], color='blue', label='actual')
	plt.plot (pred[-25:, i], color='orange', label='prediction')
	plt.plot(pred_ff[-25:, i], color='red', label='prediction ff')
	plt.title ("Actual v Prediction: Day " + str(i+1))
	plt.legend()
	plt.show ()

#Prints the most recent 10 days predictions
plt.plot (act[-1:, :].flatten(), color='blue', label='actual')
plt.plot (pred[-1:, :].flatten(), color='orange', label='prediction')
plt.plot(pred_ff[-1:, :].flatten(), color='red', label='prediction ff')
plt.legend()
plt.show ()
'''
