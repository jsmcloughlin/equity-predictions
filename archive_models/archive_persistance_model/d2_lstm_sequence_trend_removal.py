#Review more moving to class then begin hyperp tuning
from math import sqrt
from numpy import split, array
import numpy as np
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from keras.constraints import maxnorm
from sklearn.preprocessing import  RobustScaler
import pandas as pd
import numpy as np
from keras.layers import Dense, Flatten, Dropout, Activation, LSTM, RepeatVector, TimeDistributed
import matplotlib.dates as mdates
from datetime import datetime, date
from importlib import reload
import sys
sys.path.append("/home/ubuntu/model_testing")
from equity_classes import classes as cl
import random
np.seterr(divide='ignore', invalid='ignore')
'''

Top 5 results from random cross validation
Univariate (CNN) Model:
[15, 25, 100, 64, 1, 5, 0.01, 'relu', 0] 3.0468791360571563
[15, 50, 100, 32, 1, 5, 0.2, 'relu', 0] 3.05446834974386
[15, 25, 750, 32, 1, 5, 0.01, 'tanh', 0] 3.056142748526239
[15, 50, 1000, 64, 1, 5, 0.09, 'relu', 0.5] 3.0626661021825896
[15, 50, 100, 16, 1, 5, 0.09, 'tanh', 0] 3.0629027824533703

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
def build_model(train, config):

	n_input, n_nodes, n_epochs, n_batch, n_diff, n_out, n_lr, n_actfn, n_dropout = config

	train_diff = np.array(aapl.get_difference_pct(train, n_diff))
	#ar_nan = np.where(np.isnan(train_diff))
	#print(ar_nan)

	train_x, train_y = aapl.to_supervised(train_diff, n_input, n_out)
	#test_x, test_y = aapl.to_supervised(test_diff, n_input, n_out)

	# define parameters
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# define model
	model = Sequential()

	model.add(LSTM(n_nodes, input_shape=(n_timesteps, n_features), return_sequences=True))
	model.add(Activation(n_actfn))
	model.add(Dropout(n_dropout))
	model.add(LSTM(n_nodes // 2))
	model.add(Activation(n_actfn))
	model.add(Dropout(n_dropout))
	model.add(Dense(n_nodes // 2))
	model.add(Activation(n_actfn))
	model.add(Dense(n_outputs))

	decay_rate = n_lr / n_epochs
	ADAM = Adam(lr=n_lr, decay=decay_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
	model.compile(loss='mse', optimizer=ADAM)

	es = EarlyStopping(monitor='loss', mode='auto', verbose=1, patience=18)
	# model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=1, validation_data=(test_x, test_y))
	model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0, callbacks=[es])

	return model


def forecast(model, history, n_input, interval):
	'''The only difference with function: forecast_d is that there is
    an allowance for differencing (removal of the trend)'''

	data = array(history)

	# If differencing
	data = np.array(aapl.get_difference_pct(data, interval))  # difference the history data to prepare next test sample

	# retrieve last observations for input data
	input_x = data[-n_input:, :]
	input_x = input_x.reshape(1, input_x.shape[0], input_x.shape[1])

	# forecast the next week on the differenced data
	yhat = model.predict(input_x, verbose=0)

	return yhat

	# evaluate a single model
def evaluate_model(data, n_train, config):

	n_input, n_nodes, n_epochs, n_batch, n_diff, n_out, n_lr, n_actfn, n_dropout = config
	interval = 1
	train, test = aapl.split_dataset(data, n_train)
	model = build_model(train, config)

	history = train
	eval_data = test

	# walk-forward validation over each week
	predictions, actuals, predictions_ff = list(), list(), list()

	n_start = n_input
	for i in range(len(eval_data)):
		# predict the week - note forecast_c() allows for differencing and includes n_diff
		yhat_sequence = forecast(model, history, n_input, interval)
		history = np.vstack((history, eval_data[i, :]))

		n_end = n_start + n_out
		if n_end <= len(eval_data):  # Allows for the EoF (-n_diff)
			actuals.append(eval_data[n_start:n_end, 0])

			# Calculate predictions based on using each actual, previous day - not realistic because
			# we will not have each previous day when making a real world n_out day foreccast.
			# This should not be used to minimize error
			day0_n_out_less_diff = eval_data[n_start - n_diff:n_end - n_diff, 0].flatten()
			predictions.append((day0_n_out_less_diff * yhat_sequence) + day0_n_out_less_diff)
			#####################################################################################

			# If predictions are done on day's 1 - n_out, this is day 0 of an undifferenced data set
			# Predictions_ff should be used for minimizing the error and for subsequent yhat
			yhat_sequence_n = []  # instantiate each time a new sequence is generated (inside for loop)
			day0 = test[n_start - n_diff, 0]  # pull day 0 of the sequence
			for yhat_pct in yhat_sequence.flatten():  # extract each % prediction - day 1 - day n_out
				yhat = (yhat_pct * day0) + day0  # add the predicted % change to day0
				yhat_sequence_n.append(yhat)  # add to the undifferenced yhat_sequence_undifferenced
				day0 = yhat  # update day0 to the next predicted day along
			predictions_ff.append(yhat_sequence_n)  # add the n_out sequence to the overall prediction list
		#####################################################################################
		n_start += 1

	# evaluate predictions days for each week
	actuals = array(actuals)
	predictions = array(predictions)
	predictions_ff = array(predictions_ff)

	actuals = actuals.reshape(actuals.shape[0], actuals.shape[1], 1)
	predictions = predictions.reshape(predictions.shape[0], predictions.shape[2], predictions.shape[1])
	predictions_ff = predictions_ff.reshape(predictions_ff.shape[0], predictions_ff.shape[1], 1)

	score, scores = evaluate_forecasts(actuals, predictions_ff)
	score_ln, scores_ln = evaluate_forecasts(actuals[-10:], predictions_ff[-10:])

	# Prints the most recent n days predictions - including day 1-5 of each period
	plt.plot(actuals[-20:, :].flatten(), color='blue', label='actual')
	plt.plot(predictions_ff[-20:, :].flatten(), color='red', label='prediction ff')
	plt.legend()
	plt.show()

	return scores

#for i in range(0, n_out):
#	plt.plot (actuals[-20:, i], color='blue', label='actual')
#	#plt.plot (predictions[-100:, i], color='orange', label='prediction')
#	plt.plot(predictions_ff[-20:, i], color='red', label='prediction ff')
#	plt.title ("Actual v Prediction: Day " + str(i+1))
#	plt.legend()
#	plt.show ()


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

#config = [10, 100, 100, 16, 1, 5, 0.15, 'tanh', 0]

n_train = 0.8


cfg_list = random.choices(aapl.model_configs(), k=40)
config = cfg_list[0]

scores = grid_search(data, cfg_list, n_train)
print('done')

#list top configs
for cfg, error in scores[:5]:
	print(cfg, error)


