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
The best parameters applied and used with this Multivairate model:
[10, 25, 700, 64, 1, 5, 0.005, 'tanh', 0.5] - 125 and 131 epochs on repeats early stopping
[10, 25, 135, 64, 1, 5, 0.005, 'tanh', 0.5]
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



	# train is now the entire set
def build_model(train_x, train_y, config):

	n_input, n_nodes, n_epochs, n_batch, n_diff, n_out, n_lr, n_actfn, n_dropout = config

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
	model.compile(loss='mae', optimizer=ADAM)
	# fit network
	model.fit(train_x, train_y, epochs=n_epochs, batch_size=n_batch, verbose=0)

	return model

def forecast(model, input_x):

	# forecast the next 5 days - reshape from 2d to 3d in preparation
	input_x = input_x.reshape(1, input_x.shape[0], input_x.shape[1])
	yhat = model.predict(input_x, verbose=0)

	return yhat
	#return aapl.invert_scale(scaler_y, yhat)


def get_difference_pct(data, interval=1):
	'''Takes in a 2d dataset and an interval (default is 1), then returns
    a differenced value'''

	diff = list()
	for i in range(interval, len(data)):
		value = (data[i] - data[i - interval]) / data[i - interval]
		diff.append(value)

	return diff


def invert_difference_pct(data, data_diff, interval=1):
	'''Takes in a 2d dataset and an interval (default is 1), then returns
    a differenced value'''

	diff = list()
	diff.append(data[0])
	for i in range(len(data_diff)):
		value = (data_diff[i] * data[i]) + data[i]
		diff.append(value)

	return diff


 # evaluate a single model
def evaluate_model(data, config):
	n_input, n_nodes, n_epochs, n_batch, n_diff, n_out, n_lr, n_actfn, n_dropout = config

	data_diff = np.array(get_difference_pct(data, n_diff))
	train_x, train_y = aapl.to_supervised(data_diff, n_input, n_out)
	model = build_model(train_x, train_y, config)

	# walk-forward validation over each week
	predictions, actuals, predictions_ff = list(), list(), list()
	n_start = n_input
	for i in range(len(train_x)):
		# predict the week - note forecast_c() allows for differencing and includes n_diff
		yhat_sequence = forecast(model, train_x[i, :, :])

		# store the predictions
		# get real observation and add to history for predicting the next n days using the forecast() function
		#history = np.vstack((history, data[i, :]))


		n_end = n_start + n_out
		if n_end <= len(data): #Allows for the EoF
			actuals.append(data[n_start:n_end, 0])

			#Calculate predictions based on using each actual, previous day - not realistic because
			#we will not have each previous day when making a real world n_out day foreccast.
			#This should not be used to minimize error
			day0_n_out_less_diff = data[n_start - n_diff:n_end - n_diff, 0].flatten()
			predictions.append((day0_n_out_less_diff * yhat_sequence) + day0_n_out_less_diff)
			#####################################################################################

			# If predictions are done on day's 1 - n_out, this is day 0 of an undifferenced data set
			#Predictions_ff should be used for minimizing the error and for subsequent yhat
			yhat_sequence_n = [] #instantiate each time a new sequence is generated (inside for loop)
			day0 = data[n_start - n_diff, 0] #pull day 0 of the sequence
			for yhat_pct in yhat_sequence.flatten(): # extract each % prediction - day 1 - day n_out
				yhat = (yhat_pct * day0) + day0 #add the predicted % change to day0
				yhat_sequence_n.append(yhat) #add to the undifferenced yhat_sequence_undifferenced
				day0 = yhat # update day0 to the next predicted day along
			predictions_ff.append(yhat_sequence_n) # add the n_out sequence to the overall prediction list
			#####################################################################################
		n_start += 1

	# evaluate predictions days for each week
	actuals = array(actuals)
	predictions = array(predictions)
	predictions_ff = array(predictions_ff)

	actuals = actuals.reshape(actuals.shape[0], actuals.shape[1], 1)
	predictions = predictions.reshape(predictions.shape[0], predictions.shape[2], predictions.shape[1])
	predictions_ff = predictions_ff.reshape(predictions_ff.shape[0], predictions_ff.shape[1], 1)


	return actuals, predictions, predictions_ff, model


#data=ds
aapl = cl.parent_rnn('AAPL') #instantiate the object
import_df = aapl.get_prepare_stock_data()
ds = aapl.process_data(import_df)

#When calling this function, set ndim=1 for univariate, and anything else for multivariate
data = aapl.prepare_variate(ds, 0)


config = [10, 25, 135, 64, 1, 5, 0.005, 'tanh', 0.5]

actuals, predictions, predictions_ff, cnn_seq_model =  evaluate_model(data, config)
score, scores = evaluate_forecasts(actuals, predictions) #y is used to calculate loss for yhat
score_ff, scores_ff = evaluate_forecasts(actuals, predictions_ff) #yhat is used to calculate loss for yhat+1 (realistic)


summarize_scores('cnn', score, scores)
# plot scores
days = ['day1', 'day2', 'day3', 'day4', 'day5']
plt.plot(days, scores, marker='o', label='cnn')
plt.show()

summarize_scores('cnn', score_ff, scores_ff)
# plot scores
days = ['day1', 'day2', 'day3', 'day4', 'day5']
plt.plot(days, scores_ff, marker='o', label='cnn')
plt.show()


#In plotting the actuals v predictions below, it is possible that the model is
#simply learning a persistance - that is, using the most recent value to make
#the prediction.
act = np.array(actuals)

pred = np.array(predictions)
pred_ff = np.array(predictions_ff)

n_out=5
for i in range(0, n_out):
	plt.plot (actuals[-100:, i], color='blue', label='actual')
	plt.plot (predictions[-100:, i], color='orange', label='prediction')
	plt.plot(predictions_ff[-100:, i], color='red', label='prediction ff')
	plt.title ("Actual v Prediction: Day " + str(i+1))
	plt.legend()
	plt.show ()

#Prints the most recent 5 days predictions
plt.plot (actuals[-1:, :].flatten(), color='blue', label='actual')
plt.plot (predictions[-1:, :].flatten(), color='orange', label='prediction')
plt.plot(predictions_ff[-1:, :].flatten(), color='red', label='prediction ff')
plt.legend()
plt.show ()

# Prints the most recent n days predictions - including day 1-5 of each period
plt.plot(actuals[-10:, :].flatten(), color='blue', label='actual')
plt.plot (predictions[-10:, :].flatten(), color='orange', label='prediction')
plt.plot(predictions_ff[-10:, :].flatten(), color='red', label='prediction ff')
plt.legend()
plt.show()

#Next thing is to do a correlation plot

import seaborn as sns
import scipy.stats as stats

def get_predictions(y, yhat):
    return pd.DataFrame({'actual': y, 'pred': yhat})

def get_regressor_charts(y, yhat):

    df = get_predictions(y, yhat)
    labels = df.columns

    fig = plt.figure(figsize=(8, 4))
    j = sns.jointplot(x='pred', y='actual', kind='reg', data=df, height=8)
    j.annotate(stats.pearsonr)
    plt.show()

preds_ff_resh = predictions.reshape(predictions.shape[0], predictions.shape[2], predictions.shape[1])
acts_resh = actuals.reshape(actuals.shape[0], actuals.shape[2], actuals.shape[1])

for i in range(0, n_out):
	y = acts_resh[:, :, i].flatten()
	yhat = preds_ff_resh[:, :, i].flatten()
	df = get_regressor_charts(y, yhat)


############################Now save the model and weights####################
from numpy import loadtxt
from keras.models import load_model

cnn_seq_model.save('/home/ubuntu/stock_lstm/saved_models/cnn_seq_model.h5')
print("Saved model to disk")

# load model
cnn_seq_model = load_model('/home/ubuntu/stock_lstm/saved_models/cnn_seq_model.h5') # summarize model. model.summary()