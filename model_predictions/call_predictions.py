import subprocess

subprocess.call(['python3', '/home/ubuntu/model_test/model_predictions/1_model_predictions.py'])
subprocess.call(['python3', '/home/ubuntu/model_test/model_predictions/2_backtest_actuals_tradestops.py'])
subprocess.call(['python3', '/home/ubuntu/model_test/model_predictions/3_precision_recall.py'])
subprocess.call(['python3', '/home/ubuntu/model_test/model_predictions/4_oneday_predictions.py'])
