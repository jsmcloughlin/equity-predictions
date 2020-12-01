import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
import seaborn as sns
import scipy.stats as stats
import datetime as dt

pd.options.display.float_format = "{:,.2f}".format
from sklearn.metrics import classification_report


def import_y_yhat():
    pathname = r'/home/ubuntu/model_test/model_output/'
    y_import = 'actuals.csv'
    yhat_import = 'model_preds_deduped.csv'

    y_df = pd.read_csv (pathname + y_import, keep_default_na=False)
    y_df.drop_duplicates (subset=[ 'day1', 'day5', 'ticker' ], inplace=True, keep='first')

    yhat_df = pd.read_csv (pathname + yhat_import, keep_default_na=False)
    yhat_df.drop (yhat_df.columns [ 0 ], axis=1, inplace=True)
    yhat_df.drop_duplicates (subset=[ 'start', 'end', 'ticker' ], inplace=True, keep='first')
    yhat_df = yhat_df.sort_values (by=[ 'ticker', 'last_close', 'start' ])

    floats = ['close1', 'close2', 'close3', 'close4', 'close5', 'actual_pct_change', 'low', 'low_pct_change']
    for colname in floats:
        y_df [ colname ] = pd.to_numeric (y_df [ colname ], errors='coerce')

    return y_df, yhat_df

def get_histogram(pct_chg, movement):
    pct_chg_arr = np.array (pct_chg)
    pct_chg_arr = pct_chg_arr [ ~np.isnan (pct_chg_arr) ]
    sd = np.std (pct_chg_arr)
    mean = np.mean (pct_chg_arr)
    median = np.median (pct_chg_arr)
    zscore = np.divide (np.subtract (pct_chg_arr, mean), sd)  # x - xbar / sd

    print (str (movement))
    print ('Std Dev: ' + str (sd))
    print ('Mean: ' + str (mean))
    print ('Median: ' + str (median))
    print ('------------------------------------------')

    sns.distplot (pct_chg_arr, color="maroon", bins=50, hist=True)
    plt.xlabel ("5 Day % Change ", labelpad=14)
    plt.ylabel ("Frequency", labelpad=14)
    plt.title ('5 day pct chg ' + str (movement))
    plt.show ()

    # 68% of values fall within +/-1 z-scores from the mean
    # 95% of values fall within +/-1.96 z-scores from the mean
    # 99% of values fall within +/-2.58 z-scores from the mean
    sns.distplot (zscore, color="y", bins=50, hist=True)
    plt.title ('Z Score ' + str (movement))
    plt.xlabel ("zscore" + str (movement), labelpad=14)
    plt.ylabel ("frequency", labelpad=14);
    plt.show ()

    return np.column_stack ((pct_chg_arr, zscore))


# Filling in the curve for +/- 2.5% and +/- 4%
def get_percentile(data, pct, low_close):
    '''Takes in a numpy array that has the % change
    and the corresponding z score, then returns the
    percentile of z score according to a normal distribution.'''
    zperc = round (stats.norm.cdf (data [ data [ :, 0 ] == pct ] [ 0, 1 ]), 3)
    print ('5 day ' + low_close + ' of ' + str (pct) + '% is in percentile: ' + str (zperc))
    return zperc




###############################################

def merge_y_yhat(y, yhat):
    # Column to keep
    y_keep = [ 'day1', 'day5', 'actual_pct_change', 'ticker', 'actual_pct_chg_label' ]
    yhat_keep = [ 'yhat_label', 'ticker', 'yhat_reg', 'start', 'end' ]

    # Refactor date fields
    y [ [ 'day1', 'day5' ] ] = y [ [ 'day1', 'day5' ] ].apply (pd.to_datetime)
    yhat [ [ 'start', 'end' ] ] = yhat [ [ 'start', 'end' ] ].apply (pd.to_datetime)

    y = y [ y_keep ].copy ()
    yhat = yhat [ yhat_keep ].copy ()
    yhat.rename (columns={'start': 'day1', 'end': 'day5'}, inplace=True)

    df = pd.merge (y, yhat, how='inner', on=[ 'day1', 'day5', 'ticker' ])
    df [ 'year' ] = pd.DatetimeIndex (df [ 'day1' ]).year
    df [ 'month' ] = pd.DatetimeIndex (df [ 'day1' ]).month
    #df [ 'month_year' ] = pd.to_datetime (df [ 'birth_date' ]).dt.to_period ('M')
    df [ 'month_year' ] = df [ 'day1' ].dt.to_period ('M')

    return df


# target_names=['4+% Up', '1-4% Up', '+/-1%', '1-4% Down', '4+% Down', 'Other']
def get_rmse(y, yhat):
    rmse = np.sqrt (np.power (np.sum (y - yhat), 2))
    return rmse


def get_classification_report(y, yhat):
    print (classification_report (y, yhat))


def get_4pct_accuracy(df):
    '''Calculate the TP, FP and FN of the current classification model.'''

    df_4pct = df [ (df.actual_pct_chg_label == '4+% Up') | (df.yhat_label == '4+% Up') ]
    c = {'tp': 0, 'fn': 0, 'fp': 0}
    df_4pct = df_4pct.assign (**c)

    for i in range (len (df_4pct)):
        df_4pct [ 'tp' ].iloc [ i ] = (
            1 if df_4pct [ 'actual_pct_chg_label' ].iloc [ i ] == df_4pct [ 'yhat_label' ].iloc [ i ] else 0)
        df_4pct [ 'fn' ].iloc [ i ] = (
            1 if df_4pct [ 'actual_pct_chg_label' ].iloc [ i ] == '4+% Up' and df_4pct [ 'yhat_label' ].iloc [
                i ] != '4+% Up' else 0)
        df_4pct [ 'fp' ].iloc [ i ] = (
            1 if df_4pct [ 'actual_pct_chg_label' ].iloc [ i ] != '4+% Up' and df_4pct [ 'yhat_label' ].iloc [
                i ] == '4+% Up' else 0)

    return df_4pct


def precision_recall(df):
    tp = np.sum(df [ 'tp' ])
    fp = np.sum(df [ 'fp' ])
    fn = np.sum(df [ 'fn' ])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall


def prec_rec_rmse(gtruth):
    #Calculate Overall Precision and Recall and RMSE
    precision_recall_rmse = {}
    precision_recall_rmse['precision recall'] = (precision_recall (get_4pct_accuracy (gtruth)))
    precision_recall_rmse['rmse'] = get_rmse (gtruth [ 'actual_pct_change' ].values, gtruth [ 'yhat_reg' ].values)
    print ("4% overall precision and recall: " + str (precision_recall_rmse['precision recall']))
    print ("Total Model RMSE: " + str (precision_recall_rmse['rmse']))

    print ('Exporting Overall precision_recall_rmse')
    pd.DataFrame(precision_recall_rmse).to_csv (r'/home/ubuntu/model_test/model_output/precision_recall_rmse.csv', index=False)

    ##############################################################################

    #Calculate Monthly Precision, Recall and RMSE
    month_year = list(gtruth.month_year.unique())
    precision_recall_rmse_monthly = {}
    for my in month_year:
        df = gtruth[gtruth.month_year == my]
        precision_recall_rmse_monthly [ my ] = \
            (precision_recall (get_4pct_accuracy (df)), get_rmse (df [ 'actual_pct_change' ].values, df [ 'yhat_reg' ].values))
    print ("Monthly prec, rec, rmse: " + str (precision_recall_rmse_monthly))

    print ('Exporting Monthly precision_recall_rmse')
    pd.DataFrame(precision_recall_rmse).to_csv (r'/home/ubuntu/model_test/model_output/precision_recall_rmse_monthly.csv', index=False)
#############################################################################

def export_gtruth(gtruth):
        # Export and Append to model_preds. This file will be created if it
        # does not exist, or will be appended to if it does
        #Creates 2 versions: model_preds.csv and model_preds_deduped.csv
        pathname = r'/home/ubuntu/model_test/model_output/'
        filedupe = 'gtruth.csv'
        filededupe = 'gtruth_deduped.csv'

        with open(pathname + filedupe, 'a') as f:
                gtruth.to_csv(f, header=f.tell() == 0, float_format='%.3f')

        d = pd.read_csv(pathname + filedupe, keep_default_na=False)
        d.drop(d.columns[0], axis=1, inplace=True)
        d.drop_duplicates(subset=['day1', 'day5', 'ticker'], inplace=True, keep='first')


        d.to_csv(pathname + filededupe, float_format='%.3f')  # rounded to two decimals

        return d


y_df, yhat_df = import_y_yhat ()

pct_chg_close = y_df [ 'actual_pct_change' ]
pct_chg_low = y_df [ 'low_pct_change' ]
pct_chg_low_pos = y_df [ y_df.loc [ :, 'actual_pct_change' ] >= 0 ].loc [ :, 'low_pct_change' ]

z_close = get_histogram (pct_chg_close, 'Close')
z_low = get_histogram (pct_chg_low, 'Low')
z_low_pos = get_histogram (pct_chg_low_pos, 'Low Pos')

# Print out the Percentiles#####################
pct_list = [ -0.025, -0.04, -0.051, 0.025, 0.04, 0.051 ]
z_close_perc, z_low_perc, z_low_perc_pos = [ ], [ ], [ ]
for i in pct_list:
    z_close_perc.append (get_percentile (z_close, i, 'Close'))
print ('------------------------------------------------------')
for i in pct_list [ 0:3 ]:
    z_low_perc.append (get_percentile (z_low, i, 'Low'))
print ('------------------------------------------------------')
for i in pct_list [ 0:3 ]:
    z_low_perc_pos.append (get_percentile (z_low_pos, i, 'Low Pos'))


gtruth = merge_y_yhat (y_df, yhat_df)
prec_rec_rmse(gtruth)
export_gtruth(gtruth)

gtruth_4pct = get_4pct_accuracy(gtruth)

