import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
from yahoo_fin import stock_info
# from yahoo_fin.options import *
import requests_html
import stockstats
from stockstats import StockDataFrame as sdf
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override ()  # Override yfinance API in order to use Pandas DataReader
import numpy as np
import os

end = datetime.date.today ()
end = datetime.datetime (end.year, end.month, end.day)  # to midnight of that day

#Going back 3 years ensures that we can accumulate a 200 day MA for the most recent 100 days
#These are isolated after attaching the 200 day MA
start = (end - relativedelta (years=3))

# return list of Dow tickers
dow_tickers = pd.Series (stock_info.tickers_dow ()).unique()
sp_tickers = pd.Series (stock_info.tickers_sp500 ()).unique()
nasdaq_tickers = pd.Series (stock_info.tickers_nasdaq ()).unique()
all_tickers = np.unique(np.hstack((nasdaq_tickers, dow_tickers, sp_tickers)))


indicators = [ 'dma', 'volume_delta', 'close_12_ema', 'close_26_ema', 'macd', 'macd_9_ema', 'macds', 'macdh' ]


# Functions
# Returns the OCHLV for a stock over a 10 year period
def get_ticker_data(ticker):
    '''Takes a list of stock symbols and appends the open, close,
    high, low and trading day data from Yahoo Finance for that
    period.'''
    # return stock_info.get_data (ticker, start_date=start, end_date=end)
    return pdr.get_data_yahoo (ticker, start, end)


def get_indicator_data(symbols, symbol_data):
    '''Takes a list of symbols and the corresponding open, close,
    high, low and trading day data - then accesses the stockdataframe
    class to assign the MACD, EMA and other indicators for that time
    period.
        DMA, difference of 10 and 50 moving average - stock['dma']
        Volume delta against previous day - stock['volume_delta']
        MACD - stock['macd']
        MACD signal line - stock['macds']
        MACD histogram - stock['macdh']'''
    indicators = [ 'dma', 'volume_delta', 'close_12_ema', 'close_26_ema', 'macd', 'macd_9_ema', 'macds', 'macdh' ]
    indicator_data = pd.DataFrame ([ ])

    for i, j in enumerate (symbols):
        x_ind = sdf.retype (symbol_data [ symbol_data [ 'ticker' ] == j ])
        for m in indicators:
            x_ind [ x_ind [ m ] == x_ind [ m ] ]
        try:
            if i == 0:
                indicator_data = x_ind.copy ()
            else:
                indicator_data = indicator_data.append (pd.DataFrame (x_ind))
        except:
            continue
    return indicator_data


# Takes a list of ticker symbols and provides back a list of trade points
# date, open, close, high, low, volume
ticker_data = pd.DataFrame ([ ])
for ticker in dow_tickers:
    df = pd.DataFrame (get_ticker_data (ticker)).assign (ticker=ticker)
    ticker_data = ticker_data.append (df)

ticker_data.index.names = [ 'date' ]  # assigns a name to the index
ticker_data.reset_index (inplace=True)  # resets the index so that tradeday now becomes a column

# stock_ohc = pd.DataFrame ([ ])
stock_data = get_indicator_data (dow_tickers, ticker_data)


def get_movingavg(df, x):
    '''Takes in a df and numeric moving avg based on the number of days'''
    return df['close'].rolling(x).mean()



def get_prevclose(df, x):
    '''Takes in a df and variable to return the % change from the previous day'''
    # return (df [ x ] - df [ x ].shift (1)) / df [ x ].shift (1)
    return (df[x] - df[x].shift (1)) / df[x].shift (1)
    #return (df.loc [ :, x ] - df.loc [ :, x ].shift (1)) / df.loc [ :, x ].shift (1)

#Convert the df of stock data to a list of dataframes - one for each ticker
pd.options.mode.chained_assignment = None #turn of chain warning
def get_indicators(df):
    '''Creates a list of dfs, one for each stock - then
    adds moving average, previous close and volume data
    Takes as input a dataframe of stock data'''
    key_names = df['ticker'].unique().tolist()
    df_history, df_recent = {}, {}
    for i in key_names:
        df_hist = df[(df.loc[:, 'ticker'] == i)]
        df_hist["ma20"] = get_movingavg(df_hist, 20)
        df_hist["ma50"] = get_movingavg(df_hist, 50)
        df_hist["ma200"] = get_movingavg(df_hist, 200)
        df_hist["prev_close_ch"] = get_prevclose(df_hist, 'close')
        df_hist["prev_volume_ch"] = get_prevclose(df_hist, 'volume')

        df_history[i] = df_hist #append each stock baxk to df_history

        df_hist = df_hist.iloc[0:0] #empty the df for reuse
    return df_history

#stock_history, stock_recent = pd.DataFrame(), pd.DataFrame()
stock_ind = {}
stock_ind = get_indicators(stock_data)

def get_n_days(data, days):
    '''Gets the last n days - used for the predictions - next 5 days requires the 15 day historical input'''
    df={}
    for i in data.keys():
        df[i] = data[i].iloc[-days:, :]
    return df

stock_recent = {}
stock_recent = get_n_days(stock_ind, 50)

try:
    if os.path.exists('/home/ubuntu/model_test/export_files/stock_recent.csv'):
        os.remove('/home/ubuntu/model_test/export_files/stock_recent.csv') #this deletes the file
except:
    print("Stock Recent does not exist")#add this to prevent errors


f = open (r'/home/ubuntu/model_test/export_files/stock_recent.csv', "w")
for df in stock_recent:
    stock_recent[df].to_csv(f, header=False)
f.close()
