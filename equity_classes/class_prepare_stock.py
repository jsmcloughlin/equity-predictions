import requests_html
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
from yahoo_fin import stock_info
# from yahoo_fin.options import *
import stockstats
from stockstats import StockDataFrame as sdf
from pandas_datareader import data as pdr
import csv
import os
import yfinance as yf
import numpy as np
from stockstats import StockDataFrame as sdf
#from equity_classes import class_prepare_stock

yf.pdr_override ()  # Override yfinance API in order to use Pandas DataReader
pd.options.mode.chained_assignment = None  # turn of chain warning

class yfinance_scrape:

    def __init__(self, ticker):
        self.ticker = ticker
        end = datetime.date.today ()
        self.end = datetime.datetime (end.year, end.month, end.day)  # to midnight of that day
        self.start = (end - relativedelta (years=8))

        if __name__ == '__main__':
            yfinance_scrape().print_init()
        pass

    def print_init(self):
        print("Ticker is: " +str(self.ticker))
        print ("Start is: " + str (self.start))
        print ("End is: " + str (self.end))

    def get_ticker_data(self):
        '''Takes a list of stock symbols and appends the open, close,
        high, low and trading day data from Yahoo Finance for that
        period.'''
        # return stock_info.get_data (ticker, start_date=start, end_date=end)
        df = pdr.get_data_yahoo (self.ticker, self.start, self.end)
        df['ticker'] = self.ticker
        return df

    def get_indicator_data(self, df):
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

        x_ind = sdf.retype (df [ df [ 'ticker' ] == self.ticker ])
        for m in indicators:
            x_ind [ x_ind [ m ] == x_ind [ m ] ]

        return pd.DataFrame(x_ind.copy ())

    def get_movingavg(self, df, x):
        '''Takes in a df and numeric moving avg based on the number of days'''
        return df.loc [ :, 'close' ].rolling (x).mean ()

    def get_prevclose(self, df, x):
        '''Takes in a df and variable to return the % change from the previous day'''
        return (df.loc [ :, x ] - df.loc [ :, x ].shift (1)) / df.loc [ :, x ].shift (1)

    def get_indicators(self, df):
        '''Creates a list of dfs, one for each stock - then
        adds moving average, previous close and volume data
        Takes as input a dataframe of stock data'''
        df [ "ma20" ] = self.get_movingavg (df, 20)
        df [ "ma50" ] = self.get_movingavg (df, 50)
        df [ "ma200" ] = self.get_movingavg (df, 200)
        df [ "prev_close_ch" ] = self.get_prevclose (df, 'close')
        df [ "prev_volume_ch" ] = self.get_prevclose (df, 'volume')

        return df.dropna()


    def run_cps(self):
        '''Run the following class functions:
        get_ticker_data(self)
        get_indicator_data(self, df)
        get_indicators(self, df)
        Returns daily OHLC and indicator data for an equity, added at class instantiation
        '''

        return self.get_indicators (self.get_indicator_data (self.get_ticker_data ()))
