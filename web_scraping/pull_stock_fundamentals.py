import datetime
from dateutil.relativedelta import relativedelta
from yahoo_fin import stock_info
from pandas_datareader import data as pdr
import yfinance as yf
from collections import Counter
import collections
from lxml.html import parse
from urllib.request import Request, urlopen
import pandas as pd
import pickle
from equity_classes import class_prepare_stock as cps

# Ensure that there is at least 5 years of trading data for each stock to trade from
# A subset is then extracted, pulling the last month, to ensure adequate volume
end = datetime.date.today ()
end = datetime.datetime (end.year, end.month, end.day)  # to midnight of that day
months = 60
start = (end - relativedelta (months=months))

def get_ticker_list(api):
    '''Returns a ticker list for a given market'''
    tickers = api

    ticker_data = pd.DataFrame ([ ])
    for ticker in tickers:
        df = pd.DataFrame (get_ticker_data (ticker)).assign (ticker=ticker)

#This try, except statement first ensures that there is at least 60 months of trading data available
#However, before appending that stock's data, a boolean mask is created to only forward one month
#of trading data. This is needed for the volume avg. In this code note, we do not need 5 years of trading
#data. That will be extracted later. These are preliminary filters.
        try: #will only forward a ticker if there is a minimum 60 months (or whatever) of trading data
            if pd.Timestamp (start) == df.index.min():
                mask = df.index >= pd.Timestamp (end - relativedelta (months=1))
                filtered_df = df[mask]
                ticker_data = ticker_data.append (filtered_df)
        except:
            continue

    ticker_data.index.names = [ 'date' ]  # assigns a name to the index
    ticker_data.reset_index (inplace=True)  # resets the index so that tradeday now becomes a column

    return ticker_data

def get_ticker_data(ticker):
    '''Takes a list of stock symbols and appends the open, close,
    high, low and trading day data from Yahoo Finance for that
    period.'''
    # return stock_info.get_data (ticker, start_date=start, end_date=end)
    return pdr.get_data_yahoo (ticker, start, end)


def get_eligible_stocks(df):
    '''Takes as input, the output from yFinance - which contains
    ticker, volume, date etc - does the following:
    1. Number of unique trading days in the period counted - stocks must
    have traded every eligible day as a pre-requisite.
    2. The daily volume for each stock is then determined and returned in a dictionary
    as a key-value pair'''

    N = df.date.nunique() #number of days counted
    ticker_day_count = Counter(df.ticker) #Number of data days for each ticker

    #At a bare minimum, only include tickers that have traded everyday during the period
    eligible_tickers = [key for (key, value) in ticker_day_count.items() if value == N]

    volume = df.groupby('ticker')['Volume'].mean()

    return volume.to_dict()

def get_ticker_sector(sectors):

    '''Takes a list of sectors and returns a dictionary of stocks :  sector
     key-value pairs that belong to each sector.'''


    headers = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3)" + " "
        "AppleWebKit/537.36 (KHTML, like Gecko)" + " " + "Chrome/35.0.1916.47" +
        " " + "Safari/537.36"]

    sector_dict = {}

    for sector in sectors:
        url = 'https://www.stockmonitor.com/sector/' + str (sector) + '/'
        print (url)

        headers_dict = {'User-Agent': headers [ 0 ]}
        req = Request (url, headers=headers_dict)
        webpage = urlopen (req)

        tree = parse (webpage)

        for element in tree.xpath ("//tbody/tr/td[@class='text-left']/a"):
            sector_dict [ element.text ] = sector

    return sector_dict




def combine_dicts(sector_dict, volume_dict):
    '''Takes as input, the output from get_ticker_sector(), which is a (k,v)
    pair, containing the ticker and sector: sector_dict
    Also takes the output from get_eligible_stocks(df), which is also a
    dictionary, called volume_dict - which is a (k,v) pair of ticker and volume.

    Both dictionaries are combined to create a sector, ticker, volume dictionary'''

    combine, combine_subset = {}, {}
    for ticker, segment in sector_dict.items():
        if volume_dict.__contains__ (ticker):
            combine[(segment, ticker)] = volume_dict[ticker]

    #keep only those stocks with > 1m shares traded average daily
    for s, v in combine.items ():
        if v > 700000: #average daily volume
            combine_subset [ s ] = v

    return combine_subset

def get_topN_stocks(sectors, combine, N):

    '''This logic rotates through all stocks within each sector, using
    the output from combine_dicts() and the sectors list, then
    uses the most_common() function to select the top N - using
    the value - in this case, volume'''

    temp, overall_dict = {}, {}
    for sector in sectors:
        for (segment, ticker), volume in combine.items ():
            if sector == segment:
                temp [ (segment, ticker) ] = volume
        overall_dict.update (collections.Counter (temp).most_common (N))

    return overall_dict



#List of sectors
sectors = [ 'basic-materials', 'communication-services', 'consumer-cyclical',
                'consumer-defensive', 'energy', 'financial-services', 'healthcare',
                'industrials', 'technology', 'utilities' ]

yf.pdr_override ()  # Override yfinance API in orde# r to use Pandas DataReader



t0 = datetime.now ()
dow_tickers = get_ticker_list(stock_info.tickers_dow ())
sp_tickers = get_ticker_list(stock_info.tickers_sp500 ())
nasdaq_tickers = get_ticker_list(stock_info.tickers_nasdaq ())
all_tickers = pd.concat ([ dow_tickers, sp_tickers, nasdaq_tickers ], axis=0, sort=False)
all_tickers.drop_duplicates(subset=['date', 'ticker'], inplace=True, keep='first')
print ("Time Taken:", datetime.now () - t0)


volume_dict = get_eligible_stocks(all_tickers)
sector_dict = get_ticker_sector(sectors)
combine = combine_dicts(sector_dict, volume_dict)
overall_dict = get_topN_stocks(sectors, combine, 75)



#Remove any stocks that do not have a minimum of 5 years trading data to train a model from

#split into random groups and then export them
with open('/home/ubuntu/model_test/export_files/all_stock_sectors.json', 'wb') as f:
  pickle.dump(combine, f)


with open('/home/ubuntu/model_test/export_files/subset_stock_sectors.json', 'wb') as f:
  pickle.dump(overall_dict, f)


#with open('/home/ubuntu/model_test/export_files/overall_dict.json', 'rb') as f:
#  overall_dict = pickle.load(f)
