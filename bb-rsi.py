from datetime import datetime
import sys
import os
import math
import numpy as np
import pandas as pd
import pandas_ta as ta
from ta.volatility import BollingerBands
from ta.volatility import KeltnerChannel
from ta.trend import STCIndicator
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
US_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools
import matplotlib.dates as mpl_dates
from binance.client import Client
import pprint
import operator
import time
start_time = time.time()

def load_data_file(symbol):
    # valid intervals - 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    # request historical candle (or klines) data using timestamp from above, interval either every min, hr, day or month
    # starttime = '30 minutes ago UTC' for last 30 mins time
    # e.g. client.get_historical_klines(symbol='ETHUSDTUSDT', '1m', starttime)
    # starttime = '1 Dec, 2017', '1 Jan, 2018'  for last month of 2017
    # e.g. client.get_historical_klines(symbol='BTCUSDT', '1h', '1 Dec, 2017', '1 Jan, 2018')
    starttime = '1 day ago UTC'  # to start for 1 day ago
    interval = '5m'
    bars = client.get_historical_klines(symbol, interval, starttime)

    for line in bars:        # Keep only first 5 columns, 'date' 'open' 'high' 'low' 'close'
        del line[5:]

    df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close']) #  2 dimensional tabular data
    return df

def load_data_file1(symbol):
    #  Create output file name
    daily_file_name = '{}_1d_1m.csv'.format(symbol)
    daily_full_path = os.path.join('D:\\New folder\\Bollinger bands\\', daily_file_name)

    #  Check if output file exists
    if os.path.exists(daily_full_path):
        df = pd.read_csv(daily_full_path, parse_dates=True)
        if df.empty:
            return None
        df = df[['date', 'open', 'high', 'low','close']]
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index).strftime('%d-%m-%Y') # index set to first column = date_and_time
        return df
    else:
        return None


def calculate_tis(df):
    bb_window = 20
    indicator_bb = BollingerBands(close=df['close'], window=bb_window, window_dev=2)

    # Add Bollinger Bands features
    df['BB_mid'] = indicator_bb.bollinger_mavg()
    df['BB_high'] = indicator_bb.bollinger_hband()
    df['BB_low'] = indicator_bb.bollinger_lband()
        
    rsi_window = 14
    df['RSI'] = ta.rsi(df['close'].astype(float), window=rsi_window)
    
    stc_window_slow = 50
    stc_window_fast = 23
    stc_cycle = 10
    indicator_stc = STCIndicator(close=df['close'].astype(float), window_slow=stc_window_slow, window_fast=stc_window_fast, cycle=stc_cycle, smooth1=3, smooth2=3)

    # Add features
    df['STC'] = indicator_stc.stc()
    return df   

def calculate_signals(df, rsi_entry, rsi_exit, stc_entry, stc_exit):

    df['RSI_entry_ind'] = np.where(np.logical_and((df['RSI'] > rsi_entry), (df['RSI'].shift() <= rsi_entry)), 1, 0)
    df['RSI_exit_ind'] = np.where(np.logical_and((df['RSI'] < rsi_exit), (df['RSI'].shift() >= rsi_exit)), 1, 0)

    #  Calculate upper / lower boundary for BB
    close_prices = df['close'].astype(float).to_numpy()
    max_close = np.amax(close_prices)
    min_close = np.amin(close_prices)
    diff_close = max_close - min_close
    df['BB_low_adj'] = df['BB_low'] + (diff_close * 0.09)
    df['BB_entry_ind'] = np.where((df['close'].astype(float) <= df['BB_low_adj']), 1, 0)
    df['BB_high_adj'] = df['BB_high'] - (diff_close * 0.07)
    df['BB_exit_ind'] = np.where((df['close'].astype(float) >= df['BB_high_adj']), 1, 0)
    
    df['STC_entry_ind'] = np.where(np.logical_and(df['STC'] > stc_entry, df['STC'].shift() <= stc_entry), 1, 0)
    df['STC_exit_ind'] = np.where(np.logical_and(df['STC'] < stc_exit, df['STC'].shift() >= stc_exit), 1, 0)

    return df

def execute_strategy(df):

    close_prices = df['close'].astype(float).to_numpy()
    rsi_entry = df['RSI_entry_ind'].to_numpy()
    rsi_exit = df['RSI_exit_ind'].to_numpy()  
    bb_entry = df['BB_entry_ind'].to_numpy()
    bb_exit = df['BB_exit_ind'].to_numpy()
    stc_entry = df['STC_entry_ind'].to_numpy()
    stc_exit = df['STC_exit_ind'].to_numpy()  

    required_entry_signals = 3
    required_exit_signals = 3

    entry_prices = []
    exit_prices = []
    hold = 0

    for i in range(len(close_prices)):
        current_price = close_prices[i]
        num_entry_signals = 0
        num_exit_signals = 0

        lookback_ind = i - 20
        if lookback_ind >= 0:
            rsi_entry_lookback = rsi_entry[lookback_ind:i]
            if 1 in rsi_entry_lookback:
                num_entry_signals += 1
            rsi_exit_lookback = rsi_exit[lookback_ind:i]
            if 1 in rsi_exit_lookback:
                num_exit_signals += 1
            bb_entry_lookback = bb_entry[lookback_ind:i]
            if 1 in bb_entry_lookback:
                num_entry_signals += 1
            bb_exit_lookback = bb_exit[lookback_ind:i]
            if 1 in bb_exit_lookback:
                num_exit_signals += 1
    
            stc_entry_lookback = stc_entry[lookback_ind:i]
            if 1 in stc_entry_lookback:
                num_entry_signals += 1
            stc_exit_lookback = stc_exit[lookback_ind:i]
            if 1 in stc_exit_lookback:
                num_exit_signals += 1   
                
        #  Verify Entry indicators
        if hold == 0 and num_entry_signals >= required_entry_signals:
            entry_prices.append(current_price)
            exit_prices.append(np.nan)
            hold = 1
            # print('-----------------------------------close_prices: ',close_prices[i], 'lookback_ind:', (lookback_ind + 2), ',i:', (i+2))
            # print("rsi_entry_lookback ",current_price)
            # print("rsi_entry",rsi_entry[lookback_ind:i])
            # print("bb_entry",bb_entry[lookback_ind:i])
            # print("stc_entry",stc_entry[lookback_ind:i])
        #  Exit strategy
        elif hold == 1 and num_exit_signals >= required_exit_signals:
            entry_prices.append(np.nan)
            exit_prices.append(current_price)
            hold = 0
        else:
            #  Neither entry nor exit
            entry_prices.append(np.nan)
            exit_prices.append(np.nan)

    return entry_prices, exit_prices

def plot_graph(symbol, df, entry_prices, exit_prices):
    fig = make_subplots(rows=3, cols=1, subplot_titles=['Close + BB','RSI','STC'])
    
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms') # index set to first column = date_and_time
    
    #  Plot close price
    fig.add_trace(go.Line(x = df.index, y = df['close'].astype('float'), line=dict(color='blue', width=1), name='Close'), row = 1, col = 1)
    
    #  Plot bollinger bands
    bb_high = df['BB_high']
    bb_mid = df['BB_mid']
    bb_low = df['BB_low']
    fig.add_trace(go.Line(x = df.index, y = bb_high, line=dict(color='green', width=1), name='BB High'), row = 1, col = 1)
    fig.add_trace(go.Line(x = df.index, y = bb_mid, line=dict(color='#ffd866', width=1), name='BB Mid'), row = 1, col = 1)
    fig.add_trace(go.Line(x = df.index, y = bb_low, line=dict(color='red', width=1), name='BB Low'), row = 1, col = 1)
    
    #  Plot RSI
    fig.add_trace(go.Line(x = df.index, y = df['RSI'], line=dict(color='blue', width=1), name='RSI'), row = 2, col = 1)

    #  Plot STC
    fig.add_trace(go.Line(x = df.index, y = df['STC'], line=dict(color='blue', width=1), name='STC'), row = 3, col = 1)

    #  Add buy and sell indicators
    fig.add_trace(go.Scatter(x=df.index, y=entry_prices, marker_symbol='arrow-up', marker=dict(
        color='green',size=15
    ),mode='markers',name='Buy'))
    fig.add_trace(go.Scatter(x=df.index, y=exit_prices, marker_symbol='arrow-down', marker=dict(
        color='red',size=15
    ),mode='markers',name='Sell'))
        
    fig.update_layout(
        title={'text':f'{symbol} with BB-RSI-STC', 'x':0.5},
        autosize=False,
        width=1500,height=3000)
    fig.update_yaxes(range=[0,1000000000],secondary_y=True)
    fig.update_yaxes(visible=True, secondary_y=True)  #hide range slider

    fig.show()

#  Assuming reinvesting the proceeds
def calculate_profit(start_investment, entry_prices, exit_prices):
    hold = 0
    profit = 0
    available_funds = start_investment
    cost = 0
    num_stocks = 0
    for i in range(len(entry_prices)):
        current_entry_price = entry_prices[i]
        current_exit_price = exit_prices[i]

        if not math.isnan(current_entry_price) and hold == 0:
            num_stocks = available_funds / current_entry_price
            cost = num_stocks * current_entry_price
            hold = 1

        elif hold == 1 and not math.isnan(current_exit_price):
            hold = 0
            proceeds = num_stocks * current_exit_price
            profit += proceeds - cost

    return math.trunc(profit)


def log_out(df):
    output = df[['date','open','high','low','close','BB_mid','BB_high','BB_low','RSI','STC','RSI_entry_ind','BB_entry_ind',
                  'STC_entry_ind','RSI_exit_ind','BB_exit_ind','STC_exit_ind']]
    # output = output1[ (output1['RSI_entry_ind'] == 1)]

    # output = output1[(output1['RSI_entry_ind'] != 0) | (output1['BB_entry_ind'] != 0) | (output1['STC_entry_ind'] != 0) |
    #                   (output1['RSI_exit_ind'] != 0 ) | (output1['BB_exit_ind'] != 0 ) | (output1['STC_exit_ind'] != 0)]

    output.set_index('date', inplace=True)
    output.index = pd.to_datetime(output.index, unit='ms') # index set to first column = date_and_time

    with open('output-btc.txt', 'w') as f:
        f.write(output.to_string())
    
if __name__ == '__main__':

    symbol = 'ETHUSDT'   # Change symbol here e.g. BTCUSDT, BNBBTC, ETHUSDT, NEOBTC
    api_key = 'lwaoJYVsMOYVNIBXma32k3PoNzhB5kJ7A6TcRv6cQEqPUTEBMBZHPWiFKZ7bIRqM'     # passkey (saved in bashrc for linux)
    api_secret = 'aDpaIwHf9GVJBiI36aUye5Y2zd1LKCPAUjKIMD9N5ZhzJBqNOJN6Jy09Waw7HBjO' # secret (saved in bashrc for linux)

    client = Client(api_key, api_secret, tld ='us')
    print("Using Binance TestNet Server")
    
    # symbol = 'ETH-USD'
    df = load_data_file(symbol)
    # print(df)
    start_investment = 1000
    df = df.tail(1440)
    df = calculate_tis(df)
    # test_rsi(df)
    myfile = open('out-put.txt', 'w')
    # for rsi_entry in range(100):
    #     for rsi_exit in range(100):
    #         for stc_entry in range(100):
    #             for stc_exit in range(100):
    df = calculate_signals(df, 34, 84, 73, 97)
    entry_prices, exit_prices = execute_strategy(df)
    plot_graph(symbol, df, entry_prices, exit_prices)

    profit = calculate_profit(start_investment, entry_prices, exit_prices)
    trades = np.array(exit_prices)
    num_trades = len(trades[~np.isnan(trades)])
    # log_out(df)
    interest_percentage = round(profit / start_investment, 2) * 100
    
    # idx = 'rsi_entry:'+ str(rsi_entry) + '_rsi_exit:' + str(rsi_exit) + '_stc_entry:' + str(stc_entry) +'_stc_exit:' +str(stc_exit)
    # map[idx] = profit
    # rs = idx +' - ' + str(profit)
    # if(profit > 0):
    #     myfile.write("%s\n" % rs)
    # # print(f'Number of trades: {num_trades}')
    print(f'Profit with start_investment of ${start_investment}: ${profit} : {interest_percentage}%')    
    # entry_prices[np.isnan(entry_prices)] = 0
    # exit_prices[np.isnan(exit_prices)] = 0
    # exit_prices = [0 if math.isnan(x) else x for x in exit_prices]
    # entry_prices = [0 if math.isnan(x) else x for x in entry_prices]

    # print(f'total buy {np.sum(entry_prices)}, total sell {np.sum(exit_prices)}, - after: {np.sum(exit_prices) - np.sum(entry_prices)}')
   
    # max_i = max(map.items(), key=operator.itemgetter(1))[0]
    # print(f'i rsi entry = {max_i} - profit max = {map[max_i]}')
    # print("--- %s seconds ---" % (time.time() - start_time))
