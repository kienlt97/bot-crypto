import os
import math
import numpy as np
import pandas as pd
import datetime
import time
import random
from datetime import date
import pandas_ta as ta
from ta.volatility import BollingerBands
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from yahoo_fin import stock_info as si
import matplotlib.dates as mpl_dates
from binance.client import Client
import pprint

# def load_historic_data(symbol):
#     today = datetime.date.today()
#     today_str = today.strftime('%Y-%m-%d')
#     # Download data from Yahoo Finance
#     print(symbol)
#     try:
#         df = si.get_data(symbol, start_date=None, end_date=today_str, index_as_date=False)
#         return df
#     except:
#         print('Error loading stock data for ' + symbol)
#         return None

def load_historic_data(symbol):
    # valid intervals - 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    # request historical candle (or klines) data using timestamp from above, interval either every min, hr, day or month
    # starttime = '30 minutes ago UTC' for last 30 mins time
    # e.g. client.get_historical_klines(symbol='ETHUSDTUSDT', '1m', starttime)
    # starttime = '1 Dec, 2017', '1 Jan, 2018'  for last month of 2017
    # e.g. client.get_historical_klines(symbol='BTCUSDT', '1h', '1 Dec, 2017', '1 Jan, 2018')
    starttime = '7 day ago UTC'  # to start for 1 day ago
    interval = '5m'
    bars = client.get_historical_klines(symbol, interval, starttime)
    
    for line in bars:        # Keep only first 5 columns, 'date' 'open' 'high' 'low' 'close'
        del line[6:]

    df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'adjclose','volume']) #  2 dimensional tabular data

    return df
    
def load_historic_data1(symbol):
    #  Create output file name
    daily_file_name = '{}_1d_2m.csv'.format(symbol)
    daily_full_path = os.path.join('D:\\New folder\\Bollinger bands\\', daily_file_name)

    #  Check if output file exists
    if os.path.exists(daily_full_path):
        df = pd.read_csv(daily_full_path, parse_dates=True)
        if df.empty:
            return None
        df['date'] = pd.to_datetime(df['date'])
        df['date'] = df['date'].apply(mpl_dates.date2num)
        df = df[['date', 'open', 'high', 'low', 'adjclose','volume']]
        return df
    else:
        return None
    
def calculate_bollinger_bands(df):
    # Initialize Bollinger Bands Indicator
    indicator_bb = BollingerBands(close=df['adjclose'].astype('float'), window=20, window_dev=2)

    # Add Bollinger Bands features
    df['BB_mid'] = indicator_bb.bollinger_mavg()
    df['BB_high'] = indicator_bb.bollinger_hband()
    df['BB_low'] = indicator_bb.bollinger_lband()

    return df

def calculate_rsi(df):
    df['RSI'] = ta.rsi(df['adjclose'].astype('float'), length=14)
    return df

def calculate_ema(df):
    df['EMA_20'] = ta.ema(df['adjclose'].astype('float'), length=20)
    return df

def calculate_obv(df):
    df['OBV'] = ta.obv(df['adjclose'].astype('float'), df['volume'].astype('float'))
    return df

def calculate_strategy_rules(df, i):
    #  Entry Rule 1: Close price above Mid Bollinger Band
    df['BB_mid_ind'] = np.where(df['adjclose'].astype('float') > df['BB_mid'].astype('float'), 1, 0)

    #  Entry Rule 2: RSI > 50
    df['RSI_ind'] = np.where(df['RSI'].astype('float') > 50, 1, 0)

    #  Entry Rule 3: OBV is above the 20 EMA
    df['OBV_ind'] = np.where(df['OBV'].astype('float') > df['EMA_20'].astype('float'), 1, 0)
    
    #  Stop Loss and exit rule
    df['stop_loss_ind'] = np.where(df['adjclose'].astype('float') < df['BB_low'].astype('float'), 1, 0)
    # pprint.pprint(df[['EMA_20','OBV']])

    return df


#  Assumption: Entry signals need to align in the last 10 trading days in order generate a buy signal
def execute_strategy(close_prices, bb_mid_inds, rsi_inds, obv_inds, stop_inds):
    lookback_period = 10
    entry_prices = []
    exit_prices = []
    entry_signal = 0
    exit_signal = 0
    
    for i in range(len(close_prices)):
        #  Evaluate entry strategy
        bb_signal = 0
        rsi_signal = 0
        obv_signal = 0
        check_entry_signals = 0
        #  Look for combined signals considering the last x trading days
        for j in range(lookback_period):
            lookback_ind = i - lookback_period + j
            if lookback_ind < 0:
                continue
            if bb_mid_inds[lookback_ind] == 1:
                bb_signal = 1
            if rsi_inds[lookback_ind] == 1:
                rsi_signal = 1      
            if obv_inds[lookback_ind] == 1:
                obv_signal = 1    
        #  All entry signals have to align
        if bb_signal == 1 and rsi_signal == 1 and obv_signal == 1:
            check_entry_signals = 1
            
        #  Add entry prices
        if entry_signal == 0 and check_entry_signals == 1:
            entry_prices.append(close_prices[i])
            exit_prices.append(np.nan)  
            entry_signal = 1
            exit_signal = 0
        #  Evaluate exit strategy
        elif entry_signal == 1 and stop_inds[i] == 1:
            entry_prices.append(np.nan)
            exit_prices.append(close_prices[i]) 
            entry_signal = 0
            exit_signal = 1
        else:
            #  Neither entry nor exit
            entry_prices.append(np.nan) 
            exit_prices.append(np.nan) 
            
    return entry_prices, exit_prices

def plot_graph(df, entry_prices, exit_prices, bb_high, bb_mid, bb_low, rsi, ema, obv):
    fig = make_subplots(rows=3, cols=1)

    #  Plot close price
    fig.add_trace(go.Line(x = df.index, y = df['adjclose'].astype('float'), line=dict(color='blue', width=1), name='Close'), row = 1, col = 1)
    
    #  Plot bollinger bands
    fig.add_trace(go.Line(x = df.index, y = bb_high, line=dict(color='#ffdf80', width=1), name='BB High'), row = 1, col = 1)
    fig.add_trace(go.Line(x = df.index, y = bb_mid, line=dict(color='#ffd866', width=1), name='BB Mid'), row = 1, col = 1)
    fig.add_trace(go.Line(x = df.index, y = bb_low, line=dict(color='#ffd24d', width=1), name='BB Low'), row = 1, col = 1)
    
    # Plot RSI
    fig.add_trace(go.Line(x = df.index, y = rsi, line=dict(color='#ffb299', width=1), name='RSI'), row = 2, col = 1)
    
    # Plot EMA
    fig.add_trace(go.Line(x = df.index, y = ema, line=dict(color='#99b3ff', width=1), name='EMA'), row = 1, col = 1)
    
    # Plot OBV
    fig.add_trace(go.Line(x = df.index, y = obv, line=dict(color='hotpink', width=1), name='OBV'), row = 1, col = 1)
    
    #  Add buy and sell indicators
    fig.add_trace(go.Scatter(x=df.index, y=entry_prices, marker_symbol='arrow-up', marker=dict(
        color='green',
    ),mode='markers',name='Buy'))
    fig.add_trace(go.Scatter(x=df.index, y=exit_prices, marker_symbol='arrow-down', marker=dict(
        color='red'
    ),mode='markers',name='Sell'))
    
    fig.update_layout(
        title={'text':'BB + RSI + OBV', 'x':0.5},
        autosize=False,
        width=900,height=1200)
    fig.update_yaxes(range=[0,1000000000],secondary_y=True)
    fig.update_yaxes(visible=False, secondary_y=True)  #hide range slider
    
    fig.show()
    

def perform_analysis(start_investment, df, i):

    df = df.reset_index()
    # df = calculate_bollinger_bands(df)
    df = calculate_rsi(df)
    df = calculate_bollinger_bands(df)
    df = calculate_ema(df)
    df = calculate_obv(df)
    df = calculate_strategy_rules(df, i)

    # print(df['RSI_ind'])
    # print(df['BB_mid_ind'])
    
    arrRSI_ind = df['RSI_ind'].to_numpy()
    # print('RSI_ind', arrRSI_ind[arrRSI_ind > 0])

    arrBB_mid_ind = df['BB_mid_ind'].to_numpy()
    # print('BB_mid_ind', arrBB_mid_ind[arrBB_mid_ind > 0])

    arrOBV_ind = df['OBV_ind'].to_numpy()
    # print('OBV_ind', arrOBV_ind[arrOBV_ind > 0])

    entry_prices, exit_prices = execute_strategy(df['adjclose'],df['BB_mid_ind'], df['RSI_ind'], df['OBV_ind'], df['stop_loss_ind'])
    profit = calculate_profit(start_investment, entry_prices, exit_prices)
    plot_graph(df, entry_prices, exit_prices, df['BB_high'].astype('float'), df['BB_mid'].astype('float'), df['BB_low'].astype('float'),
                df['RSI'].astype('float'), df['EMA_20'].astype('float'), df['OBV'].astype('float'))

    return profit

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

if __name__ == '__main__':
    total_profit = 0
    # for symbol in nasdaq_100:
    api_key = 'lwaoJYVsMOYVNIBXma32k3PoNzhB5kJ7A6TcRv6cQEqPUTEBMBZHPWiFKZ7bIRqM'     # passkey (saved in bashrc for linux)
    api_secret = 'aDpaIwHf9GVJBiI36aUye5Y2zd1LKCPAUjKIMD9N5ZhzJBqNOJN6Jy09Waw7HBjO' # secret (saved in bashrc for linux)
    client = Client(api_key, api_secret, tld ='us')
    print('Using Binance TestNet Server')
   
    symbol = 'BTCUSDT'   # Change symbol here e.g. BTCUSDT, BNBBTC, ETHUSDT, NEOBTC
    df = load_historic_data(symbol)
    # print(df)
    # symbol = 'ETH-USD'
    # df = load_historic_data(symbol)

    df.reset_index(inplace=True)
    start_investment = 10000
    #  Random interval between remote fetch to avoid spam issues
    random_secs = random.uniform(0, 1)
    time.sleep(random_secs)
    for i in range(1):
    #  Run backtest
        profit = perform_analysis(start_investment, df, 50)
        print(f'Backtest profit for {i} : ${math.trunc(profit)}')
        # total_profit += profit
    
    
