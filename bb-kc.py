# IMPORTING PACKAGES
import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
from math import floor
from termcolor import colored as cl
import os
import matplotlib.dates as mpl_dates

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)

# EXTRACTING STOCK DATA
# EXTRACTING STOCK DATA


def get_historical_data(symbol):
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
        df = df[['date', 'open', 'high', 'low', 'close','volume']]
        return df
    else:
        return None

# BOLLINGER BANDS CALCULATION
def sma(df, lookback):
    sma = df.rolling(lookback).mean()
    return sma

def get_bb(df, lookback):
    std = df.rolling(lookback).std()
    upper_bb = sma(df, lookback) + std * 2
    lower_bb = sma(df, lookback) - std * 2
    middle_bb = sma(df, lookback)
    return upper_bb, middle_bb, lower_bb

# KELTNER CHANNEL CALCULATION
def get_kc(high, low, close, kc_lookback, multiplier, atr_lookback):
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift()))
    tr3 = pd.DataFrame(abs(low - close.shift()))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
    atr = tr.ewm(alpha = 1/atr_lookback).mean()

    kc_middle = close.ewm(kc_lookback).mean()
    kc_upper = close.ewm(kc_lookback).mean() + multiplier * atr
    kc_lower = close.ewm(kc_lookback).mean() - multiplier * atr

    return kc_middle, kc_upper, kc_lower

# RSI CALCULATION
def get_rsi(close, lookback):
    ret = close.diff()
    up = []
    down = []
    for i in range(len(ret)):
        if ret[i] < 0:
            up.append(0)
            down.append(ret[i])
        else:
            up.append(ret[i])
            down.append(0)
    up_series = pd.Series(up)
    down_series = pd.Series(down).abs()
    up_ewm = up_series.ewm(com = lookback - 1, adjust = False).mean()
    down_ewm = down_series.ewm(com = lookback - 1,adjust = False).mean()
    rs = up_ewm/down_ewm
    rsi = 100 - (100 / (1 + rs))
    rsi_df = pd.DataFrame(rsi).rename(columns  ={0:'rsi'}).set_index(close.index)
    rsi_df = rsi_df.dropna()

    return rsi_df[3:]

# TRADING STRATEGY
def bb_kc_rsi_strategy(prices, upper_bb, lower_bb, kc_upper, kc_lower, rsi):
    buy_price = []
    sell_price = []
    bb_kc_rsi_signal = []
    signal = 0
    lower_bb = lower_bb.to_numpy()
    kc_lower = kc_lower.to_numpy()
    upper_bb = upper_bb.to_numpy()
    kc_upper = kc_upper.to_numpy()
    rsi = rsi.to_numpy()

    for i in range(len(prices)):
        if lower_bb[i] < kc_lower[i] and upper_bb[i] > kc_upper[i] and rsi[i] < 30:
            if signal != 1:
                buy_price.append(prices[i])
                sell_price.append(np.nan)
                signal = 1
                bb_kc_rsi_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_kc_rsi_signal.append(0)
        elif lower_bb[i] < kc_lower[i] and upper_bb[i] > kc_upper[i] and rsi[i] > 70:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(prices[i])
                signal = -1
                bb_kc_rsi_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_kc_rsi_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            bb_kc_rsi_signal.append(0)
    return buy_price, sell_price, bb_kc_rsi_signal


# BACKTESTING

if __name__ == '__main__':

    symbol = 'ETH-USD'

    df = get_historical_data(symbol)
    df['upper_bb'], df['middle_bb'], df['lower_bb'] = get_bb(df['close'], 20)
    df['kc_middle'], df['kc_upper'], df['kc_lower'] = get_kc(df['high'], df['low'], df['close'], 20, 2, 10)
    

    df['rsi_14'] = get_rsi(df['close'], 14)
    df = df.dropna()
    # print(df)
    
    buy_price, sell_price, bb_kc_rsi_signal = bb_kc_rsi_strategy(df['close'], df['upper_bb'], df['lower_bb'], df['kc_upper'], df['kc_lower'], df['rsi_14'])
    
    # POSITION
    position = []
    for i in range(len(bb_kc_rsi_signal)):
        if bb_kc_rsi_signal[i] > 1:
            position.append(0)
        else:
            position.append(1)

    for i in range(len(df['close'])):
        if bb_kc_rsi_signal[i] == 1:
            position[i] = 1
        elif bb_kc_rsi_signal[i] == -1:
            position[i] = 0
        else:
            position[i] = position[i-1]
    # print(position)


    kc_upper = df['kc_upper']
    kc_lower = df['kc_lower']
    upper_bb = df['upper_bb']
    lower_bb = df['lower_bb']
    rsi = df['rsi_14']
    close_price = df['close']
    bb_kc_rsi_signal = pd.DataFrame(bb_kc_rsi_signal).rename(columns = {0:'bb_kc_rsi_signal'}).set_index(df.index)
    position = pd.DataFrame(position).rename(columns ={0:'bb_kc_rsi_position'}).set_index(df.index)
    frames = [close_price, kc_upper, kc_lower, upper_bb, lower_bb, rsi, bb_kc_rsi_signal, position]
    strategy = pd.concat(frames, join = 'inner', axis= 1)

    df_ret = pd.DataFrame(np.diff(df['close'])).rename(columns = {0:'returns'})
    bb_kc_rsi_strategy_ret = []

    strategy = strategy['bb_kc_rsi_position'].to_numpy()
    df_ret = df_ret['returns'].to_numpy()
    df_close = df['close'].to_numpy()

    for i in range(len(df_ret)):
        returns = df_ret[i]*strategy[i]
        bb_kc_rsi_strategy_ret.append(returns)
        
    bb_kc_rsi_strategy_ret_df = pd.DataFrame(bb_kc_rsi_strategy_ret).rename(columns = {0:'bb_kc_rsi_returns'})
    investment_value = 100000
    bb_kc_rsi_investment_ret = []
    bb_kc_rsi_returns = bb_kc_rsi_strategy_ret_df['bb_kc_rsi_returns'].to_numpy()

    for i in range(len(bb_kc_rsi_strategy_ret_df['bb_kc_rsi_returns'])):
        number_of_stocks = floor(investment_value/df_close[i])
        returns = number_of_stocks*bb_kc_rsi_returns[i]
        bb_kc_rsi_investment_ret.append(returns)

    bb_kc_rsi_investment_ret_df = pd.DataFrame(bb_kc_rsi_investment_ret).rename(columns = {0:'investment_returns'})
    total_investment_ret = round(sum(bb_kc_rsi_investment_ret_df['investment_returns']), 2)
    profit_percentage = floor((total_investment_ret/investment_value)*100)
    print(cl('Profit gained from the BB KC RSI strategy by investing $100k in df : {}'.format(total_investment_ret), attrs = ['bold']))
    print(cl('Profit percentage of the BB KC RSI strategy : {}%'.format(profit_percentage), attrs = ['bold']))