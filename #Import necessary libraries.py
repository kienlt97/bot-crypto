#Import necessary libraries
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
import numpy as np

def load_historic_data(symbol):
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

def cal_price(df):
    # Calculate RSI
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('time')
    df['returns'] = np.log(df['closeAsk'] / df['closeAsk'].shift(1))
    df['delta'] = df['returns'].diff()
    # Calculate RSI
    # RS = Average gain of up periods during the specified time frame /
    # Average loss of down periods during the specified time frame
    df['avg_gain'] = 0.00
    df['avg_loss'] = 0.00
    periods = 14
    for i in range(periods, len(df)):
        df.loc[df.index[i], 'avg_gain'] = ((df['delta'][i-periods+1:i+1][df['delta']
        df.loc[df.index[i], 'avg_loss'] = ((abs(df['delta'][i-periods+1:i+1][df['del
        df['RS'] = df['avg_gain']/df['avg_loss']
        df['RSI'] = 100 - (100/(1+df['RS']))
        # Calculate Bollinger Bands
        # Calculate rolling mean and standard deviation
        df['sma'] = df['closeAsk'].rolling(window=20).mean()
        df['std'] = df['closeAsk'].rolling(window=20).std()
        # Calculate upper and lower bands
        df['upper_band'] = df['sma'] + (df['std'] * 2)
        df['lower_band'] = df['sma'] - (df['std'] * 2)
        # Trading Strategy
        # If RSI is less than 30 and price is below lower band, buy
        # If RSI is greater than 70 and price is above upper band, sell
        # Initialize empty buy and sell lists
        buy_list = []
        sell_list = []
        for i in range(len(df)):
        # If RSI is less than 30 and price is below lower band
            if df['RSI'][i] < 30 and df['closeAsk'][i] < df['lower_band'][i]:
                buy_list.append(df.index[i])
        # If RSI is greater than 70 and price is above upper band
            elif df['RSI'][i] > 70 and df['closeAsk'][i] > df['upper_band'][i]:
                sell_list.append(df.index[i])

        # Print buy and sell signals
    print("Buy signals: ", buy_list)
    print("Sell signals: ", sell_list)