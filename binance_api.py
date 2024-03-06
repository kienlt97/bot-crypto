
from binance.client import Client

class binance_api:
    def __init__(self, radius):
        self.radius = radius

    def get_data_frame():
        # valid intervals - 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
        # request historical candle (or klines) data using timestamp from above, interval either every min, hr, day or month
        # starttime = '30 minutes ago UTC' for last 30 mins time
        # e.g. client.get_historical_klines(symbol='ETHUSDTUSDT', '1m', starttime)
        # starttime = '1 Dec, 2017', '1 Jan, 2018'  for last month of 2017
        # e.g. client.get_historical_klines(symbol='BTCUSDT', '1h', "1 Dec, 2017", "1 Jan, 2018")
        starttime = '3 day ago UTC'  # to start for 1 day ago
        interval = '5m'
        bars = client.get_historical_klines(symbol, interval, starttime)
        
        for line in bars:        # Keep only first 5 columns, "date" "open" "high" "low" "close"
            del line[5:]

        df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close']) #  2 dimensional tabular data
        return df

    def request_binance():
        api_key = 'lwaoJYVsMOYVNIBXma32k3PoNzhB5kJ7A6TcRv6cQEqPUTEBMBZHPWiFKZ7bIRqM'     # passkey (saved in bashrc for linux)
        api_secret = 'aDpaIwHf9GVJBiI36aUye5Y2zd1LKCPAUjKIMD9N5ZhzJBqNOJN6Jy09Waw7HBjO' # secret (saved in bashrc for linux)

        client = Client(api_key, api_secret, tld ='us')
        print("Using Binance TestNet Server")
        symbol = 'ETH-USDT'
        return get_data_frame()