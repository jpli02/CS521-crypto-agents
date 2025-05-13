import requests
import pandas as pd
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(description="Fetch cryptocurrency data.")
parser.add_argument('--coin_id', type=str, default='bitcoin', help='Coin ID (e.g., bitcoin)')

args = parser.parse_args()
coin_id = args.coin_id
currency = 'usd'
days = '2' # set to data for past 2 days right now 
interval = 'hourly' 

print(f"Collecting data for {coin_id} from coingecko")

url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart" # API endpoint

params = {
    'vs_currency': currency,
    'days': days,
}

response = requests.get(url, params=params)
data = response.json()

# Extract prices, market_caps, and total_volumes
prices = data['prices']           # [timestamp, price]
market_caps = data['market_caps']  # [timestamp, market_cap]
volumes = data['total_volumes']    # [timestamp, volume]

df_prices = pd.DataFrame(prices, columns=['timestamp', 'price'])
df_market_caps = pd.DataFrame(market_caps, columns=['timestamp', 'marketCap'])
df_volumes = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
df = df_prices.merge(df_market_caps, on='timestamp').merge(df_volumes, on='timestamp')


df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

df.rename(columns={'price': 'close'}, inplace=True)
ohlc = df.set_index('timestamp')['close'].resample('1H').ohlc()
ohlc.columns = ['open', 'high', 'low', 'close']

# Aggregate volume and marketCap
volume = df.set_index('timestamp')['volume'].resample('1H').sum()
marketCap = df.set_index('timestamp')['marketCap'].resample('1H').mean()
result = pd.concat([ohlc, volume, marketCap], axis=1).reset_index()

# Add timeOpen, timeClose, timeHigh, timeLow, name
result['timeOpen'] = result['timestamp']
result['timeClose'] = result['timestamp'] + pd.Timedelta(hours=1)
result['timeHigh'] = result['timestamp']
result['timeLow'] = result['timestamp']
result['name'] = coin_id

result = result[[
    'timeOpen', 'timeClose', 'timeHigh', 'timeLow', 'name',
    'open', 'high', 'low', 'close', 'volume', 'marketCap', 'timestamp'
]]

today_str = datetime.now().strftime("%Y-%m-%d")
filename = f"{coin_id}_{today_str}_data_.csv"
filepath = f"data/coingecko/{filename}"

result.to_csv(filepath, index=False)
