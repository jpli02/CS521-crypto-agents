import os
import json
import pandas as pd
from datetime import datetime, timedelta

STARTING_NET_WORTH = 10000
STARTING_CASH_RATIO = 0.5
GAS_FEE = 0.001
EX_RATE = 0.001
SMA_PERIODS = [5, 10, 15, 20]

def get_paths(args):
    price_path = f'data/sol/price.csv'
    txn_path = f'data/sol/txn_stat.csv'
    news_dir = f'data/sol/news'
    timecol = 'date'
    price_timefmt = '%Y-%m-%d'
    txn_timefmt = '%Y-%m-%d'
    return price_path, txn_path, news_dir, timecol, price_timefmt, txn_timefmt

class SOLTradingEnv:
    def __init__(self, args):
        price_path, txn_path, self.news_dir, self.timecol, self.price_timefmt, txn_timefmt = get_paths(args)
        starting_date, ending_date = args.starting_date, args.ending_date
        df = pd.read_csv(price_path)
        df = df.sort_values(self.timecol)
        df['date'] = pd.to_datetime(df[self.timecol], format=self.price_timefmt)
        
        # SMA
        for period in SMA_PERIODS:
            df[f'SMA_{period}'] = df['open'].rolling(window=period).mean()
            df[f'STD_{period}'] = df['open'].rolling(window=period).std()
        # MACD and Signal Line
        df['EMA_12'] = df['open'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['open'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        self.data = df[(df['date'] >= starting_date) & (df['date'] <= ending_date)]
        
        self.txn_stat = pd.read_csv(txn_path).sort_values('day')
        self.txn_stat['date'] = pd.to_datetime(self.txn_stat['day'], format=txn_timefmt)
        self.total_steps = len(self.data)
        self.starting_net_worth = STARTING_NET_WORTH
        self.starting_cash_ratio = STARTING_CASH_RATIO

    def get_close_state(self, today, next_day, first_day=False):
        next_open_price = next_day['open']
        close_net_worth = self.cash + self.sol_held * next_open_price
        close_roi = close_net_worth / self.starting_net_worth - 1
        today_roi = close_net_worth / self.last_net_worth - 1
        self.last_net_worth = close_net_worth

        date = today[self.timecol]
        parsed_time = datetime.strptime(date, self.price_timefmt)
        if first_day:
            parsed_time = parsed_time - timedelta(days=1)
        year, month, day = parsed_time.year, parsed_time.month, parsed_time.day

        # next day's opening technical indicators
        ma5 = next_day['SMA_5']
        ma10 = next_day['SMA_10']
        ma15 = next_day['SMA_15']
        ma20 = next_day['SMA_20']
        slma_signal = 'hold'
        short_ma = ma15
        long_ma = ma20
        if short_ma > long_ma:
            slma_signal = 'sell'
        elif short_ma < long_ma:
            slma_signal = 'buy'
        
        sma = next_day[f'SMA_20']
        sd = next_day[f'STD_20']
        multiplier = 2
        upper_band = sma + (sd * multiplier)
        lower_band = sma - (sd * multiplier)
        boll_signal = 'hold'
        if next_open_price < lower_band:
            boll_signal = 'buy'
        elif next_open_price > upper_band:
            boll_signal = 'sell'

        macd = next_day['MACD']
        macd_signal_line = next_day['Signal_Line']
        macd_signal = 'hold'
        if macd < macd_signal_line:
            macd_signal = 'buy'
        elif macd > macd_signal_line:
            macd_signal = 'sell'

        # today's txn stats
        txn_stat = self.txn_stat[self.txn_stat['date'] == parsed_time]
        txn_cols = set(self.txn_stat.columns.tolist()) - set(['date', 'day'])
        if txn_stat.empty:
            txn_data = {col: 'N/A' for col in txn_cols}
        else:
            txn_data = {col: txn_stat[col].values[0] for col in txn_cols}

        # today's news
        news_path = f"{self.news_dir}/{year}-{str(month).zfill(2)}-{str(day).zfill(2)}.json"
        if not os.path.exists(news_path):
            news = 'N/A'
        else:
            loaded_news = json.load(open(news_path))
            seen_titles = set()
            news = []
            for loaded_item in loaded_news:
                if loaded_item['title'] not in seen_titles:
                    item = {k: loaded_item[k] for k in ['id', 'time', 'title', 'content']}
                    K = 5000
                    if len(item['content']) > K:
                        item['content'] = item['content'][:K] + '...' 
                    news.append(item)
                    seen_titles.add(item['title'])

        close_state = {
            'cash': self.cash,
            'sol_held': self.sol_held,
            'open': next_open_price,
            'net_worth': close_net_worth,
            'roi': close_roi,
            'today_roi': today_roi,
            'technical': {
                'macd_signal': macd_signal,
            },
            'txnstat': txn_data,
            'news': news,
            'date': date,
        }
        return close_state

    def reset(self):
        self.current_step = 0
        next_day = today = self.data.iloc[self.current_step]
        self.starting_price = today['open']
        self.cash = self.starting_net_worth * STARTING_CASH_RATIO
        self.sol_held = (self.starting_net_worth - self.cash) / self.starting_price
        self.last_net_worth = self.starting_net_worth
        close_state = self.get_close_state(today, next_day, first_day=True)
        info = {
            'starting_cash': self.cash,
        }
        reward = 0
        self.done = False
        self.last_state = close_state
        return close_state, reward, self.done, info

    def step(self, action):
        raw_action = action
        if type(action) == str:
            actions = re.findall(r"-?(?:0(?:\.\d{1})|1\.0)", action)
            
            if len(actions) == 0:
                print(f'ERROR: Invalid llm response: {action}. Set to no action.')
                action = 0.00
            elif len(actions) == 1:
                action = float(actions[0])
            else:
                action = float(actions[-1])
        
        if not -1 <= action <= 1:
            print(f"ERROR: Invalid action: {action}. Set to no action.")
            action = 0.00

        today = self.data.iloc[self.current_step]
        next_day = self.data.iloc[self.current_step + 1]
        open_price = today['open']
        next_open_price = next_day['open']
        
        if -1 <= action < 0 and self.sol_held > 0:
            sol_diff = abs(action) * self.sol_held
            cash_diff = sol_diff * open_price
            self.sol_held -= sol_diff
            self.cash += cash_diff
            self.cash -= GAS_FEE * open_price + cash_diff * EX_RATE
        if 0 < action <= 1 and self.cash > 0:
            cash_diff = abs(action) * self.cash
            sol_diff = cash_diff / open_price
            self.cash -= cash_diff
            self.sol_held += sol_diff
            self.cash -= GAS_FEE * open_price + cash_diff * EX_RATE
        
        self.current_step += 1
        if self.current_step >= self.total_steps - 1:
            self.done = True

        close_state = self.get_close_state(today, next_day)
        reward = close_state['roi'] - self.last_state['roi']
        self.last_state = close_state
        info = {
            'raw_action': raw_action,
            'actual_action': action,
            'starting_cash': self.starting_net_worth,
            'ref_all_in': self.starting_net_worth / self.starting_price * next_open_price,
            'today': today[self.timecol],
        }
        return close_state, reward, self.done, info
    
    def close(self):
        pass 