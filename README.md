# CS521-crypto-agents

The final project for CS521: crypto agents

- Rohan Vanjani (vanjani3)
- Jianping Li (jli199)

# Data Collection Instructions

To fetch current market data that is on-chain, run

```$
python3 coingecko_fetcher.py --coin_id <token_name>
```

This outputs a file `./data/coingecko/<TOKEN>-<DATE>-data.csv`

To fetch current market data that is off-chain, run

```$
python3 news_fetcher.py --coin_id <token_name>
```

This outputs a file `./data/coingecko/<TOKEN>-<DATE>_articles.json`
