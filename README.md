# CS521-crypto-agents

The final project for CS521: crypto agents

- Rohan Vanjani (vanjani3)
- Jianping Li (jli199)
- 
## Requirements
We list main requirements of this repository below. 

- openai==1.30.5
- torch==2.3.0
- torch-cluster==1.6.1+pt20cu117
- torch-geometric==2.3.0
- torch-scatter==2.1.1+pt20cu117
- torch-sparse==0.6.17+pt20cu117
- torch-spline-conv==1.2.2+pt20cu117
- torchaudio==2.0.1+cu117
- torchmetrics==0.11.4
- torchvision==0.15.1+cu117
- transformers==4.30.2



## Data Collection Instructions

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

## Running CryptoTrade
  ``` 
  ./run_agent.sh
  ```