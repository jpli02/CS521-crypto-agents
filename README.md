# CS521-crypto-agents

The final project for CS521: crypto agents

Contributions:

- Rohan Vanjani (vanjani3)
  * Worked on Realtime Data collection
- Labdhi Jain (ljain2)
  * Worked on unning the CryptoTrade agent, specifically using OpenAI GPT3.5 Turbo to analyze quantitative data. 
  * Created a news analyzer agent, which 
    * fetches news from NewsAPI and CryptoPanic
    * Uses NLTK's VADER sentiment analyzer, enhanced with crypto-specific terminology
    * Calculates sentiment scores for different cryptocurrencies
    * Produces BUY/SELL/HOLD signals based on sentiment thresholds
    * Implements cooldown periods to prevent signal spam
    * Creates summaries of the most important news for specific currencies
    * Provides context for the generated signals
- Jianping Li (jli199)
  * Worked on CryptoTrade


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
# CryptoTrade
## Running CryptoTrade
  ``` 
  ./run_agent.sh
  ```
# NewsAgent

## Setting up News Agent
Set a `.env` file with the following values in the NewsAgent diretory:

``` 
NEWSAPI_KEY=02dda15edbdf48428608819046424a83
CRYPTOPANIC_KEY=9b8c947486d79d0f1bfa5b3a8d423dc3bd498eba 
```

## Running  News Agent
``` python NewsAgent/news_analyst_agent.py ```

Example output:
<img width="1243" alt="Screenshot 2025-05-13 at 12 04 11 AM" src="https://github.com/user-attachments/assets/050ed5cd-b74b-43b4-9d17-45d942bc7fa2" />



## Reference
Build on top of :
[EMNLP 2024] CryptoTrade: A Reflective LLM-based Agent to Guide Zero-shot Cryptocurrency Trading https://aclanthology.org/2024.emnlp-main.63.pdf

