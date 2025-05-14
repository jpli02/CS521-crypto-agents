# CS521-crypto-agents

The final project for CS521: crypto agents

Contributions:

- Rohan Vanjani (vanjani3)

  - Worked on Realtime Data collection
  - Retreiving and Parsing On-Chain data:
    - For this project, we utilize the [Coingecko API](https://support.coingecko.com/hc/en-us/categories/4538696187673-CoinGecko-API)
    - We utilize the free version which gives pricing data for a specific token by the hour within a 2-day interval
    - Once retrieved this data is then parsed using pandas and then saved to a CSV file in the `data/coingecko/` directory
  - Retreiving and Parsing Off-Chain data:
    - For getting recent new data, we utilize rss feeds, particularly the one for Coin Desk
    - Once the data is retreived, it contains raw HTML content, so that is stripped using bs4
    - For each token (BTC/ETH/SOL), we maintain a list of keywords that generated via an LLM
    - We first check if this article contains any of the relevant key words and to narrow down the final list
    - Then we utilize sumy's PlaintextParser, Tokenizer, and LsaSummarizer to summarize the article into 4 sentences
    - Once the list is generated, this data is then written to a json file in the `data/rss/` directory

- Labdhi Jain (ljain2)
  - Worked on running the CryptoTrade agent, specifically using OpenAI GPT3.5 Turbo to analyze quantitative data.
  - Created a news analyzer agent, which
    - fetches news from NewsAPI and CryptoPanic
    - Uses NLTK's VADER sentiment analyzer, enhanced with crypto-specific terminology
    - Calculates sentiment scores for different cryptocurrencies
    - Produces BUY/SELL/HOLD signals based on sentiment thresholds
    - Implements cooldown periods to prevent signal spam
    - Creates summaries of the most important news for specific currencies
    - Provides context for the generated signals
- Jianping Li (jli199)
  - Worked on CryptoTrade agent framework, using Openai gpt-4o-mini.
  - Developed and tested on different agents for portfolio management:
    - Implement pytorch code and evaluation metrics
  - Developed and tested on different market:
    - different crypto concurrency: BTC, ETH, SOL
    - different market trends: bear, bull, neutral
    - time window: 3 days to 3 months
  - Managed team meetings and discussions
- Haodong Song (hs38)
  - Contributed to initial brainstorming and high-level design discussions; tested on CryptoTrade functions

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
NEWSAPI_KEY=<your_news_api_key>
CRYPTOPANIC_KEY=<your_crypto_panic_key>
```

## Running News Agent

`python NewsAgent/news_analyst_agent.py`

Example output:
<img width="1243" alt="Screenshot 2025-05-13 at 12 04 11 AM" src="https://github.com/user-attachments/assets/050ed5cd-b74b-43b4-9d17-45d942bc7fa2" />

## Reference

Build on top of :
[EMNLP 2024] CryptoTrade: A Reflective LLM-based Agent to Guide Zero-shot Cryptocurrency Trading https://aclanthology.org/2024.emnlp-main.63.pdf
