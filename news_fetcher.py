import feedparser
import json
from bs4 import BeautifulSoup
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description="Fetch cryptocurrency data.")            
parser.add_argument('--coin_id', type=str, default='bitcoin', help='Coin ID (e.g., bitcoin)')
args = parser.parse_args()
coin_id = args.coin_id

# RSS feed from CoinDesk
rss_url = "https://www.coindesk.com/arc/outboundfeeds/rss/"

print(f"Collecting article summaries for {coin_id} from CoinDesk")
    
def clean_html(raw_content):
    return BeautifulSoup(raw_content, "html.parser").get_text()

def generate_summary(content):
    parser = PlaintextParser.from_string(content, Tokenizer("english"))
    summarizer = LsaSummarizer()
    sentences = summarizer(parser.document, 4)
    summary = " ".join(str(sentence) for sentence in sentences)
    return summary


# Parse the RSS feed
feed = feedparser.parse(rss_url)

# key words to look for in the content of RSS (Generated using GPT)

if coin_id == "bitcoin":
    coin_keywords = [
        "bitcoin", "btc", "satoshi", "bitcoin price", "btc price",
        "bitcoin mining", "bitcoin halving", "btc halving",
        "bitcoin wallet", "bitcoin transaction", "bitcoin core"
    ]
elif coin_id == "ethereum":
    coin_keywords = [
        "ethereum", "eth", "vitalik", "eth price", "ethereum price",
        "ethereum 2.0", "eth 2.0", "staking", "ethereum staking",
        "ethereum wallet", "eth transaction", "ethereum gas", "eth gas",
        "smart contract", "solidity"
    ]
elif coin_id == "solana":
    coin_keywords = [
        "solana", "sol", "solana price", "sol price",
        "solana network", "solana outage", "solana staking",
        "solana validator", "solana nft", "solana wallet",
        "solana transaction", "solana gas", "solana ecosystem"
    ]

entries = []
for entry in feed.entries:
    content = entry.title + " " + entry.content[0].value

    if any(kw.lower() in content.lower() for kw in coin_keywords):
        cleaned_html = clean_html(entry.content[0].value)
        summary = generate_summary(cleaned_html)
        
        entries.append({
            "title": entry.title,
            "published": entry.published,
            "link": entry.link,
            "summary": summary
        })

# write to json file
today_str = datetime.now().strftime("%Y-%m-%d")
filename = f"{coin_id}_{today_str}_articles_.json"
filepath = f"data/rss/{filename}"

with open(filepath, "w") as f:
    json.dump(entries, f, indent=4)
