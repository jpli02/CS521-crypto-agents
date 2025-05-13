import requests
import pandas as pd
import time
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from collections import defaultdict

# Download NLTK resources (run once)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class NewsAnalystAgent:
    """
    Agent that analyzes financial news to generate cryptocurrency trading signals.
    """
    
    def __init__(self, config_path: str = 'config.json'):
        """
        Initialize the News Analyst Agent with API keys and parameters.
        
        Args:
            config_path: Path to configuration file containing API keys and settings
        """
        self.config = self._load_config(config_path)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Add crypto-specific terms to the sentiment analyzer's lexicon
        self._enhance_sentiment_lexicon()
        
        # Create data directory if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')
            
        # Initialize signal history
        self.signal_history = self._load_signal_history()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Set default values if not present
            if 'news_sources' not in config:
                config['news_sources'] = ['cryptopanic', 'newsapi']
            if 'sentiment_threshold' not in config:
                config['sentiment_threshold'] = 0.2
            if 'signal_cooldown_hours' not in config:
                config['signal_cooldown_hours'] = 12
            if 'target_currencies' not in config:
                config['target_currencies'] = ["BTC", "ETH", "SOL", "XRP", "ADA"]
                
            return config
        except FileNotFoundError:
            # Create a default config if not found
            default_config = {
                "api_keys": {
                    "newsapi": "YOUR_NEWSAPI_KEY",
                    "cryptopanic": "YOUR_CRYPTOPANIC_KEY"
                },
                "news_sources": ["cryptopanic", "newsapi"],
                "sentiment_threshold": 0.2,
                "signal_cooldown_hours": 12,
                "target_currencies": ["BTC", "ETH", "SOL", "XRP", "ADA"]
            }
            
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
                
            print(f"Created default config at {config_path}. Please update with your API keys.")
            return default_config
    
    def _enhance_sentiment_lexicon(self):
        """Add crypto-specific terms to the sentiment lexicon."""
        # Positive terms in crypto context
        positive_terms = {
            'hodl': 3.0, 'moon': 3.5, 'bullish': 3.0, 'adoption': 2.5,
            'institutional': 2.0, 'breakthrough': 2.5, 'surge': 2.5,
            'rally': 2.0, 'partnership': 2.0, 'upgrade': 1.8,
            'integration': 1.5, 'mainstream': 1.5, 'catalyst': 1.5
        }
        
        # Negative terms in crypto context
        negative_terms = {
            'dump': -3.0, 'bearish': -3.0, 'crash': -3.5, 'ban': -3.0,
            'hack': -3.5, 'scam': -3.5, 'bubble': -2.5, 'regulation': -1.5,
            'sec': -1.0, 'investigation': -2.0, 'fraud': -3.0, 'sell-off': -2.5,
            'plummet': -3.0, 'collapse': -3.0, 'rugpull': -4.0
        }
        
        # Update the lexicon
        for term, score in positive_terms.items():
            self.sentiment_analyzer.lexicon.update({term: score})
            
        for term, score in negative_terms.items():
            self.sentiment_analyzer.lexicon.update({term: score})
    
    def _load_signal_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load signal history from disk."""
        signal_history_path = 'data/signal_history.json'
        if os.path.exists(signal_history_path):
            try:
                with open(signal_history_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("Error loading signal history, creating a new one")
        
        # Create default structure if file doesn't exist or is invalid
        return {currency: [] for currency in self.config['target_currencies']}
    
    def _save_signal_history(self):
        """Save signal history to disk."""
        signal_history_path = 'data/signal_history.json'
        with open(signal_history_path, 'w') as f:
            json.dump(self.signal_history, f, indent=4)
    
    def fetch_news(self, days_back: int = 1) -> List[Dict[str, Any]]:
        """
        Fetch news from multiple sources.
        
        Args:
            days_back: Number of days to look back for news
            
        Returns:
            List of news articles
        """
        all_news = []
        
        for source in self.config['news_sources']:
            if source == 'newsapi':
                all_news.extend(self._fetch_from_newsapi(days_back))
            elif source == 'cryptopanic':
                all_news.extend(self._fetch_from_cryptopanic(days_back))
        
        # Sort by timestamp (newest first)
        all_news.sort(key=lambda x: x.get('published_at', ''), reverse=True)
        
        return all_news
    
    def _fetch_from_newsapi(self, days_back: int = 1) -> List[Dict[str, Any]]:
        """Fetch news from NewsAPI."""
        if not self.config['api_keys'].get('newsapi'):
            print("NewsAPI key not configured, skipping this source")
            return []
        
        api_key = self.config['api_keys']['newsapi']
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)
        
        crypto_queries = ' OR '.join([
            'cryptocurrency', 'bitcoin', 'ethereum',
            'crypto', 'blockchain'
        ] + self.config['target_currencies'])
        
        url = 'https://newsapi.org/v2/everything'
        params = {
            'q': crypto_queries,
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d'),
            'language': 'en',
            'sortBy': 'publishedAt',
            'apiKey': api_key
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                # Normalize the data structure
                normalized_articles = []
                for article in articles:
                    normalized_articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', {}).get('name', 'NewsAPI'),
                        'published_at': article.get('publishedAt', '')
                    })
                
                return normalized_articles
            else:
                print(f"Error fetching from NewsAPI: {response.status_code}")
                return []
        except Exception as e:
            print(f"Exception when fetching from NewsAPI: {e}")
            return []
    
    def _fetch_from_cryptopanic(self, days_back: int = 1) -> List[Dict[str, Any]]:
        """Fetch news from CryptoPanic API."""
        if not self.config['api_keys'].get('cryptopanic'):
            print("CryptoPanic key not configured, skipping this source")
            return []
        
        api_key = self.config['api_keys']['cryptopanic']
        url = 'https://cryptopanic.com/api/v1/posts/'
        
        params = {
            'auth_token': api_key,
            'currencies': ','.join(self.config['target_currencies']),
            'filter': 'important',
            'public': 'true'
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                articles = data.get('results', [])
                
                # Normalize the data structure
                normalized_articles = []
                for article in articles:
                    # Check if the article is within the days_back timeframe
                    published_at = article.get('published_at', '')
                    if published_at:
                        publish_date = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ')
                        if (datetime.now() - publish_date).days > days_back:
                            continue
                    
                    normalized_articles.append({
                        'title': article.get('title', ''),
                        'description': '',  # CryptoPanic doesn't provide description
                        'content': '',  # CryptoPanic doesn't provide content
                        'url': article.get('url', ''),
                        'source': article.get('source', {}).get('title', 'CryptoPanic'),
                        'published_at': published_at,
                        'currencies': [c['code'] for c in article.get('currencies', [])]
                    })
                
                return normalized_articles
            else:
                print(f"Error fetching from CryptoPanic: {response.status_code}")
                return []
        except Exception as e:
            print(f"Exception when fetching from CryptoPanic: {e}")
            return []
    
    def analyze_sentiment(self, news_articles: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Analyze sentiment for each target currency from news articles.
        
        Args:
            news_articles: List of news articles to analyze
            
        Returns:
            Dictionary mapping currencies to their sentiment scores
        """
        # Initialize sentiment scores for each currency
        sentiment_scores = {currency: {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0, 'count': 0} 
                            for currency in self.config['target_currencies']}
        
        # Create regex patterns for each currency
        currency_patterns = {
            currency: re.compile(f"\\b{currency}\\b|\\b{self._get_full_name(currency)}\\b", 
                                re.IGNORECASE)
            for currency in self.config['target_currencies']
        }
        
        # Process each article
        for article in news_articles:
            title = article.get('title', '')
            description = article.get('description', '')
            content = article.get('content', '')
            
            # Combine text fields for analysis
            full_text = f"{title} {description} {content}"
            
            # Get sentiment of the entire article
            article_sentiment = self.sentiment_analyzer.polarity_scores(full_text)
            
            # Check which currencies are mentioned
            mentioned_currencies = []
            
            # If article has specific currencies tagged (from CryptoPanic)
            if 'currencies' in article:
                mentioned_currencies.extend(article['currencies'])
            
            # Also check for mentions in the text
            for currency, pattern in currency_patterns.items():
                if pattern.search(full_text):
                    mentioned_currencies.append(currency)
            
            # Remove duplicates
            mentioned_currencies = list(set(mentioned_currencies))
            
            # If no specific currencies mentioned, apply to all
            if not mentioned_currencies:
                general_crypto_terms = ['crypto', 'cryptocurrency', 'blockchain', 'digital currency']
                if any(term in full_text.lower() for term in general_crypto_terms):
                    mentioned_currencies = self.config['target_currencies']
            
            # Update sentiment scores for mentioned currencies
            for currency in mentioned_currencies:
                if currency in sentiment_scores:
                    for key in ['compound', 'pos', 'neg', 'neu']:
                        sentiment_scores[currency][key] += article_sentiment[key]
                    sentiment_scores[currency]['count'] += 1
        
        # Average the scores
        for currency, scores in sentiment_scores.items():
            if scores['count'] > 0:
                for key in ['compound', 'pos', 'neg', 'neu']:
                    scores[key] /= scores['count']
        
        return sentiment_scores
    
    def _get_full_name(self, currency_code: str) -> str:
        """Get the full name of a currency from its code."""
        currency_names = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'SOL': 'solana',
            'XRP': 'ripple',
            'ADA': 'cardano',
            'DOT': 'polkadot',
            'DOGE': 'dogecoin',
            'AVAX': 'avalanche',
            'MATIC': 'polygon',
            'LINK': 'chainlink'
        }
        return currency_names.get(currency_code, currency_code)
    
    def generate_signals(self, sentiment_scores: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """
        Generate trading signals based on sentiment analysis.
        
        Args:
            sentiment_scores: Dictionary of sentiment scores by currency
            
        Returns:
            Dictionary of trading signals by currency
        """
        signals = {}
        threshold = self.config['sentiment_threshold']
        cooldown_hours = self.config['signal_cooldown_hours']
        
        current_time = datetime.now()
        
        for currency, scores in sentiment_scores.items():
            # Skip if no news about this currency
            if scores['count'] == 0:
                signals[currency] = "HOLD (no recent news)"
                continue
            
            # Check if we're within the cooldown period of last signal
            if currency in self.signal_history and self.signal_history[currency]:
                last_signal_time = datetime.strptime(
                    self.signal_history[currency][-1]['timestamp'],
                    '%Y-%m-%d %H:%M:%S'
                )
                if (current_time - last_signal_time).total_seconds() < cooldown_hours * 3600:
                    signals[currency] = f"HOLD (signal cooldown: {cooldown_hours}h)"
                    continue
            
            # Generate signal based on compound sentiment score
            compound_score = scores['compound']
            
            if compound_score > threshold:
                signal = "BUY"
            elif compound_score < -threshold:
                signal = "SELL"
            else:
                signal = "HOLD"
            
            signals[currency] = signal
            
            # Record the signal in history
            if signal in ["BUY", "SELL"]:
                signal_record = {
                    "signal": signal,
                    "sentiment_score": compound_score,
                    "timestamp": current_time.strftime('%Y-%m-%d %H:%M:%S')
                }
                self.signal_history[currency].append(signal_record)
        
        # Save updated signal history
        self._save_signal_history()
        
        return signals
    
    def summarize_news(self, news_articles: List[Dict[str, Any]], 
                      currency: Optional[str] = None) -> str:
        """
        Create a summary of the most important news for a currency.
        
        Args:
            news_articles: List of news articles
            currency: Currency to filter for (None for all)
            
        Returns:
            Summarized news text
        """
        if not news_articles:
            return "No recent news articles found."
        
        # Filter for the specified currency if provided
        if currency:
            filtered_articles = []
            pattern = re.compile(f"\\b{currency}\\b|\\b{self._get_full_name(currency)}\\b", 
                                re.IGNORECASE)
            
            for article in news_articles:
                # Check explicitly tagged currencies
                if 'currencies' in article and currency in article['currencies']:
                    filtered_articles.append(article)
                    continue
                
                # Check text content
                full_text = f"{article.get('title', '')} {article.get('description', '')}"
                if pattern.search(full_text):
                    filtered_articles.append(article)
            
            relevant_articles = filtered_articles
        else:
            relevant_articles = news_articles
        
        # Limit to top 5 most recent articles
        relevant_articles = relevant_articles[:5]
        
        if not relevant_articles:
            return f"No recent news articles found for {currency}."
        
        # Build summary
        summary_lines = []
        
        for i, article in enumerate(relevant_articles, 1):
            title = article.get('title', 'No title')
            source = article.get('source', 'Unknown source')
            if isinstance(source, dict):
                source = source.get('name', 'Unknown source')
            
            published = article.get('published_at', '')
            if published:
                try:
                    publish_date = datetime.strptime(published, '%Y-%m-%dT%H:%M:%SZ')
                    published = publish_date.strftime('%Y-%m-%d %H:%M')
                except ValueError:
                    pass  # Keep original format if parsing fails
            
            url = article.get('url', '')
            
            line = f"{i}. {title} - {source} ({published})"
            summary_lines.append(line)
        
        return "\n".join(summary_lines)
    
    def run(self, days_back: int = 1) -> Dict[str, Any]:
        """
        Run the complete news analysis pipeline.
        
        Args:
            days_back: Number of days to look back for news
            
        Returns:
            Dictionary with analysis results and trading signals
        """
        print(f"Fetching crypto news from the last {days_back} days...")
        news_articles = self.fetch_news(days_back)
        print(f"Found {len(news_articles)} articles")
        
        print("Analyzing sentiment...")
        sentiment_scores = self.analyze_sentiment(news_articles)
        
        print("Generating trading signals...")
        signals = self.generate_signals(sentiment_scores)
        
        # Prepare results
        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'articles_analyzed': len(news_articles),
            'sentiment_scores': sentiment_scores,
            'trading_signals': signals,
            'news_summary': self.summarize_news(news_articles)
        }
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save analysis results to disk."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = f'data/analysis_results_{timestamp}.json'
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved to {results_path}")


def create_default_config():
    """Create a default configuration file if it doesn't exist."""
    config_path = 'config.json'
    if not os.path.exists(config_path):
        default_config = {
            "api_keys": {
                "newsapi": "YOUR_NEWSAPI_KEY",
                "cryptopanic": "YOUR_CRYPTOPANIC_KEY"
            },
            "news_sources": ["cryptopanic", "newsapi"],
            "sentiment_threshold": 0.2,
            "signal_cooldown_hours": 12,
            "target_currencies": ["BTC", "ETH", "SOL", "XRP", "ADA"]
        }
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        print(f"Created default config at {config_path}. Please update with your API keys.")


if __name__ == "__main__":
    # Create default config if it doesn't exist
    create_default_config()
    
    # Initialize and run the agent
    agent = NewsAnalystAgent()
    
    # Process command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Cryptocurrency News Analyst Agent")
    parser.add_argument('--days', type=int, default=1, help='Number of days to look back for news')
    parser.add_argument('--currency', type=str, help='Specific currency to analyze')
    args = parser.parse_args()
    
    # Run the agent
    results = agent.run(days_back=args.days)
    
    # Display results
    print("\n===== ANALYSIS RESULTS =====")
    print(f"Timestamp: {results['timestamp']}")
    print(f"Articles analyzed: {results['articles_analyzed']}")
    
    print("\n----- TRADING SIGNALS -----")
    for currency, signal in results['trading_signals'].items():
        print(f"{currency}: {signal}")
    
    print("\n----- SENTIMENT SCORES -----")
    for currency, scores in results['sentiment_scores'].items():
        if scores['count'] > 0:
            print(f"{currency}: Compound={scores['compound']:.3f} (based on {scores['count']} articles)")
    
    print("\n----- RECENT NEWS SUMMARY -----")
    print(results['news_summary'])
    
    # If a specific currency was requested, show detailed news for it
    if args.currency and args.currency in agent.config['target_currencies']:
        currency_summary = agent.summarize_news(agent.fetch_news(args.days), args.currency)
        print(f"\n----- {args.currency} SPECIFIC NEWS -----")
        print(currency_summary)