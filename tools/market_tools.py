from langchain_core.tools import tool
import yfinance as yf
from newsapi import NewsApiClient
import os

@tool
def get_stock_price(ticker: str) -> dict:
    """Get the current price and 1-day change for a stock ticker."""
    stock = yf.Ticker(ticker)
    info = stock.fast_info
    return {
        "price": round(info.last_price, 2),
        "change_pct": round(info.three_month_change * 100, 2),
    }

@tool
def get_news_headlines(ticker: str) -> list[str]:
    """Fetch the 5 most recent news headlines for a stock ticker."""
    client = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))
    articles = client.get_everything(q=ticker, language="en",
                                     sort_by="publishedAt", page_size=5)
    return [a["title"] for a in articles["articles"]]

@tool
def get_rsi(ticker: str, period: int = 14) -> float:
    """Calculate the Relative Strength Index (RSI) for a ticker."""
    import pandas as pd
    hist = yf.Ticker(ticker).history(period="1mo")["Close"]
    delta = hist.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2)