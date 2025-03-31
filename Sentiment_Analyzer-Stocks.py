import os
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
import yfinance as yf
import pdfplumber
import numpy_financial as npf
import numpy as np

# Load sentiment dictionaries
try:
    positive_df = pd.read_csv("LoughranMcDonald_Positive.csv")
    negative_df = pd.read_csv("LoughranMcDonald_Negative.csv")
    positive_words = positive_df["Words"].str.lower().tolist()
    negative_words = negative_df["Words"].str.lower().tolist()
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()
except KeyError as e:
    print(f"KeyError: {e}")
    exit()

# Function to calculate sentiment scores
def calculate_sentiment(article_text):
    """Calculate sentiment scores for a given text."""
    words = re.findall(r'\b\w+\b', article_text.lower())
    word_count = Counter(words)
    positive_score = sum(word_count[word] for word in positive_words if word in word_count)
    negative_score = sum(word_count[word] for word in negative_words if word in word_count)
    final_score = positive_score - negative_score
    return positive_score, negative_score, final_score

# Directory containing the articles
articles_dir = r"C:\Documents\news_articles"
if not os.path.exists(articles_dir):
    print(f"Error: Directory '{articles_dir}' does not exist.")
    exit()

sentiment_scores = []

# Process each company's articles
for company in os.listdir(articles_dir):
    company_path = os.path.join(articles_dir, company)
    if os.path.isdir(company_path):
        print(f"Processing company folder: {company_path}")
        total_positive, total_negative, total_final = 0, 0, 0
        article_count = 0
        for article in os.listdir(company_path):
            article_path = os.path.join(company_path, article)
            try:
                if article_path.lower().endswith(".pdf"):
                    with pdfplumber.open(article_path) as pdf:
                        text = "".join(page.extract_text() or "" for page in pdf.pages)
                else:
                    with open(article_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                if text:
                    positive, negative, final = calculate_sentiment(text)
                    total_positive += positive
                    total_negative += negative
                    total_final += final
                    article_count += 1
            except Exception as e:
                print(f"Error processing file {article_path}: {e}")
        if article_count > 0:
            avg_final_score = total_final / article_count
            sentiment_scores.append({
                "Company": company,
                "Positive": total_positive,
                "Negative": total_negative,
                "Final": avg_final_score
            })

# Check if any sentiment data was processed
if sentiment_scores:
    # Create a DataFrame for sentiment analysis
    sentiment_df = pd.DataFrame(sentiment_scores)
    sentiment_df.sort_values(by="Final", ascending=False, inplace=True)
    print("\nTop 10 Companies by Sentiment Score:")
    top_10_sentiment = sentiment_df.head(10)
    print(top_10_sentiment)
    sentiment_df.to_csv("sentiment_scores.csv", index=False)

    # Visualize the top 10 companies by sentiment
    plt.figure(figsize=(10, 6))
    plt.bar(top_10_sentiment["Company"], top_10_sentiment["Final"], color="skyblue")
    plt.title("Top 10 Companies by Sentiment Score")
    plt.xlabel("Company")
    plt.ylabel("Final Sentiment Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("top_10_companies_sentiment.png")
    plt.show()
else:
    print("No sentiment data was processed. Check your articles directory and files.")
    exit()

# Correct mapping of sentiment companies to stock symbols for Yahoo Finance
stock_symbol_mapping = {
    "Procter&Gamble": "PG",
    "Disney": "DIS",
    "AmericanExpress": "AXP",
    "Nvidia": "NVDA",
    "Walmart": "WMT",
    "JPMorganChase": "JPM",
    "Verizon": "VZ",
    "GoldmanSachs": "GS",
    "Microsoft": "MSFT",
    "UnitedHealth": "UNH"
}

# Fetch stock prices for all top sentiment companies using correct tickers
stock_prices = {}
for company in top_10_sentiment["Company"]:
    try:
        correct_symbol = stock_symbol_mapping.get(company)
        if correct_symbol:
            stock = yf.Ticker(correct_symbol)
            stock_info = stock.history(period="1d")
            stock_prices[company] = stock_info["Close"].iloc[0]
        else:
            stock_prices[company] = None
            print(f"Warning: No correct ticker found for {company}")
    except Exception as e:
        print(f"Error fetching stock price for {company}: {e}")
        stock_prices[company] = None

# Print fetched stock prices
print("\nFetched stock prices:")
for company, price in stock_prices.items():
    if price is not None:
        print(f"{company}: ${price:.2f}")
    else:
        print(f"{company}: Price unavailable.")

# Function to fetch monthly returns for a given stock within a date range
def get_monthly_returns(stock_symbol, start_date, end_date):
    if stock_symbol is None:  # Check if stock_symbol is None
        print(f"Warning: Skipping {stock_symbol} as no ticker was found.")
        return pd.Series()  # Return empty series to avoid error
    stock = yf.Ticker(stock_symbol)
    stock_data = stock.history(start=start_date, end=end_date)
    stock_data['Return'] = stock_data['Close'].pct_change()
    monthly_returns = stock_data.resample('M').last()['Return']
    return monthly_returns

# Date ranges for comparison
start_date = '2024-08-19'
end_date = '2024-11-18'

# Retrieve monthly returns for sentiment-based portfolio
sentiment_monthly_returns = {}
for company in top_10_sentiment["Company"]:
    stock_symbol = stock_symbol_mapping.get(company)
    if stock_symbol:
        sentiment_monthly_returns[company] = get_monthly_returns(stock_symbol, start_date, end_date)

# Retrieve monthly returns for predefined portfolio (you can replace with actual stock symbols)
predefined_stocks = ["Microsoft", "Disney", "Procter&Gamble", "Verizon", "UnitedHealth"]
predefined_monthly_returns = {}
for stock in predefined_stocks:
    stock_symbol = stock_symbol_mapping.get(stock)
    if stock_symbol:
        predefined_monthly_returns[stock] = get_monthly_returns(stock_symbol, start_date, end_date)

# Combine the sentiment and predefined returns into one DataFrame
sentiment_df = pd.DataFrame(sentiment_monthly_returns)
predefined_df = pd.DataFrame(predefined_monthly_returns)

# Calculate the mean return across each column for sentiment and predefined portfolios
combined_returns = pd.DataFrame({
    'Sentiment Portfolio': sentiment_df.mean(axis=1),
    'Predefined Portfolio': predefined_df.mean(axis=1)
})

# Now proceed with plotting and other analysis
combined_returns.plot(kind='bar', figsize=(10, 6))
plt.title('Month-by-Month Return Comparison')
plt.xlabel('Month')
plt.ylabel('Return')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("return_comparison.png")
plt.show()

# Analysis: Comparison of Sentiment Stocks vs. Predefined Stocks
print("\nAnalysis: Comparison of Sentiment Stocks vs. Predefined Stocks")
print("Top Sentiment Stocks:", top_10_sentiment["Company"].tolist())
print("Predefined Stocks:", predefined_stocks)

# Compare sentiment-ranked stocks to predefined list
overlap = list(set(top_10_sentiment["Company"]).intersection(predefined_stocks))
unique_to_sentiment = list(set(top_10_sentiment["Company"]).difference(predefined_stocks))
unique_to_predefined = list(set(predefined_stocks).difference(top_10_sentiment["Company"]))

print("\nOverlap between Top Sentiment and Predefined Stocks:", overlap)
print("Stocks Unique to Top Sentiment Analysis:", unique_to_sentiment)
print("Stocks Unique to Predefined List:", unique_to_predefined)

# Calculate IRR for the portfolios
initial_investment_per_stock = 200000  # Investment per stock
initial_investment_total = initial_investment_per_stock * 10  # Total for 10 stocks

# Function to calculate investment value and returns
def calculate_investment_value(returns, initial_investment):
    cash_flows = [-initial_investment]  # Start with the negative initial investment
    value = initial_investment
    for return_rate in returns:
        value *= (1 + return_rate)  # Apply monthly returns
        cash_flows.append(value)  # Add the updated value as the next cash flow
    return cash_flows

# Calculate monthly returns from the combined returns DataFrame for Sentiment Portfolio
sentiment_returns = combined_returns['Sentiment Portfolio'].dropna()

# Calculate cash flows for Sentiment Portfolio
sentiment_cash_flows = calculate_investment_value(sentiment_returns, initial_investment_total)

# Calculate IRR for Sentiment Portfolio
sentiment_irr = npf.irr(sentiment_cash_flows) * 100  # Convert to percentage

# Define predefined portfolio with an initial investment of $2M split across 10 stocks
predefined_returns = combined_returns['Predefined Portfolio'].dropna()
predefined_cash_flows = calculate_investment_value(predefined_returns, initial_investment_total)

# Calculate IRR for Predefined Portfolio
predefined_irr = npf.irr(predefined_cash_flows) * 100  # Convert to percentage

# Print the IRR values
print(f"\nSentiment Portfolio IRR: {sentiment_irr:.2f}%")
print(f"Predefined Portfolio IRR: {predefined_irr:.2f}%")