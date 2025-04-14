import datetime
import random
import csv

# Set a seed for reproducibility (optional)
random.seed(42)

# Define a list of stock symbols to simulate
symbols = ["AAPL", "GOOG", "AMZN", "TSLA", "MSFT"]

# Start date for our simulated trade dates
start_date = datetime.date(2023, 1, 1)

# Container for our dataset rows
rows = []

# Generate 250 rows
for i in range(25000):
    # Compute the trade date (each row gets a different day)
    trade_date = start_date + datetime.timedelta(days=i)
    date_str = trade_date.isoformat()
    year = trade_date.year
    month = trade_date.month
    day = trade_date.day

    # Choose a stock symbol at random
    act_symbol = random.choice(symbols)
    
    # Define an expiration date sometime between 10 and 90 days after the trade date
    exp_days = random.randint(10, 90)
    expiration_date = trade_date + datetime.timedelta(days=exp_days)
    exp_str = expiration_date.isoformat()
    
    # Simulate a current stock price between 50 and 150
    current_stock_price = round(random.uniform(50, 150), 2)
    # Strike is roughly within ±10% of the current stock price
    strike = round(current_stock_price * random.uniform(0.9, 1.1), 2)
    
    # Randomly choose option tyccpe
    call_put = random.choice(["Call", "Put"])
    
    # Set bid and ask prices (bid between 0.5 and 10, ask a bit higher)
    bid = round(random.uniform(0.5, 10), 2)
    ask = round(bid + random.uniform(0.1, 1.0), 2)
    
    # Simulate trade volume between 10 and 1000 contracts
    vol = random.randint(10, 1000)
    
    # Simulate delta: calls have positive delta; puts negative
    if call_put == "call":
        delta = round(random.uniform(0.3, 0.9), 2)
    else:
        delta = round(random.uniform(-0.9, -0.3), 2)
    
    # Other option Greeks (gamma, theta, vega, rho)
    gamma = round(random.uniform(0.01, 0.2), 2)
    theta = round(-random.uniform(0.001, 0.1), 3)  # typically negative
    vega = round(random.uniform(0.05, 0.5), 2)
    rho = round(random.uniform(-0.2, 0.2), 2)
    
    # Moneyness defined here as (current_stock_price - strike) / strike
    moneyness = round((current_stock_price - strike) / strike, 3)
    
    # Position size: simulate a trade size (could be long or short)
    position = random.choice([-10, -5, 1, 5, 10])
    
    # Option price is taken as the midpoint between bid and ask
    opt_price = round((bid + ask) / 2, 2)
    
    # Use index-based categorization for profit:
    # – Rows 0–24 (10%): super profitable trades,
    # – Rows 25–99 (30%): modestly profitable,
    # – Rows 100–249 (60%): negative (loss-making) trades.
    if i < 25:
        profit = round(opt_price * random.uniform(1.5, 3.0), 2)   # super profitable
    elif i < 100:
        profit = round(opt_price * random.uniform(0.05, 0.3), 2)   # modest profit
    else:
        profit = round(-opt_price * random.uniform(0.1, 1.0), 2)    # loss
    
    # Simulate a future stock price at expiration (introducing volatility)
    stock_price_at_expiration = round(current_stock_price * random.uniform(0.8, 1.5), 2)
    
    # The stock's 60‑day delta (price change percentage) as an independent metric
    stock_delta_60days = round(random.uniform(-0.5, 0.5), 2)
    
    # Average sentiment scores from two sources (0–1 scale)
    avg_pos_score = round(random.uniform(0, 1), 2)
    avg_neg_score = round(random.uniform(0, 1), 2)
    
    # Price Ratio is the ratio of current_stock_price to strike
    priceRatio = round(current_stock_price / strike, 3) if strike != 0 else 0
    # Percent difference between the current stock price and strike (as a percentage)
    percent_to_strike = round(abs(current_stock_price - strike) / strike * 100, 2) if strike != 0 else 0
    
    # Implied volatility (IV) as a fraction (from 20% to 100%)
    iv = round(random.uniform(0.2, 1.0), 2)
    
    # Append the row using the column ordering specified
    rows.append([
        date_str,
        act_symbol,
        exp_str,
        strike,
        call_put,
        bid,
        ask,
        vol,
        delta,
        gamma,
        theta,
        vega,
        rho,
        year,
        month,
        day,
        current_stock_price,
        stock_price_at_expiration,
        moneyness,
        position,
        opt_price,
        profit,
        stock_delta_60days,
        avg_pos_score,
        avg_neg_score,
        priceRatio,
        percent_to_strike,
        iv
    ])

# Specify the header (column names)
header = [
    "date", "act_symbol", "expiration", "strike", "call_put", "bid", "ask", "vol",
    "delta", "gamma", "theta", "vega", "rho", "date_year", "date_month", "date_day",
    "current_stock_price", "stock_price_at_expiration", "moneyness", "position",
    "opt_price", "profit", "stock_delta_60days", "avg_pos_score", "avg_neg_score",
    "priceRatio", "percent_to_strike", "iv"
]

# Write the dataset to a CSV file
with open("simulatedData.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(rows)

print("Dataset generated and saved as simulated_options_data.csv")
