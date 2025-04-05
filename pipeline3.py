import pandas as pd
df = pd.read_csv('cleaned_optData_with_prices.csv')

## calculating implied volatility for each option  using black sholes model

import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve

def black_scholes_call(S, K, T, r, sigma):
    """Calculates the Black-Scholes call option price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_vega(S, K, T, r, sigma):
    """Calculates the Vega of a Black-Scholes option."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def implied_volatility(price, S, K, T, r, option_type='call'):
    """Calculates the implied volatility using the Newton-Raphson method."""
    sigma = 0.5  # Initial guess for volatility
    for i in range(100):
        if option_type == 'call':
            price_calc = black_scholes_call(S, K, T, r, sigma)
            vega = black_scholes_vega(S, K, T, r, sigma)
        elif option_type == 'put':
             price_calc = black_scholes_put(S, K, T, r, sigma)
             vega = black_scholes_vega(S, K, T, r, sigma)
        else:
            raise ValueError("Invalid option type. Choose 'call' or 'put'.")
        
        diff = price_calc - price
        if abs(diff) < 0.0001:
            return sigma
        sigma = sigma - diff / vega
    return np.nan  # Return NaN if it fails to converge

def black_scholes_put(S, K, T, r, sigma):
    """ Calculates the Black-Scholes put option price. """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
# Example usage:
# market_price = 10.0  # Observed market price of the option
# S = 100.0  # Current stock price
# K = 100.0  # Strike price
# T = 1.0  # Time to expiration in years
r = 0.05  # Change based on fed activities
df['iv'] = np.nan  
for index, row in df.iterrows():
    market_price = row['price']  # Market price of the option
    S = row['stock_price']  # Current stock price
    K = row['strike_price']  # Strike price
    T = row['time_to_expiration'] / 365.0  # Time to expiration in years
    if T > 0:  # Only calculate if time to expiration is positive
        iv = implied_volatility(market_price, S, K, T, r, option_type=row['call_put'].lower())
        df.at[index, 'iv'] = iv


iv = implied_volatility(market_price, S, K, T, r, option_type='call')
print("Implied Volatility:", iv)

market_price_put = 5.0
iv_put = implied_volatility(market_price_put, S, K, T, r, option_type='put')
print("Implied Volatility for put:", iv_put)

