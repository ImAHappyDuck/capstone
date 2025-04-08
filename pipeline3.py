import pandas as pd

##Adding sentiment scores to the data 

df =pd.read_csv('cleaned_optData_with_prices.csv')
df2 =pd.read_csv('cleanedFinNews.csv')
print(df.shape)
df =df[df['profit'].notna() & (df['profit'] !=0)]
print(df.columns)
df2['month_year'] =pd.to_datetime(df2['Date']).dt.to_period('M')
sentiment_by_stock_month =df2.groupby(['Stock_symbol','month_year']).agg({
    'pos_score': 'mean',
    'neg_score': 'mean'
}).reset_index()

sentiment_by_stock_month =sentiment_by_stock_month.rename(columns={
    'pos_score': 'avg_pos_score',
    'neg_score': 'avg_neg_score'
})

df['month_year'] =pd.to_datetime(df['date']).dt.to_period('M')

df =df.merge(
    sentiment_by_stock_month,
    left_on=['act_symbol','month_year'],
    right_on=['Stock_symbol','month_year'],
    how='left',
    suffixes=('','_drop')  )

df =df.drop(columns=[col for col in df.columns if col.endswith('_drop') or col in ['month_year','Stock_symbol']])
df =df.drop(['Stock_symbol_x','Stock_symbol_y'],errors='ignore',axis=1)  


# df.to_csv('NewestDataset.csv',index=False)
# print("file saved")

print(df.shape)




## Add price ratios
df['priceRatio'] = df['strike'] - df['current_stock_price'] / df['opt_price']

df['percent_to_strike'] =(df['strike'] /df['current_stock_price'])



import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
df = df[df['call_put'].isin(['Call','Put'])].copy()


def black_scholes_price(S,K,T,r,sigma,option_type='call'):
    if T <= 0 or S <= 0 or K <= 0 or sigma <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
def implied_volatility(S,K,T,r,market_price,option_type='call'):
    if T <=0 or market_price <=0 or S <=0 or K <=0:
        return np.nan
    try:
        return brentq(
            lambda sigma: black_scholes_price(S,K,T,r,sigma,option_type) - market_price,
            1e-6,5.0  # bounds for sigma (IV)
        )
    except (ValueError,RuntimeError):
        return np.nan

def add_implied_volatility(df,risk_free_rate=0.0431):
    # Convert dates
    df['date'] = pd.to_datetime({
    'year': df['date_year'],
    'month': df['date_month'],
    'day': df['date_day']
})
    df['expiration'] =pd.to_datetime(df['expiration'])
    df['time_to_exp'] =(df['expiration'] - df['date']).dt.days / 365


    # Compute IV
    def compute_iv(row):
        return implied_volatility(
            S=row['current_stock_price'],
            K=row['strike'],
            T=row['time_to_exp'],
            r=risk_free_rate,
            market_price=row['opt_price'],
            option_type=row['call_put'].lower()
        )

    import swifter
    swifter.set_defaults(progress_bar=True)

    df['iv'] = df.swifter.apply(compute_iv,axis=1)

    return df


df =add_implied_volatility(df)

df.drop(columns=['time_to_exp'],inplace=True)
df.to_csv('NewestDataset.csv',index=False)
print("file saved")
print(df.shape)