import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')
from cycler import cycler


##### Option Valuation Model (BSM) prior to expiration (t<T)
# https://www.simtrade.fr/blog_simtrade/black-scholes-merton-option-pricing-model/
# under following assumptions:
# - The model considers European options, which can only be exercised at their expiration date.
# - The price of the underlying asset follows a geometric Brownian motion (corresponding to log-normal distribution for the price at a given point in time).
# - The risk-free rate remains constant over time until the expiration date.
# - The volatility of the underlying asset price remains constant over time until the expiration date.
# - There are no dividend payments on the underlying asset.
# - There are no transaction costs on the underlying asset.
# - There are no arbitrage opportunities.

def d1(S, X, r, stdev, T): # Delta on Call option, (1 - norm.cdf(d1)) = Delta on Put option
    return (np.log(S / X) + (r + (stdev ** 2) / 2) * T) / (stdev * np.sqrt(T))

# d2 = d1 - stdev * np.sqrt(T)
def d2(S, X, r, stdev, T):
    return (np.log(S / X) + (r - (stdev ** 2) / 2) * T) / (stdev * np.sqrt(T))

# def BSM(S, X, r, stdev, T):
#     return (S * norm.cdf(d1(S, X, r, stdev, T))) - (X * np.exp(-r * T) * norm.cdf(d2(S, X, r, stdev, T)))
    
def BSM_Call(S, X, r, T, d1, d2):
    return (S * norm.cdf(d1)) - (X * np.exp(-r * T) * norm.cdf(d2))
    
def BSM_Put(S, X, r, T, d1, d2):
    return (X * np.exp(-r * T) * norm.cdf(-d2)) - (S * norm.cdf(-d1))
    #return (X * np.exp(-r * T) * (1 - norm.cdf(d2))) - (S * (1 - norm.cdf(d1)))

##### Option Payoff Functions at maturity (t=T)
def Long_Stock(S, buy_price, N):
    return (S - buy_price) * N, buy_price # (Payoff, Break Even Point)

def Short_Stock(S, buy_price, N):
    return -(S - buy_price) * N, buy_price # (Payoff, Break Even Point)

def Long_ZCB(S, X):
    return [X] * len(S) # X = Zero Coupon Bond Par Value

def Short_ZCB(S, X):
    return [-X] * len(S) # X = Zero Coupon Bond Par Value

def Long_Call(S, X, N, option_premium):
    return list(map(lambda s: (max(0, s-X) - option_premium) * N, S)), X+option_premium # (Payoff, Break Even Point)

def Short_Call(S, X, N, option_premium):
    return list(map(lambda s: -(max(0, s-X) - option_premium) * N, S)), X+option_premium # (Payoff, Break Even Point)

def Long_Put(S, X, N, option_premium):
    return list(map(lambda s: (max(0, X-s) - option_premium) * N, S)), X-option_premium # (Payoff, Break Even Point)

def Short_Put(S, X, N, option_premium):
    return list(map(lambda s: -(max(0, X-s) - option_premium) * N, S)), X-option_premium # (Payoff, Break Even Point)

def graph_payoffs(selected_positions, S, payoff_df):
    
    fig,ax = plt.subplots(figsize=(6,8), dpi=100)
    for position in selected_positions:

        plt.plot(payoff_df[position], label=position)
        
        # if multiple 0s in option payoff e.g. when option premium = 0
        if payoff_df[position].isin([0]).sum() > 1: 
            # if Long Call or Short Call: BEP is the LAST S that makes Option Payoff = 0
            if 'Call' in position:
                BEP = payoff_df[position][payoff_df[position] == 0].index[-1]
                print(f'{position} BEP: S = {BEP}')
                plt.scatter(x=BEP, y=0, marker='o', s=100, c='dodgerblue')

            # if Long Put or Short Put: BEP is the FIRST S that makes Option Payoff = 0
            elif 'Put' in position:
                BEP = payoff_df[position][payoff_df[position] == 0].index[0]
                print(f'{position} BEP: S = {BEP}')
                plt.scatter(x=BEP, y=0, marker='o', s=100, c='dodgerblue')
        
        elif payoff_df[position].isin([0]).sum() > 0: 
            BEP = payoff_df[position][payoff_df[position] == 0].index
            print(f'{position} BEP: S = {BEP[0]}')
            plt.scatter(x=BEP, y=0, marker='o', s=100, c='dodgerblue')

    # Graph Synthetic Position
    if 'Synthetic Position' in payoff_df.columns:
        plt.plot(payoff_df['Synthetic Position'], label='Synthetic Position', linestyle='--', linewidth=3)

        if payoff_df['Synthetic Position'].isin([0]).sum() == 2: # Two 0s in Option Payoff e.g. Straddle (Two BEPs)
            BEP_PF_lst = list(payoff_df['Synthetic Position'][payoff_df['Synthetic Position'].isin([0])].index)
            print(f'Synthetic Position BEP: S = {BEP_PF_lst}')
            for BEP_PF in BEP_PF_lst:
                plt.scatter(x=BEP_PF, y=0, marker='v', s=100, c='tab:red')        

        elif payoff_df['Synthetic Position'].isin([0]).sum() > 1: # More than one 0s in Option Payoff (Many BEPs)
            lagged_comparison = payoff_df['Synthetic Position'] == payoff_df['Synthetic Position'].shift(-1).replace(np.nan)
            BEP_PF = lagged_comparison.where(lagged_comparison == True).first_valid_index()
            print(f'Synthetic Position BEP: S = {BEP_PF}')
            plt.scatter(x=BEP_PF, y=0, marker='v', s=100, c='tab:red')

        elif payoff_df['Synthetic Position'].isin([0]).sum() > 0: # If any 0s in Option Payoff (BEP exist)
            BEP_PF = payoff_df['Synthetic Position'][payoff_df['Synthetic Position'] == 0].index
            print(f'Synthetic Position BEP: S = {BEP_PF[0]}')
            plt.scatter(x=BEP_PF, y=0, marker='v', s=100, c='tab:red')    

    # fix graph scale
    ax.set_xlim(0,max(S))
    yscale = payoff_df.max(axis=0).max()
    ax.set_ylim(-yscale,yscale)
    
    # Move bottom x-axis to centre, passing through (0,0)
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    
    ax.set_xlabel('Spot price at Expiration(t=T)', loc='right')
    ax.set_ylabel('Payoff at Expiration')

    fig.legend();




