import numpy as np
import pandas as pd
from scipy.stats import norm

def bsm_greeks(S, K, r, q, v, T, call_or_put : str) -> float:
    """
    Black-Scholes-Merton price for a European call or put option.

    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike price
    r : float
        Risk-free rate
    q : float
        Continuous dividend yield
    v : float
        Volatility
    T : float
        Time to maturity (in years)
    call_or_put : str
        'call' or 'put'

    Returns
    -------
    float : Option price
    """

    call_or_put = call_or_put.lower().strip() # robust

    if call_or_put not in ['call','put']:
        raise TypeError("call_or_put must be either 'call' or 'put'")
        
    d1 = (np.log(S / K) + (r - q + 0.5*v**2)*T) / (v * np.sqrt(T))
    d2 = d1 - v * np.sqrt(T)

    # Z ~ N(0,1)
    N = norm.cdf # CDF
    n = norm.pdf # PDF

    if call_or_put == 'call':
        price =  S * np.exp(-q*T) * N(d1) - K * np.exp(-r*T) * N(d2)
        delta = np.exp(-q*T) * N(d1)

    else: #call_or_put == 'put'
        price = K * np.exp(-r*T) * N(-d2) - S * np.exp(-q*T) * N(-d1)
        delta = np.exp(-q*T) * (N(d1) - 1)

    gamma = (np.exp(-q*T) * n(d1)) / (S * v * np.sqrt(T))
    vega = S * np.exp(-q*T) * n(d1) * np.sqrt(T)
    vanna = -np.exp(-q*T) * n(d1) * (d2 / v)
    volga = vega * (d1 * d2) / v

    return {
        'price' : price,
        'delta' : delta,
        'gamma' : gamma,
        'vega'  : vega,
        'vanna' : vanna,
        'volga' : volga
    }

def create_greeks_df(
            vary: str, 
            low_b: float, up_b: float, 
            step: float, 
            base_params: dict
            ) -> pd.DataFrame:
    
    greeks_dict = {}
    for w in np.arange(low_b, up_b, step):
        params = base_params.copy()
        params[vary] = base_params[vary] * (1 + w)
        t = bsm_greeks(**params)
        greeks_dict[w] = t
    return pd.DataFrame(greeks_dict)