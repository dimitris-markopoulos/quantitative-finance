import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

#=========================
# Task 2.          
# Prepare for Regression          
#=========================

df = (
    pd.read_csv('final.csv')
    .assign(
        returns = lambda y:
                        y.groupby(by='ticker')['close']
                        .transform(lambda x: np.log(x/x.shift(1)))
                        .copy()
    )
) ## Engineer target using feature "close"

aapl_df = (
    df
    .copy()
    .loc[df['ticker'] == 'AAPL']
    .drop(columns='ticker')
) ## Perform regression on one ticker - lets say AAPL

## Dont need to one-hot encode introduces co-linear feaures which creates singular matrix -> not invertible
"""
quarter_df = pd.DataFrame({quarter:[0]*aapl_df.shape[0] for quarter in (aapl_df['quarter'].unique())})
for i in range(aapl_df.shape[0]):
    quarter = aapl_df.iloc[i,1]   # column 1 is associated with quarter
    quarter_df.loc[i,quarter] = 1 # fill dummies
aapl_df = pd.concat([aapl_df,quarter_df],axis=1)
"""

buy_hold_sell_map = {'Buy' : 1, 'Hold' : 0, 'Sell' : -1, np.nan : 0} # missing should just be neutral
aapl_df = (
    aapl_df
    .assign(
        GS = lambda x: x['GS'].apply(lambda y: buy_hold_sell_map[y]),
        JPM = lambda x: x['JPM'].apply(lambda y: buy_hold_sell_map[y]),
        MS = lambda x: x['MS'].apply(lambda y: buy_hold_sell_map[y])
    )
    .drop(columns=['quarter','year','period_start','period_end','close'])
    .set_index('date')
)

#=========================
# Task 3.          
# Manual Regression          
#=========================

aapl_df = aapl_df.dropna()
y = aapl_df['returns'].copy()              ## Return nx1 array
X = aapl_df.drop(columns='returns').copy() ## Pull nxp matrix
X['intercept'] = [1]*X.shape[0]

## Drop features (buy rating) that may be colinear with intercept and cause singular matrix issues
drop_cols = X.columns[X.std(numeric_only=True) == 0]
X = X.drop(columns=drop_cols)

def train_test_split(df : pd.DataFrame, split_proportion : float) -> pd.DataFrame:
    """
    Returns training, testing respectively.
    """
    n = int(X.shape[0]*split_proportion)
    return df.iloc[:n].copy().to_numpy(), df.iloc[n:].copy().to_numpy()
    
X_tr, X_ts = train_test_split(df=X,split_proportion=0.8)
y_tr, y_ts = train_test_split(df=y,split_proportion=0.8)

ols_beta = np.linalg.inv(X_tr.T @ X_tr) @ X_tr.T @ y_tr
y_pred = X_ts @ ols_beta

n_ts = X_ts.shape[0]
test_mse = (1 / n_ts) * (y_ts - X_ts @ ols_beta).T @ (y_ts - X_ts @ ols_beta)
test_r_squared = 1 - test_mse / np.var(y_ts)

#=========================
# Task 4.          
# Verify Using Package          
#=========================

fit               = LinearRegression().fit(X_tr,y_tr)
ols_beta_sk       = fit.coef_
y_pred_sk         = fit.predict(X_ts)
test_r_squared_sk = r2_score(y_ts,y_pred_sk)


if __name__ == '__main__':
    print(
    f"""
    ==== MANUAL IMPLEMENTATION ====

    OLS Beta Coefficients : {ols_beta},
    Test MSE              : {test_mse},
    Test R-squared        : {test_r_squared}

    ===============================
    """
    )

    print(
    f"""
    ==== SKLEARN IMPLEMENTATION ====

    OLS Beta Coefficients : {ols_beta_sk},
    Test R-squared        : {test_r_squared_sk}

    ================================
    """
    )
