import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

#======================
# Task 0.          
# Load Data          
#======================

eps_wide_df      = pd.read_csv('eps_wide.csv')
monthly_panel_df = pd.read_csv('monthly_panel.csv')

#======================
# Task 1.          
# Reshape          
#======================

eps_long_df = (
    eps_wide_df
    .melt(id_vars='ticker',var_name='variable',value_name='eps')
    .assign(
        year    = lambda x: x['variable'].apply(lambda y: int(y.split('Q')[0])),         # int column
        quarter = lambda x: x['variable'].apply(lambda y: int(int(y.split('Q')[1]))),    # int column
        ticker  = lambda x: x['ticker'].astype(str)
    )
    .drop(columns='variable')
    .sort_values(by=['ticker','year','quarter'])
    [['ticker','year','quarter','eps']]
)

## Sanity check - pivot back
sanity_check_df = (
    eps_long_df
    .pivot(columns=['year','quarter'],values='eps',index='ticker')
)
sanity_check_df.columns = [f'{x[0]}Q{x[1]}' for x in sanity_check_df.columns.to_flat_index()]
assert (sanity_check_df - eps_wide_df.set_index('ticker')).sum().sum() < 1e-3, 'Warning, there is an issue with your calculations.'

#======================
# Task 2.          
# Feature Engineering          
#======================

monthly_panel_df['date'] = pd.to_datetime(monthly_panel_df['date'])

engineered_monthly_panel_df = (
    monthly_panel_df
    .groupby(by=['date','ticker','sector'],as_index=False)[['adj_close','volume_m','revenue_b','eps']].sum() # remove sector level
    .drop(columns='sector')
    .assign(
        revenue_yoy = lambda x: x.groupby(by=['ticker'],as_index=False)['revenue_b'].transform(lambda y: y.pct_change(12)),
        revenue_mom = lambda x: x.groupby(by=['ticker'])['revenue_b'].transform(lambda y: y.pct_change(1)),
        eps_yoy = lambda x: x.groupby(by=['ticker'])['eps'].transform(lambda y: y.pct_change(12)),
        eps_mom = lambda x: x.groupby(by=['ticker'])['eps'].transform(lambda y: y.pct_change(1)),
    )
    .sort_values(by=['ticker','date'])
)

#======================
# Task 3.          
# Viz        
#======================

final_df = (
    engineered_monthly_panel_df
    .assign(
        returns = lambda x: x.groupby(by='ticker')['adj_close'].transform(lambda y: y.pct_change(1)) # Monthly returns
    )
    .drop(columns=['adj_close','volume_m','revenue_b','eps'])
    .set_index('date')
)

## Viz
ticker_list = list(final_df['ticker'].unique())
fig,ax = plt.subplots(len(ticker_list), figsize = (10,5*len(ticker_list)))
for i,ticker in enumerate(ticker_list):
    for col in ['revenue_yoy','revenue_mom','eps_yoy','eps_mom','returns']:
        ax[i].plot(final_df.loc[final_df['ticker']==ticker,col], label=col)
    ax[i].set_title(f'{ticker} - viz')
    ax[i].legend()
plt.savefig('metrics_viz.png')
# plt.show()

#======================
# Task 4.          
# Regression        
#======================

def train_test_split(df : pd.DataFrame, proportion : float) -> pd.DataFrame:
    n = int(df.shape[0]*proportion) # 80% 20% train test split
    tr, ts = df.iloc[:n].copy(), df.iloc[n:].copy()
    return tr, ts

results_dict = {}
for i, ticker in enumerate(ticker_list):
    df = final_df.loc[final_df['ticker']==ticker].dropna()                      # filter specific ticker
    X, y = df.loc[:,'revenue_yoy':'eps_mom'].copy(), df.loc[:,'returns'].copy() # extract target

    ## split into training and testing set
    X_tr,X_ts = train_test_split(X,0.8)
    y_tr,y_ts = train_test_split(y,0.8)

    ## fit and evaluate
    reg = LinearRegression().fit(X_tr,y_tr)
    y_pred = reg.predict(X_ts)
    rsquared = r2_score(y_true=y_ts, y_pred=y_pred)

    results_dict[i] = [ticker,reg.intercept_] + list(reg.coef_) + [rsquared]

results_df = pd.DataFrame(results_dict, index=['ticker','intercept','beta_1','beta_2','beta_3','beta_4','r-squared']).T

#======================
# Task 5.          
# Regularization        
#======================

ticker_list = list(final_df['ticker'].unique())
fig,ax = plt.subplots(len(ticker_list), figsize = (10,5*len(ticker_list)))

ridge_results_dict = {}
for i, ticker in enumerate(ticker_list):
    df = final_df.loc[final_df['ticker']==ticker].dropna()                      # filter specific ticker
    X, y = df.loc[:,'revenue_yoy':'eps_mom'].copy(), df.loc[:,'returns'].copy() # extract target

    ## split into training and testing set
    X_tr,X_ts = train_test_split(X,0.8)
    y_tr,y_ts = train_test_split(y,0.8)

    ## fit and evaluate
    reg = Ridge().fit(X_tr,y_tr)
    y_pred = reg.predict(X_ts)
    rsquared = r2_score(y_true=y_ts, y_pred=y_pred)

    ridge_results_dict[i] = [ticker,reg.intercept_] + list(reg.coef_) + [rsquared]

    eps = y_ts - y_pred
    ax[i].scatter(y_pred, eps)
    ax[i].set_title(f'{ticker} - residual plot')

plt.savefig('residual_plot.png')
# plt.show()

ridge_results_df = pd.DataFrame(ridge_results_dict, index=['ticker','intercept','beta_1','beta_2','beta_3','beta_4','r-squared']).T

if __name__ == '__main__':
    print('===== OLS =====')
    print(results_df)
    print('===== OLS + RIDGE =====')
    print(ridge_results_df)