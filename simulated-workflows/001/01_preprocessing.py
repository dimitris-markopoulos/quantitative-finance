import pandas as pd

#======================
# Task 0.          
# Load Data          
#======================

fundamentals_df = pd.read_csv('fundamentals.csv')
prices_df       = pd.read_csv('prices.csv')
ratings_df      = pd.read_csv('ratings.csv')

#======================
# Task 1.          
# Data Preprocessing     
#======================

mapping_dict = {
    'Q1' : '03-31',
    'Q2' : '06-30',
    'Q3' : '09-30',
    'Q4' : '12-31',
}

fundamentals_long_df = (
    fundamentals_df
    .melt(id_vars='ticker',var_name='variable',value_name='value')
    .assign(
        quarter    = lambda x: x['variable'].apply(lambda y: y.split('_')[0][-2:]),
        year       = lambda x: x['variable'].apply(lambda y: y.split('_')[0][:-2]),
        MM_DD      = lambda x: x['quarter'].apply(lambda y: mapping_dict[y]),
        period_end = lambda x: x['year'] + '-' + x['MM_DD'],
        type       = lambda x: x['variable'].apply(lambda y: y.split('_')[1])
    )
    .pivot_table(values=['value'],index=['ticker','quarter','year','period_end'],columns='type')
    .pipe(lambda x: x.set_axis([c[1] for c in list(x.columns.to_flat_index())],axis=1))
    .reset_index(drop=False)
    .astype({
        'year'      : 'int',
        'quarter'   : 'string',
        'period_end': 'datetime64[ns]',
        'ticker'    : 'string',
        'eps'       : 'float',
        'rev'       : 'float'
    })
    .sort_values(by=['ticker','year','quarter'])
    .assign(
        period_start = lambda x: x.groupby(by='ticker')['period_end'].transform(lambda y: y.shift(1) + pd.Timedelta(days=1)),
    )
    .pipe(lambda y: y.assign(
        period_start = lambda x: x["period_start"].fillna(x["period_end"] - pd.DateOffset(months=3) + pd.Timedelta(days=1))
    )
    )
    [['ticker','year','quarter','period_start','period_end','eps','rev']]
)

month_to_quarter_map = {
    '01':'Q1','02':'Q1','03':'Q1',
    '04':'Q2','05':'Q2','06':'Q2',
    '07':'Q3','08':'Q3','09':'Q3',
    '10':'Q4','11':'Q4','12':'Q4',
}

merged_df = (
    prices_df
    .assign(
        year = lambda x: x['date'].apply(lambda y: y.split('-')[0]),
        quarter = lambda x: x['date'].apply(lambda y: month_to_quarter_map[y.split('-')[1]]),
    )
    .astype({
        'year'      : 'int',
        'quarter'   : 'string',
        'ticker'    : 'string',
    })
    .pipe(lambda x:
            x.merge(
                right = fundamentals_long_df,
                on = ['ticker','year','quarter'],
                how = 'left'
            )
    )
    [['date','ticker','quarter','year','period_start','period_end','close','volume','eps','rev']]
)

ratings_long_df = (
    ratings_df
    .pivot_table(
        values  = 'rating',
        index   = ['ticker','event_date'],
        columns = 'analyst',
        aggfunc = 'sum'
    )
    .reset_index(drop=False)
    .assign(
        year = lambda x: x['event_date'].apply(lambda y: y.split('-')[0]),
        quarter = lambda x: x['event_date'].apply(lambda y: month_to_quarter_map[y.split('-')[1]]),   
    )
    .astype({
        'year'      : 'int',
        'quarter'   : 'string',
        'ticker'    : 'string',
    })
    .rename(columns={'event_date':'date'})
    [['ticker','year','quarter','date','GS','JPM','MS']]
)

final_df = (
    merged_df
    .merge(
        right = ratings_long_df,
        on = ['ticker','year','quarter','date'],
        how = 'left'
    )
)
## ffill the analyst rating until next rating
for col in ['GS','JPM','MS']:
    final_df[col] = final_df.groupby("ticker")[[col]].transform(lambda y: y.ffill())


if __name__ == '__main__':
    pass
    # final_df.to_csv('final.csv',index=False)