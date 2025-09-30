import pandas as pd
import numpy as np

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_signal_window(
        df : pd.DataFrame,
        hit_df : pd.DataFrame,
        path : str = 'html_files/signal_window.html'
    ):

    fig = go.Figure()

    for i in range(len(hit_df)):
        hit_date = hit_df.iloc[i]['Date']
        entry_price = df.loc[df['Date'] == hit_date, 'XRT Price'].iloc[0]

        # Slice +- 60 days
        sliced_df = df.loc[
            (df['Date'] >= hit_date - pd.Timedelta(days=60)) &
            (df['Date'] <= hit_date + pd.Timedelta(days=60))
        ][['Date','XRT Price']].copy()

        # Compute forward cumulative return relative to entry price
        sliced_df['ForwardReturn'] = sliced_df['XRT Price'] / entry_price - 1

        # Add trace for this signal
        fig.add_trace(go.Scatter(
            x=sliced_df['Date'],
            y=sliced_df['ForwardReturn'],
            mode='lines',
            name=str(hit_date.date()),
            opacity=0.6
        ))

        # Add vertical line at hit date
        fig.add_vline(
            x=hit_date,
            line=dict(color='red', dash='dash'),
            opacity=0.5
        )

    fig.update_layout(
        title="±60 Day Paths Around Signals<br><sup>Scaled to 0% at signal entry</sup>",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        height=600,
        width=1200,
        xaxis=dict(rangeslider=dict(visible=True))  # makes it scrollable
    )

    fig.write_html(path, include_plotlyjs="cdn")

def signal_matrix(
    df : pd.DataFrame,
    hit_df : pd.DataFrame
    ):
    """
    Create a matrix of stock path vectors (returns) after (and including) the signal
    - column is date of signal
    """

    matrix_dict = {}
    for i in range(len(hit_df)):
        signal_date = hit_df['Date'].iloc[i]
        price_vector = df.loc[(df['Date'] >= signal_date)].iloc[:61]['XRT Price'].values
        matrix_dict[signal_date.strftime('%Y-%m-%d')] = price_vector
    matrix_df = pd.DataFrame(matrix_dict)
    for signal_date in matrix_df.columns:
        price_vector = matrix_df[signal_date]
        entry_spot = price_vector[0]
        matrix_df[signal_date] = matrix_df[signal_date].apply(lambda x: (x - entry_spot) / entry_spot)
    matrix_df.index.name = "offset_days"

    return matrix_df

def plot_dist_endpoints(matrix_df : pd.DataFrame, path : str = 'html_files/endpoint_distribution.html'):
    endpoints = matrix_df.iloc[-1, :] # last row = day 60
    endpoints.name = 'fwd_60day_returns'

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Distribution (Boxplot)", "Histogram")
    )

    # Boxplot
    fig.add_trace(
        go.Box(y=endpoints, boxpoints="all", name="Returns"),
        row=1, col=1
    )

    # Histogram
    fig.add_trace(
        go.Histogram(x=endpoints, nbinsx=10, name="Returns", marker_color="steelblue",opacity=0.6),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(
        title="Distribution of +60 Day Returns Across Signals",
        showlegend=False,
        height=500, width=1300
    )

    fig.update_xaxes(title="Cumulative Return at +60 Days", row=1, col=2)
    fig.update_yaxes(title="Cumulative Return at +60 Days", row=1, col=1)
    fig.update_yaxes(title="Frequency", row=1, col=2)

    fig.write_html(path, include_plotlyjs="cdn")

def plot_return_paths_post_signal(matrix_df : pd.DataFrame, path : str = 'html_files/return_path_after_signal.html'):
    fig = go.Figure()

    for col in matrix_df.columns:
        fig.add_trace(go.Scatter(
            x=matrix_df.index,        # offset_days
            y=matrix_df[col],         # cumulative returns
            mode='lines',
            name=str(col)             # label = signal date
        ))

    fig.update_layout(
        title="Forward Return Paths by Signal<br><sup>Scaled to 0% at signal entry</sup>",
        xaxis_title="Offset Days (relative to signal)",
        yaxis_title="Cumulative Return",
        height=550, width=1000
    )

    fig.write_html(path, include_plotlyjs="cdn")

def sharpe_matrix_calculations(
    df : pd.DataFrame,
    hit_df : pd.DataFrame
    ):
    """
    Create a matrix of stock path vectors (returns) after (and including) the signal
    - column is date of signal
    """

    # Price matrix per signal
    matrix_dict = {}
    for i in range(len(hit_df)):
        signal_date = hit_df['Date'].iloc[i]
        price_vector = df.loc[(df['Date'] >= signal_date)].iloc[:61]['XRT Price'].values
        matrix_dict[signal_date.strftime('%Y-%m-%d')] = price_vector
    raw_matrix_df = pd.DataFrame(matrix_dict)
    raw_matrix_df.index.name = "offset_days"

    # Daily return matrix per signal
    daily_return_df = np.log(raw_matrix_df.shift(1) / raw_matrix_df).dropna()

    # Sharpe value per signal (daily and annualized)
    sharpes = daily_return_df.mean() / daily_return_df.std()
    avg_sharpe = sharpes.mean()

    # For the 60-day hold, the daily and annual sharpe per signal are computed below.
    sharpe_df = (
        sharpes
        .to_frame()
        .rename(columns={0:'daily_sharpe'})
        .assign(
            annualized_sharpe = lambda x: x['daily_sharpe'] * np.sqrt(252)
        )
    )

    return sharpe_df

def sharpe_plot(sharpe_df : pd.DataFrame, path : str):

    sharpes_annual=sharpe_df['annualized_sharpe']
    avg_sharpe_annual = sharpes_annual.mean()

    fig = go.Figure()

    # Bar chart of Sharpe ratios
    fig.add_trace(go.Bar(
        x=sharpes_annual.index.astype(str),
        y=sharpes_annual.values,
        name="Sharpe per signal"
    ))

    # Average line
    fig.add_hline(
        y=avg_sharpe_annual,
        line_dash="dash",
        line_color="red"
    )

    # Legend entry
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="lines",
        line=dict(color="red", dash="dash"),
        name=f"Average Sharpe = {avg_sharpe_annual:.2f}"
    ))

    fig.update_layout(
        title="Sharpe Ratios of Each 60-Day Trade",
        xaxis_title="Signal Date",
        yaxis_title="Annualized Sharpe",
        height=500, width=1000
    )

    fig.write_html(path, include_plotlyjs="cdn")


if __name__ == '__main__':
    df = pd.read_excel('bw_data.xlsx')
    df["fwd_60d_return"] = np.log(df["XRT Price"].shift(-60) / df["XRT Price"])
    raw_signals = (df['Deviation'] >= 2.0).astype(int)
    signal = pd.Series(0, index=df.index)

    i = 0
    while i < len(df):
        if raw_signals.iloc[i] == 1:
            signal.iloc[i] = 1
            i += 30   # skip next 30 days
        else:
            i += 1

    df["signal"] = signal
    hit_df = df[df['signal'] == 1].copy()

    # PART 1
    plot_signal_window(
        df=df,
        hit_df=hit_df,
        path='html_files/signal_window.html'
    )

    # PART 2
    matrix_df = signal_matrix(df=df,hit_df=hit_df)
    plot_dist_endpoints(matrix_df=matrix_df, path='html_files/endpoint_distribution.html')
    plot_return_paths_post_signal(matrix_df=matrix_df, path='html_files/return_path_after_signal.html')

    sharpe_df = sharpe_matrix_calculations(df=df,hit_df=hit_df)
    sharpe_df.to_html('html_files/sharpe_df.html')
    
    sharpe_plot(sharpe_df=sharpe_df, path='html_files/sharpe_plot.html')
