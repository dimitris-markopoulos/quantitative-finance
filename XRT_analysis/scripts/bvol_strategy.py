import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from IPython.display import HTML

def create_df(
        short_win : int = 20,
        long_win : int = 60,
        holding_period : int = 20,
    ):
    """
    DEFAULT PARAMETERIZATION,

    =================== PARAMS ===================
    short_win = 20        # MA short window
    long_win  = 60        # MA long window
    holding_period = 20   # 10, 20, 30 days ...
    ==============================================
    """
    #================== LOAD DATA ==================
    df = pd.read_excel('bw_data.xlsx', sheet_name='BVol')


    #================== PREPARE DATA ==================
    df = df.sort_values('Date').copy()
    df['BVOL_s'] = df['Bvol'].rolling(short_win).mean()
    df['BVOL_l'] = df['Bvol'].rolling(long_win).mean()
    df = df.dropna(subset=['BVOL_s','BVOL_l']).reset_index(drop=True)

    #================== SIGNALS ==================
    diff = df['BVOL_s'] - df['BVOL_l']

    # Crossover detection
    cross_up   = (diff > 0) & (diff.shift(1) <= 0)   # rising-vol regime begins -> SHORT ; logic = short > long and previous day is of a different sign
    cross_down = (diff < 0) & (diff.shift(1) >= 0)   # calm-vol regime begins   -> LONG ; logic = long > short and previous day is of a different sign

    # Raw signals: -1 = short entry, +1 = long entry, 0 = no trade
    raw_signals = np.zeros(len(df), dtype=int)
    raw_signals[cross_down] = 1
    raw_signals[cross_up]   = -1

    #================== SPARSIFY ==================
    signals = np.zeros(len(df), dtype=int)

    i = 0
    while i < len(raw_signals):
        if raw_signals[i] != 0:
            # Keep this signal
            signals[i] = raw_signals[i]
            # Skip forward by the holding period to avoid overlap
            i += holding_period
        else:
            i += 1

    df['signal'] = signals
    return df


def backtest(df : pd.DataFrame, holding_period : int):
    #================== SIGNAL MATRIX CALCULATION ==================
    signal_df = df[df['signal']!=0].copy()

    df = df.set_index('Date')
    signal_matrix_dict = {}

    for date, signal in zip(signal_df['Date'], signal_df['signal']):
        start_idx = df.index.get_loc(date)
        
        # Ensure we have enough future data
        if start_idx + holding_period < len(df):
            x = df['XRT Price'].iloc[start_idx:start_idx+holding_period+1]
            entry_price = x.iloc[0]
            path = (x / entry_price - 1) * signal   # returns path adjusted by signal
            
            signal_matrix_dict[f'{date.strftime("%Y-%m-%d")}_pos{signal}'] = path.values
        else:
            print(f"Throw away: {date.strftime('%Y-%m-%d')} (not enough data)")
            
    signal_matrix_df = pd.DataFrame(signal_matrix_dict)

    #================== SUMMARY STATS ==================
    final_returns = signal_matrix_df.iloc[-1,:].copy().values
    avg_return = np.mean(final_returns)
    median_return = np.median(final_returns)
    sharpe_ratio  = np.mean(final_returns) / np.std(final_returns) * np.sqrt(len(final_returns))

    summary_stats = pd.DataFrame({
        "Average Return": [avg_return],
        "Median Return": [median_return],
        "Sharpe Ratio": [sharpe_ratio]
    })

    return signal_matrix_df, summary_stats

def apply_stoploss(signal_matrix_df : pd.DataFrame, stop_loss : float = -0.05):
    """
    STOP LOSS SHOULD BE NEGATIVE FLOAT BETWEEN (-1,0)
    """
    signal_matrix_stoploss_dict = {}
    for i in range(signal_matrix_df.shape[1]):
        col = signal_matrix_df.columns[i]
        raw_path = signal_matrix_df.iloc[:,i]
        n = raw_path.shape[0]
        path = np.array([0] + (n-1)*[np.nan])
        j = 0
        while j < raw_path.shape[0]:
            if raw_path.iloc[j] < stop_loss:
                path[j:] = stop_loss
                break
            else: # within tolerance
                path[j] = raw_path[j]
            j+= 1
        
        signal_matrix_stoploss_dict[col] = path
    return pd.DataFrame(signal_matrix_stoploss_dict)

#==========================================================================================
#==========================================================================================
#==========================================================================================
#==========================================================================================
#==========================================================================================
#==========================================================================================
#==========================================================================================
#==========================================================================================
#==========================================================================================
#================== VIZ!!!!!!!!!!!!!!!!!! =================================================


#================== VIZ: BVOL + MAs + Signals ==================

def plot_bvol_signal(df: pd.DataFrame, path: str, params: dict):
    holding_period = params['holding_period']

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3],
        subplot_titles=("", "Trade Signals")
    )

    # Top panel: BVOL + MAs
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Bvol'], mode='lines', name='BVOL', line=dict(color='black', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BVOL_s'], mode='lines', name=f"SMA {params['short_win']}", line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BVOL_l'], mode='lines', name=f"SMA {params['long_win']}", line=dict(color='red', width=2)),
        row=1, col=1
    )

    # Bottom panel: signals
    long_signals = df[df['signal'] == 1]
    short_signals = df[df['signal'] == -1]

    fig.add_trace(
        go.Scatter(
            x=long_signals.index, y=[1]*len(long_signals),
            mode="markers", marker=dict(symbol="triangle-up", size=10, color="green"),
            name="Long Entry"
        ), row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=short_signals.index, y=[-1]*len(short_signals),
            mode="markers", marker=dict(symbol="triangle-down", size=10, color="red"),
            name="Short Entry"
        ), row=2, col=1
    )

    # Layout + embed holding period
    fig.update_layout(
        height=500,
        width=4000,
        title_text=f"BVOL Strategy Signals",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        template="plotly_white"
    )

    fig.add_annotation(
        text=f"Holding Period: {holding_period} days",
        xref="paper", yref="paper",
        x=0, y=-0.25,
        showarrow=False,
        font=dict(size=12, color="gray"),
        align="left"
    )

    fig.write_html(path, include_plotlyjs="cdn")

def plot_return_distribution(signal_matrix_df, path, holding_period):
    final_returns = signal_matrix_df.iloc[-1, :].values
    mean_val   = float(np.mean(final_returns))
    median_val = float(np.median(final_returns))

    fig = px.histogram(
        x=final_returns,
        nbins=30,
        title=f"Distribution of {holding_period}-Day Trade Returns",
        labels={'x': 'Final Return', 'y': 'Count'},
        opacity=0.7,
        template="plotly_white"
    )

    fig.update_traces(showlegend=False)
    fig.add_vline(x=mean_val,   line_dash="dash", line_color="blue")
    fig.add_vline(x=median_val, line_dash="dot",  line_color="red")

    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color="blue", dash="dash"),
        name=f"Mean = {mean_val:.3f}", hoverinfo="skip",
        visible=True
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color="red", dash="dot"),
        name=f"Median = {median_val:.3f}", hoverinfo="skip",
        visible=True
    ))

    fig.update_layout(
        width=1000, height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5),
        bargap=0.1
    )

    fig.write_html(path, include_plotlyjs="cdn")



def bvol_summary_statistics(signal_matrix_df : pd.DataFrame, path : str = "html_files/bvol_summary_stats.html"):

    final_returns = signal_matrix_df.iloc[-1,:].copy().values
    avg_return = np.mean(final_returns)
    median_return = np.median(final_returns)
    sharpe_ratio  = np.mean(final_returns) / np.std(final_returns) * np.sqrt(len(final_returns))

    summary_stats = pd.DataFrame({
        "Average Return": [avg_return],
        "Median Return": [median_return],
        "Sharpe Ratio (of Trade Distribution)": [sharpe_ratio]
    })

    summary_stats = summary_stats.round(4)

    # HTML
    html_table = summary_stats.to_html(
        classes="summary-table",
        index=False,
        border=0,
        justify="center"
    )

    with open(path, "w") as f:
        # Nice html with chatgpt format
        f.write(f"""
        <html>
        <head>
        <style>
            body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 10px;
            }}
            h2 {{
            text-align: center;
            margin-bottom: 15px;
            }}
            table.summary-table {{
            border-collapse: collapse;
            margin: auto;
            font-size: 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            table.summary-table th {{
            background-color: #004c6d;
            color: white;
            padding: 8px 12px;
            text-align: center;
            }}
            table.summary-table td {{
            border: 1px solid #ddd;
            padding: 8px 12px;
            text-align: center;
            }}
            table.summary-table tr:nth-child(even) {{
            background-color: #f9f9f9;
            }}
        </style>
        </head>
        <body>
        {html_table}
        </body>
        </html>
        """)

def plot_stoploss_with_stats(
        signal_matrix_df : pd.DataFrame, 
        stoploss_matrix_df : pd.DataFrame, 
        stop_loss : float, 
        path : str
    ):

    # Compute final returns
    raw_final = signal_matrix_df.iloc[-1]
    stop_final = stoploss_matrix_df.iloc[-1]

    # Compute summary stats
    summary_stats = pd.DataFrame({
        "Scenario": ["No Stop", f"Stop @ {int(stop_loss*100)}%"],
        "Mean Return": [raw_final.mean(), stop_final.mean()],
        "Sharpe Ratio": [
            raw_final.mean() / raw_final.std() * np.sqrt(len(raw_final)),
            stop_final.mean() / stop_final.std() * np.sqrt(len(stop_final))
        ]
    }).round(3)

    # Build subplot layout
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        subplot_titles=(f"Trade Paths with Stop-Loss @ {int(stop_loss*100)}%", "Summary Stats"),
        specs=[[{"type": "xy"}, {"type": "table"}]]
    )

    cols_to_plot = signal_matrix_df.columns
    for col in cols_to_plot:
        raw_path = signal_matrix_df[col].values
        stop_path = stoploss_matrix_df[col].values

        # If stop triggered
        if np.any(stop_path == stop_loss):
            stop_idx = np.where(stop_path == stop_loss)[0][0]
            fig.add_trace(go.Scatter(
                x=np.arange(stop_idx+1),
                y=stop_path[:stop_idx+1],
                mode="lines",
                line=dict(color="red", width=1.5),
                showlegend=False,
                opacity=0.5
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=np.arange(len(raw_path)),
                y=raw_path,
                mode="lines",
                line=dict(color="grey", width=1),
                showlegend=False,
                opacity=0.5
            ), row=1, col=1)

    # Add summary stats as a table on the right
    fig.add_trace(go.Table(
        header=dict(values=list(summary_stats.columns), fill_color="lightgrey", align="center"),
        cells=dict(values=[summary_stats[c] for c in summary_stats.columns], align="center")
    ), row=1, col=2)

    # Layout
    fig.update_layout(
        height=600, width=1500,
        title_text="",
        xaxis_title="Holding Day",
        yaxis_title="Cumulative Return"
    )
    fig.write_html(path, include_plotlyjs="cdn")


###########################################################################
if __name__ == '__main__':

    params_dict = {
        'short_win'      : 20,
        'long_win'       : 60,
        'holding_period' : 20
    }
    holding_period = params_dict['holding_period']
    df = create_df(**params_dict)
    signal_matrix_df, summary_stats = (
        backtest(
            df=df,
            holding_period=holding_period
        )
    )
    
    plot_bvol_signal(
        df=df, 
        path='html_files/bvol_signal_plot.html',
        params=params_dict
    )

    plot_return_distribution(
        signal_matrix_df=signal_matrix_df,
        path='html_files/bvol_final_returns.html',
        holding_period=holding_period
    )

    bvol_summary_statistics(
        signal_matrix_df = signal_matrix_df, 
        path = "html_files/bvol_summary_stats.html"
    )

    # OPTIMIZATION 

    # 3%
    signal_matrix_stoploss_3perc_df = (
        apply_stoploss(
            signal_matrix_df=signal_matrix_df,
            stop_loss= -0.03
        )
    )

    plot_stoploss_with_stats(
        signal_matrix_df=signal_matrix_df,
        stoploss_matrix_df=signal_matrix_stoploss_3perc_df,
        stop_loss=-0.03,
        path = 'html_files/stop_loss_plots/perc3_stoploss.html'
    )

    # 5%
    signal_matrix_stoploss_5perc_df = (
        apply_stoploss(
            signal_matrix_df=signal_matrix_df,
            stop_loss= -0.05
        )
    )
    
    plot_stoploss_with_stats(
        signal_matrix_df=signal_matrix_df,
        stoploss_matrix_df=signal_matrix_stoploss_5perc_df,
        stop_loss=-0.05,
        path = 'html_files/stop_loss_plots/perc5_stoploss.html'
    )

    # 10%
    signal_matrix_stoploss_10perc_df = (
        apply_stoploss(
            signal_matrix_df=signal_matrix_df,
            stop_loss= -0.10
        )
    )
    
    plot_stoploss_with_stats(
        signal_matrix_df=signal_matrix_df,
        stoploss_matrix_df=signal_matrix_stoploss_10perc_df,
        stop_loss=-0.10,
        path = 'html_files/stop_loss_plots/perc10_stoploss.html'
    )