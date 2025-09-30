import pandas as pd
import numpy as np

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline

from scipy.stats import gaussian_kde


def viz_scatterplot(df : pd.DataFrame, path : str = "html_files/forward_returns_vs_deviation.html"):
    #======= Compute forward returns =======
    df["Forward_20d_Return"] = np.log(df["XRT Price"].shift(-20) / df["XRT Price"])
    plot_df = df.dropna(subset=["Forward_20d_Return"])

    #======= Data =======
    X = plot_df["Deviation"].values.reshape(-1,1)
    y = plot_df["Forward_20d_Return"].values

    #======= OLS Regression =======
    ols = LinearRegression()
    ols.fit(X, y)

    x_range = np.linspace(X.min(), X.max(), 300).reshape(-1,1)
    y_pred_ols = ols.predict(x_range)

    #======= Spline Regression =======
    spline_model = make_pipeline(
        SplineTransformer(degree=3, n_knots=3, include_bias=False),
        LinearRegression()
    )
    spline_model.fit(X, y)
    y_pred_spline = spline_model.predict(x_range)

    #======= Plotly Scatter =======
    fig = go.Figure()

    # Scatter points
    fig.add_trace(
        go.Scatter(
            x=plot_df["Deviation"],
            y=plot_df["Forward_20d_Return"],
            mode="markers",
            marker=dict(size=3, opacity=0.5, color="black"),
            name="Forward 20d Return"
        )
    )

    # OLS line
    fig.add_trace(
        go.Scatter(
            x=x_range.flatten(),
            y=y_pred_ols,
            mode="lines",
            line=dict(color="red", width=1, dash="dash"),
            name="OLS Fit"
        )
    )

    # Spline line
    fig.add_trace(
        go.Scatter(
            x=x_range.flatten(),
            y=y_pred_spline,
            mode="lines",
            line=dict(color="blue", width=2, dash="dot"),
            name="Spline Fit"
        )
    )

    fig.update_layout(
        title="Forward 20-Day Returns vs Deviation",
        xaxis_title="Deviation",
        yaxis_title="Forward 20-Day Return",
        height=500,
        width=1000
    )

    fig.write_html(path, include_plotlyjs="cdn")

def plot_hitrate(df: pd.DataFrame, path: str = 'html_files/hitrate.html'):
    data = df['Deviation']

    # Compute forward 20d return
    df["Forward_20d_Return"] = np.log(df["XRT Price"].shift(-20) / df["XRT Price"])
    plot_df = df.dropna(subset=["Forward_20d_Return"])

    # Custom bins
    bins = [(-7 + i, -7 + 1 + i) for i in range(11)]
    bin_labels = [f"[{a}, {b})" for a, b in bins]

    pos_counts = []
    neg_counts = []

    # Count positive/negative returns per bin
    for (a, b) in bins:
        subset = plot_df[(plot_df["Deviation"] >= a) & (plot_df["Deviation"] < b)]
        pos_counts.append((subset["Forward_20d_Return"] > 0).sum())
        neg_counts.append((subset["Forward_20d_Return"] <= 0).sum())

    # Subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Hit Rate by Bin (20-day lag)", "Probability Density Function")
    )

    # Left
    fig.add_trace(
        go.Bar(
            x=bin_labels,
            y=pos_counts,
            name="Positive Returns",
            marker_color="green",
            opacity=0.5
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            x=bin_labels,
            y=[-c for c in neg_counts],
            name="Negative Returns",
            marker_color="red",
            opacity=0.5
        ),
        row=1, col=1
    )

    # Right
    fig.add_trace(
        go.Histogram(
            x=data,
            nbinsx=100,
            histnorm="probability density",
            marker=dict(color="steelblue"),
            opacity=0.5,
            name="Histogram"
        ),
        row=1, col=2
    )

    kde = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 500)
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=kde(x_range),
            mode="lines",
            line=dict(color="red", width=2),
            name="KDE"
        ),
        row=1, col=2
    )

    # Layout
    fig.update_layout(
        title="Deviation Distribution with Hit Rate",
        height=500,
        width=1500,
        bargap=0.05,
        barmode="relative",  # allows stacking up/down
        showlegend=True
    )

    # Axes
    fig.update_xaxes(title="Deviation", row=1, col=1)
    fig.update_yaxes(title="Count (pos above, neg below)", row=1, col=1)

    fig.update_xaxes(title="Deviation", row=1, col=2)
    fig.update_yaxes(title="Density", row=1, col=2)

    fig.write_html(path, include_plotlyjs="cdn")

if __name__ == '__main__':
    df = pd.read_excel('bw_data.xlsx')
    viz_scatterplot(df, path='html_files/forward_returns_vs_deviation.html')
    plot_hitrate(df, path='html_files/hitrate.html')
