import pandas as pd
from plotly import graph_objs as go

def plot_scenario_analysis(df: pd.DataFrame, vary_metric: str, base_params: dict, save_path: str):
    base_val = base_params[vary_metric]
    absolute_x = base_val * (1 + df.columns.astype(float))

    fig = go.Figure()

    for metric in df.index:
        fig.add_trace(
            go.Scatter(
                x=absolute_x,
                y=df.loc[metric, :],
                mode='lines+markers',
                name=metric
            )
        )

    fig.update_layout(
        height=50 * len(df.index),
        showlegend=True,
        title=f"Scenario Analysis - {vary_metric}",
        xaxis_title=f"{vary_metric}",
        yaxis_title="Metric value"
    )
    if save_path:
        fig.write_html(save_path, include_plotlyjs="cdn")
    fig.show()
