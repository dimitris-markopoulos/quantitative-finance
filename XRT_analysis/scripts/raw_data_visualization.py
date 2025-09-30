import pandas as pd
import plotly.graph_objs as go

def to_html_with_ellipsis(df, n_head=5, n_tail=5, path="path.html"):
    """Export DataFrame to HTML with head, ellipsis row, and tail."""
    if len(df) <= n_head + n_tail:
        html = df.to_html(index=False)
    else:
        head = df.head(n_head)
        tail = df.tail(n_tail)

        # Placeholder row with "..." for each column
        ellipsis_row = pd.DataFrame(
            [["..."] * df.shape[1]], columns=df.columns
        )

        new_df = pd.concat([head, ellipsis_row, tail], ignore_index=True)
        html = new_df.to_html(index=False, escape=False)

    with open(path, "w") as f:
        f.write(html)

def timeseries_plot(df, path = "path.html"):
    """Export Viz to HTML"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df['Date'].values,
            y=df['XRT Price'],
            mode="lines",
            line=dict(color="red", width=1),
            name="Spot"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df['Date'].values,
            y=df['Deviation'],
            mode="lines",
            line=dict(color="black", width=1),
            name="Deviation"
        )
    )

    fig.update_layout(
        title="XRT ETF Time Series",
        height=500,
        width=1000
    )

    fig.write_html(path, include_plotlyjs="cdn")

if __name__ == '__main__':
    df = pd.read_excel('bw_data.xlsx')
    x = df.copy()
    x["Date"] = x["Date"].astype(str)

    # save table html
    to_html_with_ellipsis(x, n_head=7, n_tail=7, path="html_files/truncated_table.html")

    # save plot html
    timeseries_plot(df, path = "html_files/XRT_timeseries.html")

    print("*** Successfully Saved ***")
