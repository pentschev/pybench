import altair as alt
from altair.utils import Undefined


def grouped_bar_chart(
    df,
    x_column,
    y_column,
    group_column,
    domain,
    y_title=Undefined,
    y_scale_type="symlog",
    y_tick_count=Undefined,
    group_title=Undefined,
    bar_title_angle=0,
    legend_title=Undefined,
    group_height=500,
    group_width=80,
):
    """Creates a grouped bar chart with altair.

    Parameters
    ----------
    df: pandas.DataFrame
        A pandas DataFrame containing benchmarking data.
    x_column: str
        Name of the column to be used as x-axis.
    y_column: str
        Name of the column to be used as y-axis.
    group_column: str
        Name of the column to be used for each group.
    domain: list of str
        The domain to be used for each bar group, its content will be used as
        legend for each bar. This will usually be a list of strings.
    y_title: str
        The title for the y-axis. If not specified, will use the content of
        ``y_column``.
    y_scale_type: str
        Scale type for the y-axis.
    y_tick_count: int
        Number of ticks to be plotted for the y-axis.
    group_title: str
        The title for the bar groups. If not specified, will use the content of
        ``group_column``.
    bar_title_angle: int, float
        The angle to be used for the title of each bar group.
    legend_title: str
        The title for the legend box. If not specified, will use the content of
        ``x_column``.
    group_height: int
        The height in pixels of each bar group.
    group_width
        The width in pixels of each bar group.

    Returns
    -------
    altair.FacetChart
        The grouped bar chart that can be visualized in a Jupyter Notebook or
        be saved to a file.

    Example
    -------
    Assume we have a pandas.DataFrame ``df``, for which we have columns
    ``'size'`` and ``'speedup'`` containing, respectively, the values for
    ``x_column`` and ``y_column``, and a column ``'operation'`` to designate
    ``group_column`` for each bar group we want to plot in the chart. Assume
    also that the value of ``x_column`` identifies the number of rows for each
    bar in every bar group with sizes '1M', '2M', '4M'. Finally, we want to
    relabel the legend to 'Rows', the title of the y-axis to
    'GPU Speedup Over CPU', the group title to 'Operation', and have each group
    having height of 500 pixels and width of 80 pixels, this is how this
    function would be called:
    >>> grouped_bar_chart(
    >>>     df,
    >>>     'size',
    >>>     'speedup',
    >>>     'operation',
    >>>     ['1M', '2M', '4M'],
    >>>     legend_title='Rows',
    >>>     y_title='GPU Speedup Over CPU',
    >>>     group_title='Operation',
    >>>     group_height=500,
    >>>     group_width=80)
    """

    bars = (
        alt.Chart(df)
        .mark_bar(opacity=1.0)
        .encode(
            x=alt.X(x_column),
            y=alt.Y(
                y_column,
                scale=alt.Scale(type=y_scale_type),
                axis=alt.Axis(title=y_title, tickCount=y_tick_count),
                stack=None,
            ),
            color=alt.Color(
                x_column, title=legend_title, scale=alt.Scale(domain=domain)
            ),
        )
        .properties(height=group_height, width=group_width)
    )

    # FIXME: This is a workaround to ensure positive-/negative-valued bars show
    # their marks above/below the bar. The text will be plotted twice, once
    # inside and once outside the bar, but the plotting inside will have the
    # same color and won't be visible.
    text = bars.mark_text(dy=-5).encode(text=y_column)
    text_neg = bars.mark_text(dy=7).encode(text=y_column)
    text = text + text_neg

    chart = (
        alt.layer(bars, text, data=df)
        .facet(
            column=alt.Column(
                group_column,
                title=group_title,
                sort=alt.EncodingSortField(
                    field=y_column, op="sum", order="descending"
                ),
            )
        )
        .configure_header(labelAngle=bar_title_angle)
        .configure_axisX(
            # Omit X-axis label in favor of Legend
            tickSize=0,
            labelFontSize=0,
            titleFontSize=0,
        )
    )

    return chart
