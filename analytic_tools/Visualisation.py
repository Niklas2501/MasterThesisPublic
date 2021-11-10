import os
import sys
from copy import copy

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from configuration.Configuration import Configuration
from analytic_tools.DataHandler import DataHandler
from configuration.Enums import DatasetPart as DPE

########################################################
dark_theme = False
default_range = range(30)
#######################################################

if dark_theme:
    dash_theme = dbc.themes.DARKLY
    plotly_theme = "plotly_dark"  # https://plotly.com/python/templates/

else:
    dash_theme = dbc.themes.BOOTSTRAP
    plotly_theme = "plotly_white"

menu_item_style = {'width': '200px', "margin-right": "25px", "margin-top": "0px",
                   }  # "height": "90%" "display": "inline-block", "float": "left",
div_style = {"width": "100%", "display": "flex", "margin": "25px 0px 25px 0px", "font-size": "large",
             "align-items": "center"}
text_style = {"margin-right": "25px", "display": "inline", "margin-top": "5px"}

last_fig = None
last_fig_path = None

config = Configuration()
data_handler = DataHandler(config, default_range)

app = dash.Dash(external_stylesheets=[dash_theme])

app.layout = dbc.Container(
    [
        html.P(),
        html.Hr(),
        html.H1("FT Dataset Example Visualiser"),
        html.Hr(),
        html.Div(
            [
                dbc.Select(
                    id="part_selection",
                    options=[{'label': i, 'value': i} for i in [DPE.TRAIN.value, DPE.TEST.value]],
                    value=DPE.TRAIN,
                    style=menu_item_style,
                ),

                dbc.Input(id="attribute_range", type="text", style=menu_item_style,
                          value=f"0:{len(default_range)}"),

                dbc.Button(
                    "Update view",
                    block=True,
                    id="update_button",
                    style=menu_item_style
                ),

                dbc.Button(
                    "Export last plot",
                    block=True,
                    id="export_button",
                    style=menu_item_style
                ),

                dbc.FormGroup(
                    [
                        dbc.Checklist(
                            options=[
                                {"label": "Fix y-axis to [0,1]", "value": 1},
                                {"label": "Comparison Mode", "value": 2},
                            ],
                            value=[1],
                            id="options",
                            inline=True,
                            switch=True
                        ),
                    ],
                ),
            ],
            style=div_style
        ),

        html.Div(
            [
                dbc.Select(
                    id="ex_1_class",
                    options=[{'label': i, 'value': i} for i in data_handler.dataset.unique_labels_overall],
                    value='no_failure',
                    style=menu_item_style,
                ),

                dbc.Input(id="ex_1_index", type="number", value=0,
                          min=0,
                          style=menu_item_style,
                          className="mb-3"),

                html.H5('Example 1', style=text_style),
            ],
            id='example_1_div',
            style=div_style,
        ),

        html.Div(
            [
                dbc.Select(
                    id="ex_2_class",
                    options=[{'label': i, 'value': i} for i in data_handler.dataset.unique_labels_overall],
                    value='no_failure',
                    style=menu_item_style,
                ),

                dbc.Input(id="ex_2_index", type="number", value=0,
                          min=0,
                          style=menu_item_style,
                          className="mb-3"),
                html.H5('Example 2', style=text_style),
            ],

            id='example_2_div',
            style=div_style,
        ),

        dcc.Graph(id='graph'),
        html.Div(id='hidden_div', style={'display': 'none'}),
    ],

    style={'max-width': '2400px'}
)


def validate_input(part_selection, stream_ranges, toggle_values):
    # Convert string into enum
    part = DPE(part_selection)
    stream_indices = data_handler.ranges_string_to_indices(stream_ranges)

    fix_y_axis = True if 1 in toggle_values else False
    comparison_enabled = True if 2 in toggle_values else False

    return part, stream_indices, fix_y_axis, comparison_enabled


@app.callback(Output('graph', 'figure'),
              # Only update when button is pressed
              [Input('update_button', 'n_clicks')],
              state=[
                  State('part_selection', 'value'),
                  State('ex_1_index', 'value'),
                  State('ex_2_index', 'value'),
                  State('ex_1_class', 'value'),
                  State('ex_2_class', 'value'),
                  State('attribute_range', 'value'),
                  State('options', 'value'),
              ]
              )
def update_graph(n_clicks, part_selection, ex_1_index, ex_2_index, ex_1_class, ex_2_class, attribute_range, options):
    part, stream_indices, fix_y_axis, comparison_enabled = validate_input(part_selection, attribute_range, options)

    # TODO Make resampling frequency configurable
    df1, index_in_part_1, out_of_range_1 = data_handler.prepare_example(part, ex_1_class, ex_1_index, 4)
    traces1 = data_handler.get_traces(df1, stream_indices)

    # TODO Replace with output to html div.
    title = f'Example 1: {index_in_part_1} with label {ex_1_class}.'
    if out_of_range_1:
        title += f'(Requested index {ex_1_index} oor!.)'

    if comparison_enabled:
        df2, index_in_part_2, out_of_range_2 = data_handler.prepare_example(part, ex_2_class, ex_2_index, 4)

        # TODO Fix zoom in comparison mode, this does not work
        df2.index = df1.index

        traces2 = data_handler.get_traces(df2, stream_indices)

        traces = [None] * (len(traces1) + len(traces2))
        traces[::2] = traces1
        traces[1::2] = traces2
        shared_x_axes = False

        title += f'\nExample 2: {index_in_part_2} with label {ex_2_class}.'
        if out_of_range_2:
            title += f'(Requested index {ex_2_index} oor!.)'

    else:
        traces = traces1
        shared_x_axes = True

    fig = make_subplots(rows=len(traces),
                        cols=1,
                        shared_xaxes=shared_x_axes,
                        shared_yaxes=True)

    for index, trace in enumerate(traces):
        row_index = index + 1  # starts at 1
        fig.add_trace(trace, row=row_index, col=1)
        fig.update_xaxes(rangeslider={'visible': False}, row=row_index, col=1)

    # Enable range slider for last subplot
    # fig.update_xaxes(rangeslider={'visible': True, 'thickness': 0.5}, row=len(traces), col=1)

    fig.update_layout(height=len(traces) * 120,
                      template=plotly_theme,
                      title=title,
                      hoverlabel=dict(
                          bgcolor="#9e9e9e",
                          font_size=16,
                      ))

    if fix_y_axis:
        # Fix range of y axes for all subplots and set ticks
        fig.update_yaxes(range=[-0.1, 1.1])
        fig.update_yaxes(tickmode='linear', tick0=0.0, dtick=0.5)

    if comparison_enabled:
        fig.update_xaxes(visible=False, showticklabels=False)

    file_name = f'plot_{part_selection}_{ex_1_class}-{index_in_part_1}'
    file_name += f'_{ex_2_class}-{index_in_part_2}.svg' if comparison_enabled else '.svg'

    fig_export: go.Figure = copy(fig)
    fig_export.update_layout(title='')
    fig_export.update_xaxes(visible=False, showticklabels=False)
    path = config.get_dataset_path() + 'visualisation/' + file_name

    global last_fig, last_fig_path
    last_fig = fig_export
    last_fig_path = path

    return fig


# noinspection PyUnresolvedReferences
@app.callback(Output('hidden_div', 'children'),
              [Input('export_button', 'n_clicks')])
def export_figure(n_clicks):
    if n_clicks is None:
        raise PreventUpdate

    last_fig.write_image(last_fig_path)

    raise PreventUpdate


# @app.callback(
#     Output('example_2_div', 'style'),
#     [Input('options', 'value')],
#     [State('example_2_div', 'style')])
# def more_output(options, style):
#     print(options)
#     print(style)
#
#     if 2 in options:
#         if style == {}:
#             return div_style
#         else:
#             raise PreventUpdate
#     else:
#         if not style == {}:
#             return {}
#         else:
#             raise PreventUpdate


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
