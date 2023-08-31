import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import norm  # Import scipy's norm function for normal distribution

app = dash.Dash(__name__)

# Sample data
time = np.linspace(0, 10, 100)
mean_data_1 = np.sin(time) + 5
mean_data_2 = np.cos(time) + 7
std_dev = 0.5  # Example standard deviation

app.layout = html.Div([
    dcc.Graph(
        id='main-plot',
        figure={
            'data': [
                go.Scatter(x=time, y=mean_data_1, mode='lines', name='Line 1'),
                go.Scatter(x=time, y=mean_data_2, mode='lines', name='Line 2')
            ],
            'layout': {'title': 'Mean of Normal Distribution Over Time'}
        }
    ),
    html.Div(
        style={'display': 'flex', 'flexDirection': 'row'},
        children=[
            dcc.Graph(id='sub-plot'),
            html.Div(id='table-container')
        ]
    )
])

@app.callback(
    Output('sub-plot', 'figure'),
    Output('table-container', 'children'),
    Input('main-plot', 'hoverData')
)
def display_sub_plot_and_table(hover_data):
    if hover_data:
        x_selected = hover_data['points'][0]['x']
        closest_index = np.argmin(np.abs(time - x_selected))
        hover_time = time[closest_index]
        hover_mean_2 = mean_data_2[closest_index]
        
        y_sub = np.linspace(hover_mean_2 - 3 * std_dev, hover_mean_2 + 3 * std_dev, 100)
        x_sub = norm.pdf(y_sub, hover_mean_2, std_dev)
        
        sub_trace = go.Scatter(x=x_sub, y=y_sub, mode='lines', name='Sub Plot')
        sub_layout = go.Layout(title='Normal Distribution at Time {}'.format(hover_time))
        
        # Rotate the sub-plot by swapping x and y axes and updating labels
        sub_layout.xaxis = go.layout.XAxis(title='Density')
        sub_layout.yaxis = go.layout.YAxis(title='X Axis')
        
        sub_fig = {'data': [sub_trace], 'layout': sub_layout}
        
        # Create a table with some example values
        table_data = [{'Metric': 'Mean', 'Value': hover_mean_2},
                      {'Metric': 'Standard Deviation', 'Value': std_dev}]
        table = dash_table.DataTable(
            columns=[{'name': col, 'id': col} for col in table_data[0].keys()],
            data=table_data
        )
        
        return sub_fig, table
    
    return {'data': [], 'layout': {}}, None

if __name__ == '__main__':
    app.run_server(debug=True)
