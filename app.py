# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 08:21:16 2019

@author: sjv1030_hp
"""


import dash
import dash_core_components as dcc
#import dash_table as dt
import dash_html_components as html
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.api as sms

from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate

from fredapi import Fred

import plotly.graph_objs as go

app = dash.Dash(__name__)
server = app.server
                
fred = Fred(api_key='0d3a129121b29e16035b20ea3947ecf5')

gdp = fred.get_series('GDPC1')
inf = fred.get_series('PCEPILFE')
isratio = fred.get_series('ISRATIO')
lei = fred.get_series('USALOLITONOSTSAM')

infyy = inf.pct_change(12)*100
gdpyy = gdp.pct_change(4)*100
spx = fred.get_series('SP500')
wti = fred.get_series('WTISPLC')
yc = fred.get_series('T10Y2YM')
dollar = fred.get_series('TWEXBMTH')
unrate = fred.get_series('UNRATE')
ahe = fred.get_series('AHETPI')
hhdebt = fred.get_series('TDSP')
pce = fred.get_series('PCE')

pceyy = pce.pct_change(12)*100
aheyy = ahe.pct_change(12)*100
mtg = fred.get_series('MORTGAGE30US')
hpi = fred.get_series('SPCS20RSA')
sales = fred.get_series('HSN1F')
starts = fred.get_series('HOUST')

hpiyy = hpi.pct_change(12)*100
salesyy = sales.pct_change(12)*100
startsyy = starts.pct_change(12)*100
        
database = {
        'gdpyy':gdpyy,
        'infyy':infyy,
        'isratio':isratio,
        'lei':lei,
        'spx':spx,
        'wti':wti,
        'yc':yc*100,
        'dollar':dollar,
        'unrate':unrate,
        'aheyy':aheyy,
        'pceyy':pceyy,
        'hhdebt':hhdebt,
        'mtg':mtg,
        'hpiyy':hpiyy,
        'salesyy':salesyy,
        'startsyy':startsyy
        }


def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

intro = '''
My final project plots data sourced from the St. Louis Federal Reserve via the
Fred API. A user can select the type of variable to be displayed using the dropdown.
Then, the table and plots will automatically update with 4 key variables
for the chosen type. The plots are meant to be displayed as a small multiple,
but the plots are also interactive to allow the user to zoom, pan, etc.

The table below shows the values for the latest, minimum, maximum, and z-scores.
NAs are dropped and the data series were Winsorized to the bottom 10% and top 90%, 
to remove outliers.

The intended user of this interactive tool is an economsit, financial analyst,
or even financial reporter interested in looking at various aspects of the US
economy. Furthermore, the user can further the analysis by investigating the
sensitivities of a variable subject to the variation of independent variables
by performing a simple, contemporaneous regression.
'''

app.layout = html.Div([   
    html.H2(children='Silverio J. Vasquez - Data 608 - Final Project'),  
    
    html.Div(children=[
            html.H3(children='Overview'),
            dcc.Markdown(children=intro),  
    ]), 
    
    dcc.Dropdown(style={'width':'500px'},id='data-dropdown', 
        options=[
        {'label': 'Macro', 'value': 'macro'},
        {'label': 'Financial', 'value': 'financial'},
        {'label': 'Consumer', 'value': 'consumer'},
        {'label': 'Housing', 'value': 'housing'}
        ], value='macro',
    ),
    html.Br(),
    dcc.Loading(id="loading-1",
            children=[ 
                html.Div(id='dd-output-container'), 
                
                html.Div([
                    html.Div([
                        dcc.Graph(id='g1')
                    ], className="six columns"),
            
                    html.Div([
                        dcc.Graph(id='g2')
                    ], className="six columns"),
                ], className="row"),
            
                html.Div([
                    html.Div([
                        dcc.Graph(id='g3')
                    ], className="six columns"),
            
                    html.Div([
                        dcc.Graph(id='g4')
                    ], className="six columns"),
                ], className="row"),
                      html.Div([
    dcc.Dropdown(style={'width':'500px'},
        id = 'dropdown-to-show_or_hide-element',
        options=[
            {'label': 'Show regression', 'value': 'on'},
            {'label': 'Hide regression', 'value': 'off'}
        ], 
        value = 'off'
    ),html.P('Above is a dropdown to select further functionality, \
               i.e., an ordinary least-squares regression.'),

    # Create Div to place a conditionally visible element inside
    html.Div([
        html.Br(),
        html.P('Select variables to create a contemporaneous regression. \
               The first variable will be the dependent variable. \
               The subsequent variables will be the indepdent variables.'),
        # Create element to hide/show, in this case an 'Input Component'
        dcc.Dropdown(id='regression',
        options=[
            {'label':'Inflation','value':'infyy'},
            {'label':'IS Ratio','value':'isratio'},
            {'label':'LEI','value':'lei'},
            {'label':'Unemployment Rate','value':'unrate'},
            {'label':'Wage Growth','value':'aheyy'},
            {'label':'Personal Consumption','value':'pceyy'},
            {'label':'House Price','value':'hpiyy'},
            {'label':'Housing Sales','value':'salesyy'},
            {'label':'Housing Starts','value':'startsyy'}
                ],
        placeholder = 'select variables',
         multi=True
        ),
        html.Br(),
        html.Button('Submit', id='button'),
        html.Br(),
        dcc.Markdown(id='explanation'),
        html.Div([],id='regression-output-header'),
        html.Div([],id='regression-output'),
        html.Br(),
        html.Br()
    ], id = 'element-to-hide', style= {'display': 'none'}), # <-- This is the line that will be changed by the dropdown callback
    
    ])     
            ])
      
], style={'width': '90%', 'display': 'inline-block', 'vertical-align': 'middle'})
    
@app.callback(
   Output(component_id='element-to-hide', component_property='style'),
   [Input(component_id='dropdown-to-show_or_hide-element', component_property='value')])

def show_hide_element(visibility_state):
    if visibility_state == 'on':
        return {'display': 'block'}
    if visibility_state == 'off':
        return {'display': 'none'}

@app.callback(
   [Output(component_id='regression-output-header', component_property='children'),
    Output('regression-output','children'),
    Output('explanation','children')],
    [Input(component_id='button',component_property='n_clicks')],
    [State('regression','value')])    

def show_regression(n_clicks,value):
    if value is None:
        raise PreventUpdate
    if len(value) < 2:
        return html.P('You need to select more than one variable.'),None,None
    else:
        reg_df = pd.DataFrame(database[value[0]])
        for i in range(1,len(value)):
            reg_df = pd.concat([reg_df,pd.DataFrame(database[value[i]])],axis=1)    
        reg_df.columns = value
        reg_df.dropna(inplace=True)
        y = reg_df[value[0]]
        x = reg_df[value[1:]]
        X = sm.add_constant(x)
        model = sm.OLS(y,X).fit(cov_type='HC3')
        
        model_df1 = pd.DataFrame(
                {'nobs':model.nobs,
                 'Adj_R-square':round(model.rsquared_adj,3),
                 'DW_test':round(sms.durbin_watson(model.resid),3)
                 },index=[0])             
        
        names = ['constant'] + value[1:]

        model_df2 = pd.DataFrame.from_dict(
                {
                 'regressors':names,
                 'coefficients':[round(x,3) for x in model.params],
                 'T-stats':[round(x,3) for x in model.tvalues]
                 })
        explanation = '''
           Below are two tables showing the regression output. 
           The first table shows the number of observations, the adjusted \
           r-square, and the Durbin Watson statistic for serial correlation.

           The second table shows the independent variables entered in the \
           order selected with the constant displayed first. \
           The second column displays the beta coefficient, and \
           the third column shows the T-statistic for a two-tailed test.
           '''
        return generate_table(model_df1),generate_table(model_df2),explanation
        
def _replace_tags(html):
    return html.replace('<table><td>','html.Table([html.Tr(')
    
@app.callback([
    Output('dd-output-container','children'),
    Output('g1','figure'),
    Output('g2','figure'),
    Output('g3','figure'),
    Output('g4','figure')
], [Input('data-dropdown', 'value')])

def multi_output(value):
    if value is None:
        raise PreventUpdate
    
    series_list = list(_get_data(value))
    series_list_ = [_clip(x) for x in series_list]
    series1 = series_list_[0]
    series2 = series_list_[1]
    series3 = series_list_[2]
    series4 = series_list_[3]
                                
    table = _make_table(series1,series2,series3,series4,value)
    
    figure1 = go.Figure(
                [go.Scatter(x=series1.loc['2000'::].index, y=series1.loc['2000'::].values)])
    figure1.update_layout(title_text=table['Indicator'][0],xaxis_rangeslider_visible=True)

    figure2 = go.Figure(
                [go.Scatter(x=series2.loc['2000'::].index, y=series2.loc['2000'::].values)])
    figure2.update_layout(title_text=table['Indicator'][1],xaxis_rangeslider_visible=True)

    figure3 = go.Figure(
                [go.Scatter(x=series3.loc['2000'::].index, y=series3.loc['2000'::].values)])
    figure3.update_layout(title_text=table['Indicator'][2],xaxis_rangeslider_visible=True)
        
    figure4 = go.Figure(
                [go.Scatter(x=series4.loc['2000'::].index, y=series4.loc['2000'::].values)])
    figure4.update_layout(title_text=table['Indicator'][3],xaxis_rangeslider_visible=True)
    
    return generate_table(table), figure1, figure2, figure3, figure4

def _get_data(value):
    if value == 'macro':
        return database['gdpyy'], database['infyy'], database['isratio'], database['lei']
    
    elif value == 'financial':
        return database['spx'], database['wti'], database['yc'], database['dollar']
    
    elif value == 'consumer':
        return database['unrate'], database['aheyy'], database['hhdebt'], database['pceyy']
    
    else:
        return database['mtg'], database['hpiyy'], database['salesyy'], database['startsyy']
        
    
def _make_table(x1,x2,x3,x4,key='macro'):
    if key == 'macro':
        _dict = {
        'Real GDP YoY%':[_latest(x1),_min(x1),_max(x1),_zscore(x1)],
        'Core PCE YoY%':[_latest(x2),_min(x2),_max(x2),_zscore(x2)],
        'Inventory-to-Sales Ratio':[_latest(x3),_min(x3),_max(x3),_zscore(x3)],
        'US LEI (OECD)':[_latest(x4),_min(x4),_max(x4),_zscore(x4)]
        }
    elif key == 'financial':
        _dict = {
        'S&P 500':[_latest(x1),_min(x1),_max(x1),_zscore(x1)],
        'WTI - Oil Price $':[_latest(x2),_min(x2),_max(x2),_zscore(x2)],
        'Yield Curve (bps)':[_latest(x3),_min(x3),_max(x3),_zscore(x3)],
        'Trade Weighted Dollar Index':[_latest(x4),_min(x4),_max(x4),_zscore(x4)]
        }
    elif key == 'consumer':
        _dict = {
        'Unemployment Rate %':[_latest(x1),_min(x1),_max(x1),_zscore(x1)],
        'Average Hourly Earnings YoY%':[_latest(x2),_min(x2),_max(x2),_zscore(x2)],
        'Household Debt % Income':[_latest(x3),_min(x3),_max(x3),_zscore(x3)],
        'Personal Consumption YoY%':[_latest(x4),_min(x4),_max(x4),_zscore(x4)]        
        }
    else:
        _dict = {
        '30-Year Mortgage Rate':[_latest(x1),_min(x1),_max(x1),_zscore(x1)],
        'Case-Shiller House Prices YoY%':[_latest(x2),_min(x2),_max(x2),_zscore(x2)],
        'New Single Family Houses Sold YoY%':[_latest(x3),_min(x3),_max(x3),_zscore(x3)],
        'New Housing Starts YoY%':[_latest(x4),_min(x4),_max(x4),_zscore(x4)],        
        }
        
    return _make_df(_dict)

def _make_df(d):
    new_dict = d    
    df = pd.DataFrame(columns=['Latest','Minimum','Maximum','Z-Score'])
    df = df.append(pd.DataFrame.from_dict(new_dict,orient='index',
                            columns=['Latest','Minimum','Maximum','Z-Score']))
    df.reset_index(inplace=True)
    df.columns = ['Indicator','Minimum','Maximum','Latest','Z-Score']
    return df


def _clip(series):
    _series = pd.Series(series)
    return _series.clip(lower=_series.quantile(0.1),
                       upper=_series.quantile(0.9))

def _latest(series):
    return round(series.iloc[-1],3)
def _min(series):
    return round(series.min(),3)
def _max(series):
    return round(series.max(),3)
def _zscore(series):
    return round((series.iloc[-1] - series.mean()) / series.std(),3)

if __name__ == '__main__':
    app.run_server(debug=True)