#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 15:45:33 2023

@author: martazaniolo
"""


import seaborn as sns
sns.set_style("whitegrid") 
import plotly.io as pio
pio.renderers.default = 'browser'
import dash
import base64
from dash import dcc, html, Input, Output
from dash_factor_mapping_utils import *
import csv

import numpy as np
import sys
sys.path.append('ptreeopt')
sys.path.append('src')
import pickle
from os.path import exists
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.matlib as matlib
import pandas as pd
import configparser
import copy
#import dash_bootstrap_components as dbc

class Solution():
    pass


# Setup the style from the link:
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# Embed the style to the dashabord:
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


### load factors
DU_names = []
DU_factors = []
with open('scenarios.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            DU_names.append(row)
            line_count += 1
        else:
            DU_factors.append([float(r) for r in row])
            line_count += 1

factors = []
for a in [0,10,20]:#[0.0,5.0,10.0,15.0,20.0,25.0]:
    bb = copy.deepcopy(DU_factors)
    [b.extend([a]) for b in bb] 
    factors.extend(bb)
    
DU_factors = np.asarray(factors)
DU_names[0].append('deficit')

### load desal capacity

obj_time = []
obj_value = []
with open('expansions.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            obj_time.append(row)
            line_count += 1
        else:
            obj_value.append([float(r) for r in row])
            line_count += 1


#obj_value = np.asarray(obj_value)

###
for xs in ['_10', '_25']:#['_5','_10','_15','_20','_25' ]:
    obj_time10 = []
    #obj_value10 = []
    with open('expansions'+xs+'.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                obj_time10.append(row)
                line_count += 1
            else:
                obj_value.append([float(r) for r in row])
                line_count += 1
    
    
    #obj_value.append(obj_value10)
obj_value = np.asarray(obj_value)




font_size = '10px'
slider_carryover = dcc.RangeSlider(
    0,
    100,
    step=25,
    id='slider_carryover',
    value=[0,100],
    marks={str(i): str(i) for i in range(0,101,25)})
    # marks={ '0' : '0',
    #        '33.3' : '33.3',
    #        '66.6' : '66.6',
    #        '99.9' : '100'})
           
carry_label = html.Label('carryover in cachuma (%)', style={'font-size': font_size,'font-weight': 'bold', "text-align": "left"})

slider_cach_all = dcc.RangeSlider(
    0,
    100,
    step=25,
    id='slider_cach_all',
    value=[0,100],
    marks={str(i): str(i) for i in range(0,101,25)})
cach_all_label = html.Label('Cachuma allocation (%)', style={'font-size': font_size,'font-weight': 'bold', "text-align": "left"})

slider_swp_all = dcc.RangeSlider(
    0,
    100,
    step=50,
    id='slider_swp_all',
    value=[0,100],
    marks={str(i): str(i) for i in range(0,101,50)})
swp_all_label = html.Label('SWP allocation (%)', style={'font-size': font_size,'font-weight': 'bold', "text-align": "left"})


slider_gib_stor = dcc.RangeSlider(
    1,
    3,
    step=1,
    id='slider_gib_stor',
    value=[1,3],
    marks={str(i): str(i) for i in range(0,4,1)})

gibr_stor_label = html.Label('Gibraltar sedimentation scenario (%)', style={'font-size': font_size,'font-weight': 'bold', "text-align": "left"})

slider_intensity = dcc.RangeSlider(
    0,
    2,
    step=1,
    id='slider_intensity',
    value=[0,2],
    marks={'0':'historical', '1': 'intense', '2': 'long+intense'})

intensity_label = html.Label('droughts', style={'font-size': font_size,'font-weight': 'bold', "text-align": "left"})

slider_efficiency = dcc.RangeSlider(
    0.7,
    0.9,
    step=0.1,
    id='slider_efficiency',
    value=[0.7, 0.9],
    marks={str(i): str(i) for i in range(0,3,1)})

efficiency_label = html.Label('Desal capacity factor', style={'font-size': font_size,'font-weight': 'bold', "text-align": "left"})

slider_demand = dcc.RangeSlider(
    0,
    2,
    step=1,
    id='slider_demand',
    value=[0 , 2],
    marks={0: {'label': 'lower bound'},
           1: {'label': 'baseline'},
           2: {'label': 'upper bound'}
           }
    )

demand_label = html.Label('Demand scenario', style={'font-size': font_size,'font-weight': 'bold', "text-align": "left"})


slider_water_sold = dcc.RangeSlider(
    0,
    1,
    step=1,
    id='slider_water_sold',
    value=[0,1],
    marks={ 1: {'label': 'Montecito'},
            2: {'label': 'Montecito+Lacumbra'}})

watersold_label = html.Label('Desal water sold', style={'font-size': font_size,'font-weight': 'bold', "text-align": "left"})

slider_deficit = dcc.RangeSlider(
    0,
    20,
    step=10,
    id='slider_deficit',
    value=[0,20],
    marks={str(i): str(i) for i in range(0,20,10)})

deficit_label = html.Label('Max annual deficit', style={'font-size': font_size,'font-weight': 'bold', "text-align": "left"})



year_robust = dcc.Slider(min=10, max=50, step=10, value=30, id='year_robust', included=False)
#year_label = html.Label('Select robustness horizon', style={'font-size': font_size,'font-weight': 'bold', "text-align": "left"})

desal_capacity = dcc.Slider(3125, 10000, value=7500,
                            id='desal_capacity',
    marks={
        3125: {'label': '3,125'},
        5500: {'label': '5,500'},
        7500: {'label': '7,500'},
        10000: {'label': '10,000'}
    },
    included=False
)
desal_label = html.Label('Select desal capacity', style={'font-size': font_size,'font-weight': 'bold', "text-align": "left"})


######### PLOTS #########
[factor, id_factors] = feature_rank(DU_factors, obj_value, DU_names, 30, 7500)
heatmap = heatmap_plot(DU_factors, obj_value, DU_names, 30, 7500, id_factors)
cap_rob = capacity_robustness(obj_value, 7500)
[gain_rob, loss_rob] = difference_robustness(obj_value, DU_factors, DU_names, id_factors, 7500, 30)

#heatmap10 = heatmap_diff(DU_factors, obj_value, obj_value10, DU_names, 30, 7500, id_factors)
#heatmap25 = heatmap_diff(DU_factors, obj_value, obj_value25, DU_names, 30, 7500, id_factors)

factor_plot = dcc.Graph(
        id='factor_plot',
        figure=factor,
        style={'height': '50vh', 'padding':10}
    )

heat_plot = dcc.Graph(
        id='heat_plot',
        figure=heatmap,
        style={'height': '50vh','padding':10}
    )

caprob_plot = dcc.Graph(
        id='caprob_plot',
        figure=heatmap,
        style={'height': '50vh', 'padding':10}
    )

gain_plot = dcc.Graph(
        id='gain_plot',
        figure=heatmap,
        style={'height': '50vh', 'padding':10}
    )

loss_plot = dcc.Graph(
        id='loss_plot',
        figure=heatmap,
        style={'height': '50vh', 'padding':10}
    )


title = html.Div([
     html.H1('Desal expansion robustness in Santa Barbara ')
 ])


year_title = html.Div([
     html.H4('Select robustness horizon')
 ],style={'margin-left':'5'})

capacity_title = html.Div([
     html.H4('Select desal capacity')
 ],style={'margin-left':'5'})

future_title = html.Div([
     html.H4('Constrain scenarios')
 ],style={'margin-left':'5'})


tit = html.Div(children=title,  style={'vertical-align': 'top','width': '100%', 'display': 'inline-block'})
row11 = html.Div(children=[year_title, year_robust, capacity_title, desal_capacity, 
                           future_title, carry_label, slider_carryover, cach_all_label, slider_cach_all, swp_all_label, slider_swp_all, 
                                                      gibr_stor_label, slider_gib_stor, intensity_label, slider_intensity, efficiency_label, slider_efficiency, 
                                                      watersold_label, slider_water_sold, demand_label, slider_demand, deficit_label, slider_deficit], style={ 'width': '30%', 'display': 'inline-block'}) #'height': 1000
#row12 = html.Div(children=[future_title, carry_label, slider_carryover, cach_all_label, slider_cach_all, swp_all_label, slider_swp_all, 
#                           gibr_stor_label, slider_gib_stor, intensity_label, slider_intensity, efficiency_label, slider_efficiency, 
#                          watersold_label, slider_water_sold, demand_label, slider_demand], style={'vertical-align': 'top','width': '35%', 'display': 'inline-block'}) #'height': 1000
row21 = html.Div(children=[factor_plot], style={ 'vertical-align': 'top','width': '30%', 'display': 'inline-block'}) #'height': 1000
row22 = html.Div(children=[heat_plot], style={'vertical-align': 'top','width': '30%', 'display': 'inline-block'}) #'height': 1000
row31 = html.Div(children=[caprob_plot], style={'vertical-align': 'top','width': '30%', 'display': 'inline-block'}) #'height': 1000
row41 = html.Div(children=[gain_plot], style={ 'vertical-align': 'top','width': '30%', 'display': 'inline-block'}) #'height': 1000
row42 = html.Div(children=[loss_plot], style={'vertical-align': 'top','width': '30%', 'display': 'inline-block'}) #'height': 1000
#row51 = html.Div(children=[heat_plot10], style={'vertical-align': 'top','width': '30%', 'display': 'inline-block'}) #'height': 1000
#row52 = html.Div(children=[heat_plot25], style={'vertical-align': 'top','width': '30%', 'display': 'inline-block'}) #'height': 1000
layout = html.Div(children=[tit, row11, row31, row21, row22, row41, row42], style={"text-align": "center"})
app.layout = layout
server = app.server


@app.callback(
    Output('factor_plot', 'figure'),
    Output('heat_plot', 'figure'),
    Output('caprob_plot', 'figure'),
    Output('gain_plot', 'figure'),
    Output('loss_plot', 'figure'),
    Input('slider_carryover', 'value'),#1
    Input('slider_cach_all', 'value'),#2
    Input('slider_swp_all', 'value'),#3
    Input('slider_gib_stor', 'value'),#4
    Input('slider_intensity', 'value'),#5
    Input('slider_efficiency', 'value'),#6
    Input('slider_water_sold', 'value'),#7
    Input('slider_demand', 'value'),#8
    Input('year_robust', 'value'),#9
    Input('desal_capacity', 'value'), #10
    Input('slider_deficit', 'value') )#11

def update_figures(value1, value2, value3, value4, value5, value6, value7, value8, value9, value10, value11):
    
    #intens = ['111', '124']#'baseline', 'lower_bound', 'upper_bound']
    if value1[1] == 99.9:
        value1[1] = 100
        
    idx = np.where( 
        (DU_factors[:,0] >= value4[0]) & (DU_factors[:,0] <= value4[1]) & #gibraltar
        (DU_factors[:,1] >= value1[0]) & (DU_factors[:,1] <= value1[1]) &     #carryover
        (DU_factors[:,2] >= value2[0]/100) & (DU_factors[:,2] <= value2[1]/100) &    #cach allocation
        (DU_factors[:,3] >= value3[0]/100) & (DU_factors[:,3] <= value3[1]/100) &  #swp alloc
        (DU_factors[:,4] >= value5[0]) & (DU_factors[:,4] <= value5[1]) & #intens
        (DU_factors[:,5] >= value7[0]) & (DU_factors[:,5] <= value7[1]) &    #water sold
        (DU_factors[:,6] >= value6[0]) & (DU_factors[:,6] <= value6[1]) &    #efficiency
        (DU_factors[:,7] >= value8[0]) & (DU_factors[:,7] <= value8[1]) &   #demand
        (DU_factors[:,8] >= value11[0]) & (DU_factors[:,8] <= value11[1]) ) #deficit
        
    DU_filter  = DU_factors[idx[0], :]
    obj_filter = obj_value[idx]
    [factor, id_factors] = feature_rank(DU_filter, obj_filter, DU_names, value9, value10)
    heatmap = heatmap_plot(DU_filter, obj_filter, DU_names, value9, value10, id_factors)
    cap_rob = capacity_robustness(obj_filter, value10)
    [gain_rob, loss_rob] = difference_robustness(obj_filter, DU_filter, DU_names, id_factors, value10, value9)
   # heatmap10 = heatmap_diff(DU_filter, obj_filter, obj_filter10, DU_names, value9, value10, id_factors)
    #heatmap25 = heatmap_diff(DU_filter, obj_filter, obj_filter25, DU_names, value9, value10, id_factors)
    
    
# 1gibraltar_storage
# 2cachuma_carryover
# 3cachuma_allocation
# 4swp_allocation
# 5intensity
# 6water_sold
# 7efficiency
# 8demand_projection
    
    return [factor, heatmap, cap_rob, gain_rob, loss_rob]
        
    

if __name__ == '__main__':
    app.run_server(debug = True)





