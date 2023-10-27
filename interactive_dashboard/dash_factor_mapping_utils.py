#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 16:02:16 2023

@author: martazaniolo
"""

from sklearn.ensemble import GradientBoostingClassifier
import csv
import seaborn as sns
sns.set_style("whitegrid") 
import plotly.express as px
import plotly.graph_objs as go
import plotly.tools as pxtool
from plotly.subplots import make_subplots
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt


def gradient_boost(DU_factors, bool_obj, DU_names):
    # create a gradient boosted classifier object
    gbc = GradientBoostingClassifier(n_estimators=500,
                                     learning_rate=0.01,
                                     max_depth=5)
    # fit the classifier
    gbc.fit(DU_factors, bool_obj)
    
    
    ##### Factor Ranking #####
     
    # Extract the feature importances
    feature_importances = deepcopy(gbc.feature_importances_)
     
    # rank the feature importances and plot
    importances_sorted_idx = np.argsort(feature_importances)
    sorted_names = [DU_names[0][imp] for imp in importances_sorted_idx]
    return [feature_importances, importances_sorted_idx, sorted_names]
    

def capacity_robustness(obj_value, id_select):
    fig1 = make_subplots(rows=1, cols=1, subplot_titles=("Robustness of other Desal capacities", ""))
    desal_size = np.asarray([3125, 5500, 7500, 10000])
    
    id_ = (np.abs(desal_size - id_select)).argmin()

    
    ds = []
    for thr in desal_size:
        bool_obj = [o<thr for o in obj_value]
        ds.append(np.mean(bool_obj)*100)
        
    color = []
    txt = []
    for i in range(4):
        if i == id_:
            color.append('#AAE2FB')    #('#FF8E00')
            txt.append('Selected capacity')
        else:
            color.append('#003F7D')
            txt.append('')

    
    fig1.add_trace(go.Bar(#name = 'Other desal expansion options',
            x=['current 3,125 AF','small expansion <br> 5,500 AFy', 'medium expansion <br> 7,500 AFy', 'max expansion <br> 10,000 AFy'], #[1,2,3,4],
            y=ds
        ))
    fig1.update_traces(marker_color=color, texttemplate=txt, textfont_color="black")
           
           
    #fig1.add_trace(go.Bar(name = 'Selected expansion'))
    fig1.update_yaxes(range=[0, 100], title = '% of scenarios where robustness is ensured' )
    fig1.update_xaxes(title = 'Desal capacity' )
    fig1.update_layout(template='plotly_white')
    
    return fig1



def feature_rank(DU_factors, obj_value, DU_names, year, desal_size):
    fig1 = make_subplots(rows=1, cols=1, subplot_titles=("Factor importance", ""))


    yy = int(( year/10 )- 1)
    obj = obj_value[:,yy] 

    threshold = desal_size
    
    bool_obj = [o<threshold for o in obj]
    
    if sum(bool_obj) == 0:
        feature_importances    = [0]*8
        importances_sorted_idx = range(7)
        sorted_names = DU_names
    elif sum(bool_obj) == len(bool_obj):
        feature_importances    = [0]*8
        importances_sorted_idx = range(7)
        sorted_names = DU_names
    else: 
        [feature_importances, importances_sorted_idx, sorted_names] = gradient_boost(DU_factors, bool_obj, DU_names)
     
    fig1.add_trace(go.Bar(
            x=[feature_importances[i] for i in importances_sorted_idx],
            y=sorted_names,
            orientation='h'), row = 1, col = 1)
    
    fig1.update_traces(marker_color= '#FF8E00')
    id_factors = importances_sorted_idx[-2:]
    fig1.update_xaxes(range=[0, 1], title = 'Fraction of variability explained by factors')
    fig1.update_layout(template='plotly_white')

    
    return [fig1, id_factors]
    
def heatmap_plot(DU_factors, obj_value, DU_names, year, desal_size, id_factors):
    fig1 = make_subplots(rows=1, cols=1, subplot_titles=("Main drivers of robustness", ""))
    yy = int(( year/10 )- 1)
    obj = obj_value[:,yy] 
    threshold = desal_size
    
    bool_obj = [o<threshold for o in obj]

    important_factors = DU_factors[:, id_factors  ]
    
    un2 = np.unique(important_factors[:,0])
    un1 = np.unique(important_factors[:,1])
    
    #print(DU_names[0][id_factors[0]])
    perc_success = []
    prc = np.zeros((len(un2) ,len(un1)))
    tot_perc = np.zeros( (len(important_factors),1) )
    k1 = 0
    k2 = 0
    for u1 in un1:
        k2 = 0
        for u2 in un2:
            idx = np.where( np.logical_and(important_factors[:,1] == u1, important_factors[:,0] == u2 ))
            bool_point = [ bool_obj[ii] for ii in idx[0]]
            perc_success.append(sum(bool_point) /len(idx[0]))
            
            ps = sum(bool_point)/len(idx[0])
            tot_perc[idx[0]]=ps
            
            prc[k2][k1] = ps
            #print(prc)
            k2 += 1
        k1 += 1
     

    #heat_dec = sns.heatmap( prc,  cmap='Greens', linewidths=0,  vmin=0, vmax=+0.2, annot=True, fmt=".0%", cbar=False) 
    fig1.add_trace(go.Heatmap(z = prc, x = un1, y = un2, colorscale = 'RdBu', zmax = 1, zmin = 0 ,
                                      colorbar=dict(title="Fraction of scenarios where robustness is ensured", titleside="right") ), row = 1, col = 1)

    fig1.update_xaxes(title = DU_names[0][id_factors[1] ], tickvals = un1 ) 
    fig1.update_yaxes(title = DU_names[0][id_factors[0] ], tickvals = un2 )
    
    return fig1
    
def heatmap_diff(DU_factors, obj_value, obj_value10, DU_names, year, desal_size, id_factors):
    fig1 = make_subplots(rows=1, cols=1, subplot_titles=("Allow deficit", ""))
    yy = int(( year/10 )- 1)
    obj = obj_value[:,yy] 
    obj10 = obj_value10[:,yy] 
    threshold = desal_size
    
    bool_obj = [o<threshold for o in obj]
    bool_obj10 = [o<threshold for o in obj10]

    important_factors = DU_factors[:, id_factors  ]
    
    un2 = np.unique(important_factors[:,0])
    un1 = np.unique(important_factors[:,1])
    
    #print(DU_names[0][id_factors[0]])
    perc_success = []
    prc = np.zeros((len(un2) ,len(un1)))
    prc10 = np.zeros((len(un2) ,len(un1)))
    tot_perc = np.zeros( (len(important_factors),1) )
    k1 = 0
    k2 = 0
    for u1 in un1:
        k2 = 0
        for u2 in un2:
            idx = np.where( np.logical_and(important_factors[:,1] == u1, important_factors[:,0] == u2 ))
            bool_point = [ bool_obj[ii] for ii in idx[0]]
            bool_point10 = [ bool_obj10[ii] for ii in idx[0]]
            perc_success.append(sum(bool_point) /len(idx[0]))
            
            ps = sum(bool_point)/len(idx[0])
            ps10 = sum(bool_point10)/len(idx[0])
            tot_perc[idx[0]]=ps
            
            prc[k2][k1] = ps
            prc10[k2][k1] = ps10
            #print(prc)
            k2 += 1
        k1 += 1
     

    #heat_dec = sns.heatmap( prc,  cmap='Greens', linewidths=0,  vmin=0, vmax=+0.2, annot=True, fmt=".0%", cbar=False) 
    fig1.add_trace(go.Heatmap(z = prc10 - prc, x = un1, y = un2, colorscale = 'Blues', zmax = 0.3, zmin = 0 ,
                                      colorbar=dict(title="Robustness gain [Fraction of scenarios]", titleside="right") ), row = 1, col = 1)

    fig1.update_xaxes(title = DU_names[0][id_factors[1] ], tickvals = un1 ) 
    fig1.update_yaxes(title = DU_names[0][id_factors[0] ], tickvals = un2 )
    
    return fig1
    

def deficit_fig(avg_y10, avg_y20, ny10, ny20):
    sns.set_style("whitegrid") 
    fig = make_subplots(subplot_titles=("Accepted deficit", "") , specs=[[{"secondary_y": True}]])
    #fig = go.Figure()
    fig.add_trace(go.Box(x=[0], y=[0], name='0%', marker_color = 'indianred')) #x=[1]*len(avg_y10),
    fig.add_trace(go.Box(x = [1]*len(avg_y10), y=avg_y10, name='10%', marker_color = 'indianred')) #x=[2]*len(avg_y20), 
    fig.add_trace(go.Box(x = [2]*len(avg_y10), y=avg_y20, name='20%', marker_color = 'indianred')) #x=[0], 
    
    fig.add_trace(go.Scatter(x=[0,1,2], y=[0,np.mean(ny10), np.mean(ny20)],
                    mode='lines+markers', line=dict(color='royalblue'),
                    name='avg number of deficit years'), secondary_y=True)
        
    fig.update_xaxes(title = 'Max annual deficit allowed', tickvals = [0,1,2], ticktext = ['0%', '10%', '20%'] ) 
    fig.update_yaxes(title = 'Deficit as fraction of demand during deficit years', secondary_y = False)
    fig.update_yaxes(title = 'Number of deficit years', secondary_y = True, )     
    
    fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01))

    return fig





def difference_robustness(obj_value, DU_factors, DU_names, id_factors, id_select, year):
    
    yy = int(( year/10 )- 1)
    
    desal_size = np.asarray([3125, 5500, 7500, 10000])
    
    id_ = (np.abs(desal_size - id_select)).argmin()

            
    desal_size = [3125, 5500, 7500, 10000]
    fig1 = make_subplots(rows=1, cols=1, subplot_titles=("Loss in robustness of smaller desal", ""))
    fig2 = make_subplots(rows=1, cols=1, subplot_titles=("Gain in robustness of larger desal", ""))
    
    if id_ < 4:
    
        bool_obj = []
        for thr in desal_size:
            bool_obj.append([o[yy]<thr for o in obj_value])
            
    
        if id_>0:
            if id_ == 3:
                gain_previous = [not(a==b) for a, b in zip(bool_obj[id_], bool_obj[id_-1])]
                gain_next = 'no'
            else:              
                gain_previous = [not(a==b) for a, b in zip(bool_obj[id_], bool_obj[id_-1])]
                gain_next     = [not(a==b) for a, b in zip(bool_obj[id_+1], bool_obj[id_])]
        else:
            gain_next     = [not(a==b) for a, b in zip(bool_obj[id_+1], bool_obj[id_])]
            gain_previous = 'no'
    
    
        important_factors = DU_factors[:, id_factors  ]
        un2 = np.unique(important_factors[:,0])
        un1 = np.unique(important_factors[:,1])
        
        
        ######## additional success
    
        
        
        if not(gain_previous == 'no'):

            perc_success = []
            prc = np.zeros((len(un2) ,len(un1)))
            tot_perc = np.zeros( (len(important_factors),1) )
            k1 = 0
            k2 = 0
            for u1 in un1:
                k2 = 0
                for u2 in un2:
                    idx = np.where( np.logical_and(important_factors[:,1] == u1, important_factors[:,0] == u2 ))
                    bool_point = [ gain_previous[ii] for ii in idx[0]]
                    perc_success.append(sum(bool_point) /len(idx[0]))
                    
                    ps = sum(bool_point)/len(idx[0])
                    tot_perc[idx[0]]=ps
                    
                    prc[k2][k1] = ps
                    #print(prc)
                    k2 += 1
                k1 += 1
             
        
            #heat_dec = sns.heatmap( prc,  cmap='Greens', linewidths=0,  vmin=0, vmax=+0.2, annot=True, fmt=".0%", cbar=False) 
            fig1.add_trace(go.Heatmap(z = prc, x = un1, y = un2, colorscale = 'Reds', zmax = 1, zmin = 0,  
                                      colorbar=dict(title="Robustness loss", titleside="right") ), row = 1, col = 1)
            #fig1.update_zaxis(title = 'Robustness fraction')   titleside="top"
            #fig1.update_layout( title_text = "Capex ratio and LCOW",
                               #title_font_size = 20, showlegend=False)
            #print(prc)
            fig1.update_xaxes(title = DU_names[0][id_factors[1] ], tickvals = un1 ) 
            fig1.update_yaxes(title = DU_names[0][id_factors[0] ], tickvals = un2 )
            
        
    
        ######## additional loss
        
        
        if not(gain_next == 'no'):
            perc_success = []
            prc = np.zeros((len(un2) ,len(un1)))
            tot_perc = np.zeros( (len(important_factors),1) )
            k1 = 0
            k2 = 0
            for u1 in un1:
                k2 = 0
                for u2 in un2:
                    idx = np.where( np.logical_and(important_factors[:,1] == u1, important_factors[:,0] == u2 ))
                    bool_point = [ gain_next[ii] for ii in idx[0]]
                    perc_success.append(sum(bool_point) /len(idx[0]))
                    
                    ps = sum(bool_point)/len(idx[0])
                    tot_perc[idx[0]]=ps
                    
                    prc[k2][k1] = ps
                    #print(prc)
                    k2 += 1
                k1 += 1
             
        
            #heat_dec = sns.heatmap( prc,  cmap='Greens', linewidths=0,  vmin=0, vmax=+0.2, annot=True, fmt=".0%", cbar=False) 
            fig2.add_trace(go.Heatmap(z = prc, x = un1, y = un2, colorscale = 'Greens', zmax = 1, zmin = 0 , 
                                      colorbar=dict(title="Robustness gain", titleside="right") ), row = 1, col = 1)
            #fig1.update_layout( title_text = "Capex ratio and LCOW",
                               #title_font_size = 20, showlegend=False)
            #print(prc)
            fig2.update_xaxes(title = DU_names[0][id_factors[1] ], tickvals = un1 ) 
            fig2.update_yaxes(title = DU_names[0][id_factors[0] ], tickvals = un2 )
        
    
    return [fig1, fig2]



def factor_ranking_over_time(DU_factors, obj_value, DU_names, desal_size):
    fig1 = make_subplots(rows=1, cols=1, subplot_titles=("Factor importance over time", ""))

    # create a gradient boosted classifier object
    gbc = GradientBoostingClassifier(n_estimators=300,
                                     learning_rate=0.01,
                                     max_depth=9)
    
    feature_importances = []
    for year in range(10,51,10):
        yy = int(( year/10 )- 1)
        print(yy)
        obj = obj_value[:,yy] 
        
    
        threshold = desal_size
        
        bool_obj = [o<threshold for o in obj]
        if sum(bool_obj)/len(bool_obj)>0.01:
        # fit the classifier
            gbc.fit(DU_factors, bool_obj)
            
            ##### Factor Ranking #####
             
            # Extract the feature importances
            feature_importances.append(deepcopy(list(gbc.feature_importances_)))
            
            
    FI = np.array(feature_importances).T.tolist()
    FI_r = np.flipud( np.array([FI[2], FI[7], FI[1],  FI[4] , FI[6],  FI[8], FI[3], FI[5], FI[0]])).tolist()
    names = np.flipud( np.array([DU_names[0][2],DU_names[0][7], DU_names[0][1],  DU_names[0][4], DU_names[0][6], DU_names[0][8], DU_names[0][3], DU_names[0][5], DU_names[0][0] ])).tolist()
    names = np.flipud(['Cachuma <br> allocation', 'Demand <br> scenario', 'Cachuma <br> carryover', 'Droughts', 'Capacity <br> factor', 'Deficit', 'SWP <br> allocation', 'Selling <br> agreements', 'Gibraltar <br> sedimentation' ])
    diction = {0: names[0], 1: names[1], 2: names[2], 3: names[3], 4: names[4], 5: names[5], 6: names[6], 7: names[7]}
    
    #Heatmap
    #fig1.add_trace(go.Heatmap(z = FI_r, x = list(range(0,51,10)), y = list(range(len(names))), colorscale = 'oranges',
     #                         colorbar=dict(title="Fraction of variability explained by factor", titleside="right")) , row = 1, col = 1)
     
    for fi, n in zip(FI_r, names):
        fig1.add_trace(go.Scatter( x = list(range(10,51,10)), y = fi, name = n,
                                  ) , row = 1, col = 1)
                              
                              
    fig1.update_xaxes(title = 'Time [years]', tickvals = list(range(0,51,10)),  ) 
    fig1.update_yaxes(title = 'Factors', tickvals = list(range(len(names))), ticktext = names) #, labelalias = diction )  
    fig1.update_layout(xaxis_range=[10,50])
    #fig1.write_image("factor_time.pdf")



    FI10y = [f[1] for f in FI_r]
    FI50y = [f[-1] for f in FI_r]
    
    fig1.show()















