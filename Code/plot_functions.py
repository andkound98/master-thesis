#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Andreas Koundouros, University of Bonn
Date: 04.05.2023
"""

import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import econpizza
from grgrlib import grbar3d

from custom_functions import find_closest_grid_point

###############################################################################
###############################################################################
# Function for plotting steady state policies
def make_stst_policiy_plots(model, 
                            plot_asset_policy=True, 
                            plot_consumption_policy=True,
                            plot_labour_policy=True,
                            cutoff=False):
    if type(model) == econpizza.__init__.PizzaModel:
        # Get asset grid
        a_grid = model['context']['a_grid']
        
        # Initialise arrays 
        assets_policy = np.array(a_grid)
        assets_policy_columns = ['grid']
        consumption_policy = np.array(a_grid)
        consumption_policy_columns = ['grid']
        labour_policy = np.array(a_grid)
        labour_policy_columns = ['grid']
        
        # Loop through skills_grid to get policy functions
        for no_states in range(model['distributions']['dist']['skills']['n']):
            asset_column = model['steady_state']['decisions']['a'][no_states]
            consumption_column = model['steady_state']['decisions']['c'][no_states]
            labour_column = model['steady_state']['decisions']['n'][no_states]
            
            if assets_policy.size == 0:
                assets_policy = asset_column
            else:
                assets_policy = np.column_stack((assets_policy, asset_column))
                
            assets_policy_columns.append(f'skills_{no_states}')
            
            if consumption_policy.size == 0:
                consumption_policy = consumption_column
            else:
                consumption_policy = np.column_stack((consumption_policy, 
                                                      consumption_column))
                
            consumption_policy_columns.append(f'skills_{no_states}')
            
            if labour_policy.size == 0:
                labour_policy = labour_column
            else:
                labour_policy = np.column_stack((labour_policy, 
                                                      labour_column))
                
            labour_policy_columns.append(f'skills_{no_states}')
        
        # Create handy data frames for plotting
        assets_policy_df = pd.DataFrame(assets_policy, 
                                        columns = assets_policy_columns)
        consumption_policy_df = pd.DataFrame(consumption_policy, 
                                             columns = consumption_policy_columns)
        labour_policy_df = pd.DataFrame(labour_policy, 
                                             columns = labour_policy_columns)
        
        # Cut data frames short at borrowing limit if specified
        if cutoff == True:
            cutoff_value, _ = find_closest_grid_point(model['steady_state']['fixed_values']['lower_bound_a'], 
                                                   a_grid)
            
            assets_policy_df.loc[assets_policy_df['grid'] < cutoff_value, :] = np.nan
            consumption_policy_df.loc[consumption_policy_df['grid'] < cutoff_value, :] = np.nan
            labour_policy_df.loc[labour_policy_df['grid'] < cutoff_value, :] = np.nan
        
        if plot_asset_policy == True:
            # Plot asset policies 
            fig_assets =  px.line(assets_policy_df,
                                  x = 'grid',
                                  y = assets_policy_columns,
                                  title='Asset Policy',
                                  #markers = True,
                                  color_discrete_sequence=px.colors.qualitative.Plotly[:assets_policy_df.shape[1]]) 
            fig_assets.update_layout(xaxis_title='Bonds Holdings Today', 
                                     yaxis_title='Bonds Holdings Tomorrow',
                                     plot_bgcolor = 'whitesmoke', 
                                     font=dict(size=20), 
                                     margin=dict(l=15, r=15, t=50, b=5), 
                                     legend_title='')
            fig_assets.show()

        if plot_consumption_policy == True:
            # Plot consumption policies
            fig_consumption =  px.line(consumption_policy_df,
                                       x = 'grid',
                                       y = consumption_policy_columns,
                                       title='Consumption Policy',
                                       #markers = True,
                                       color_discrete_sequence=px.colors.qualitative.Plotly[:consumption_policy_df.shape[1]]) 
            fig_consumption.update_layout(xaxis_title='Bonds Holdings', 
                                          yaxis_title='Consumption',
                                          plot_bgcolor = 'whitesmoke', 
                                          font=dict(size=20), 
                                          margin=dict(l=15, r=15, t=50, b=5), 
                                          legend_title='')
            fig_consumption.show()
            
        if plot_labour_policy == True:
            # Plot labour supply policies
            fig_labour =  px.line(labour_policy_df,
                                       x = 'grid',
                                       y = labour_policy_columns,
                                       title='Labour Supply Policy',
                                       #markers = True,
                                       color_discrete_sequence=px.colors.qualitative.Plotly[:labour_policy_df.shape[1]]) 
            fig_labour.update_layout(xaxis_title='Bonds Holdings', 
                                          yaxis_title='Labour Supply',
                                          plot_bgcolor = 'whitesmoke', 
                                          font=dict(size=20), 
                                          margin=dict(l=15, r=15, t=50, b=5), 
                                          legend_title='')
            fig_labour.show()
        
    else:
        print('Error: Input must be of type PizzaModel.')
        
###############################################################################
###############################################################################
# Function for plotting steady state distributions
def make_stst_dist_plots(model, 
                         plot_dist_skills_and_assets=True, 
                         plot_dist_assets=True,
                         plot_dist_mpc=True,
                         plot_dist_n=True):
    if type(model) == econpizza.__init__.PizzaModel:
        
        # Get asset grid
        a_grid = model['context']['a_grid']
        
        # Distribution over skills and assets
        distribution_skills_and_assets = model['steady_state']['distributions'][0]
        
        if plot_dist_skills_and_assets == True:
            fig_dist_skills_and_assets, _ = grbar3d(100*distribution_skills_and_assets, 
                                                    xedges=jnp.arange(1, (len(distribution_skills_and_assets)+1)), 
                                                    yedges=a_grid, 
                                                    figsize=(9,7), 
                                                    depth=.5) # create 3D plot
            fig_dist_skills_and_assets.set_xlabel('Productivity')
            fig_dist_skills_and_assets.set_ylabel('Bond Holdings')
            fig_dist_skills_and_assets.set_zlabel('Share')
            fig_dist_skills_and_assets.view_init(azim=120) # rotate

        # Distribution over assets
        distribution_assets = np.column_stack([a_grid, 
                                               100*jnp.sum(distribution_skills_and_assets, 
                                                           axis = 0)])
        distribution_assets_df = pd.DataFrame(distribution_assets, 
                                              columns = ['grid', 'distribution'])
        
        if plot_dist_assets == True:
            fig_distr_assets = px.line(distribution_assets_df, 
                                      x = 'grid', 
                                      y = 'distribution',
                                      title='Bond Distribution',
                                      color_discrete_sequence=[px.colors.qualitative.Plotly[0]])
            fig_distr_assets.update_layout(xaxis_title='Bond Holdings', 
                                          yaxis_title='Share',
                                          plot_bgcolor = 'whitesmoke', 
                                          font=dict(size=20), 
                                          margin=dict(l=15, r=15, t=50, b=5), 
                                          legend_title='')
            fig_distr_assets.show()
            
            # fig_distr_assets = px.bar(distribution_assets_df, 
            #                           x = 'grid', 
            #                           y = 'distribution',
            #                           title='Bond Distribution',
            #                           color_discrete_sequence=[px.colors.qualitative.Plotly[0]])
            # fig_distr_assets.update_layout(xaxis_title='Bond Holdings', 
            #                               yaxis_title='Share',
            #                               plot_bgcolor = 'whitesmoke', 
            #                               font=dict(size=20), 
            #                               margin=dict(l=15, r=15, t=50, b=5), 
            #                               legend_title='')
            # fig_distr_assets.update_yaxes(range=[0., 1.])
            # fig_distr_assets.update_xaxes(range=[-2., 20.])
            # fig_distr_assets.show()

        # Distribution of MPCs over skills and assets
        distribution_mpc = model['steady_state']['decisions']['mpc']
        
        if plot_dist_mpc == True:
            fig_dist_mpc, _ = grbar3d(100*distribution_mpc, 
                                      xedges=jnp.arange(1, (len(distribution_skills_and_assets)+1)), 
                                      yedges=a_grid, 
                                      figsize=(9,7), 
                                      depth=.5) # create 3D plot
            fig_dist_mpc.set_xlabel('Productivity')
            fig_dist_mpc.set_ylabel('Bond Holdings')
            fig_dist_mpc.set_zlabel('MPC')
            fig_dist_mpc.view_init(azim=120) # rotate
            
        # Distribution of labour supply over skills and assets
        distribution_n = model['steady_state']['decisions']['n']
        
        if plot_dist_n == True:
            fig_dist_n, _ = grbar3d(distribution_n, 
                                      xedges=jnp.arange(1, (len(distribution_skills_and_assets)+1)), 
                                      yedges=a_grid, 
                                      figsize=(9,7), 
                                      depth=.5) # create 3D plot
            fig_dist_n.set_xlabel('Productivity')
            fig_dist_n.set_ylabel('Bond Holdings')
            fig_dist_n.set_zlabel('Labour Supply')
            fig_dist_n.view_init(azim=120) # rotate
        
    else:
        print('Error: Input must be of type PizzaModel.')
    
###############################################################################
###############################################################################
# Function for plotting the transition of all variables
def plot_all(x_trans, var_names, horizon=30):
    for i,v in enumerate(var_names):
        plt.figure()
        plt.plot(x_trans[:horizon, i])
        plt.title(v)

###############################################################################
###############################################################################
# Function for plotting the transition of a single variable
def plot_single_transition(model, x_trans, variable, var_name, horizon, percent=100):   
    time = list(range(0, horizon, 1)) # Time vector
    
    variable = [variable] # Make variable a list
    var_index = [model['variables'].index(v) for v in variable] # Find variable index
    
    stst = x_trans[-1, var_index] # Find steady state
    
    variable_interest_rate = ['R', 'Rn', 'Rstar', 'Rr']
    if variable in variable_interest_rate:
        x_single_transition = np.column_stack([time, # Concatenate IRF and time vector
                                               (x_trans[:horizon,var_index] - stst)])
    else:
        x_single_transition = np.column_stack([time, # Concatenate IRF and time vector
                                               percent*((x_trans[:horizon,var_index] - stst)/stst)])
        
    x_single_transition_df = pd.DataFrame(x_single_transition, # Turn into data frame
                                          columns = ['Quarters', f'{var_name}'])
    
    fig = px.line(x_single_transition_df, # Create plot
                  x = 'Quarters',
                  y = f'{var_name}',
                  color_discrete_map={f'{var_name}': '#636EFA'})
    fig.update_layout(title='', # Empty title
                       xaxis_title='Quarters', # x-axis labeling
                       yaxis_title=f'{var_name}', # y-axis labeling
                       font=dict(size=20),
                       legend=dict(orientation="h", # For horizontal legend
                                   yanchor="bottom", y=1.02, xanchor="right", x=1), 
                       legend_title=None, plot_bgcolor = 'whitesmoke', 
                       margin=dict(l=15, r=15, t=5, b=5))
    fig.update_traces(line=dict(width=2))
    fig.show() # Show plot
    
###############################################################################
###############################################################################
# Function for plotting the transition of a list of selected variables
def plot_selected_transition(list_of_variables, 
                             model, x_trans, horizon, percent=100):
    for sublist in list_of_variables:
        variable = sublist[0]
        variable_name = sublist[1]
        
        plot_single_transition(model, x_trans, 
                               variable, variable_name, 
                               horizon, percent)