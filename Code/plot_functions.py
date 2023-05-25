#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Andreas Koundouros, University of Bonn
Date: 04.05.2023

This file contains custom functions for plotting results from the main file.
"""

import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import econpizza
from grgrlib import grbar3d
import matplotlib.cm as cm
import plotly.graph_objects as go

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
                
            assets_policy_columns.append(f'\u03B8_{no_states}')
            
            if consumption_policy.size == 0:
                consumption_policy = consumption_column
            else:
                consumption_policy = np.column_stack((consumption_policy, 
                                                      consumption_column))
                
            consumption_policy_columns.append(f'\u03B8_{no_states}')
            
            if labour_policy.size == 0:
                labour_policy = labour_column
            else:
                labour_policy = np.column_stack((labour_policy, 
                                                 labour_column))
                
            labour_policy_columns.append(f'\u03B8_{no_states}')
        
        # Create data frames for plotting
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
        
        # Plot asset policies 
        if plot_asset_policy == True:
            fig_assets =  px.line(assets_policy_df, # Create plot
                                  x = 'grid',
                                  y = assets_policy_columns,
                                  title='Asset Policy',
                                  color_discrete_sequence=px.colors.qualitative.Plotly[:assets_policy_df.shape[1]]) 
            fig_assets.update_layout(xaxis_title='Bonds Holdings Today', 
                                     yaxis_title='Bonds Holdings Tomorrow',
                                     plot_bgcolor = 'whitesmoke', 
                                     font=dict(family="Times New Roman",
                                               size=20,
                                               color="black"),
                                     margin=dict(l=15, r=15, t=50, b=5), 
                                     legend_title='')
            fig_assets.show()
            
        # Plot consumption policies
        if plot_consumption_policy == True:
            fig_consumption =  px.line(consumption_policy_df, # Create plot
                                       x = 'grid',
                                       y = consumption_policy_columns,
                                       title='Consumption Policy',
                                       color_discrete_sequence=px.colors.qualitative.Plotly[:consumption_policy_df.shape[1]]) 
            fig_consumption.update_layout(xaxis_title='Bonds Holdings', 
                                          yaxis_title='Consumption',
                                          plot_bgcolor = 'whitesmoke', 
                                          font=dict(family="Times New Roman",
                                                    size=20,
                                                    color="black"),
                                          margin=dict(l=15, r=15, t=50, b=5), 
                                          legend_title='')
            fig_consumption.show()
            
        # Plot labour supply policies
        if 'n' in model['steady_state']['decisions'].keys() and plot_labour_policy == True:
            fig_labour =  px.line(labour_policy_df, # Create plot
                                       x = 'grid',
                                       y = labour_policy_columns,
                                       title='Labour Supply Policy',
                                       color_discrete_sequence=px.colors.qualitative.Plotly[:labour_policy_df.shape[1]]) 
            fig_labour.update_layout(xaxis_title='Bonds Holdings', 
                                     yaxis_title='Labour Supply',
                                     plot_bgcolor = 'whitesmoke', 
                                     font=dict(family="Times New Roman",
                                               size=20,
                                               color="black"),
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
                         plot_dist_assets=True):
    if type(model) == econpizza.__init__.PizzaModel:
        
        # Get asset grid
        a_grid = model['context']['a_grid']
        
        # Distribution over skills and assets
        distribution_skills_and_assets = model['steady_state']['distributions'][0]
        
        if plot_dist_skills_and_assets == True:
            fig_dist_skills_and_assets, _ = grbar3d(100*distribution_skills_and_assets, # Create plot
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
            fig_distr_assets = px.line(distribution_assets_df, # Create plot
                                      x = 'grid', 
                                      y = 'distribution',
                                      title='Bond Distribution',
                                      color_discrete_sequence=[px.colors.qualitative.Plotly[0]])
            fig_distr_assets.update_layout(xaxis_title='Bond Holdings', 
                                          yaxis_title='Share',
                                          plot_bgcolor = 'whitesmoke', 
                                          font=dict(family="Times New Roman",
                                                    size=20,
                                                    color="black"),
                                          margin=dict(l=15, r=15, t=50, b=5), 
                                          legend_title='')
            fig_distr_assets.show()
        
    else:
        print('Error: Input must be of type PizzaModel.')
        
def shorten_asset_dist(model, x_threshold):
    # Get asset grid
    a_grid = model['context']['a_grid']
    
    # Distribution over skills and assets
    distribution_skills_and_assets = model['steady_state']['distributions'][0]
    
    # Distribution over assets
    distribution_assets = np.column_stack([a_grid, 
                                           100*jnp.sum(distribution_skills_and_assets, 
                                                       axis = 0)])
    distribution_assets_df = pd.DataFrame(distribution_assets, 
                                          columns = ['grid', 'distribution'])

    # Filter the data frame based on the threshold
    filtered_df = distribution_assets_df[distribution_assets_df['grid'] < x_threshold]

    # Calculate the sum of shares for grid points above or equal to the threshold
    sum_share = distribution_assets_df[distribution_assets_df['grid'] >= x_threshold]['distribution'].sum()

    # Create a new row with the threshold and the sum of shares
    threshold_row = pd.DataFrame({'grid': [x_threshold], 'distribution': [sum_share]})

    # Concatenate the filtered data frame with the threshold row
    short_asset_dist = pd.concat([filtered_df, threshold_row])

    # Reset the index of the new data frame
    short_asset_dist.reset_index(drop=True, inplace=True)
    
    return short_asset_dist

def bar_plot_asset_dist(model, 
                        shorten=False, x_threshold = None, 
                        y_threshold = None):
    # Create data frame for plotting depending on whether shortening is required
    if shorten == True and x_threshold == None:
        print('Threshold for shortening required.')
        
    elif shorten == True and x_threshold != None:
        short_distribution_assets_df = shorten_asset_dist(model, x_threshold)
        
        a_grid = short_distribution_assets_df['grid']
        y = short_distribution_assets_df['distribution']
        
    elif shorten == False:
        a_grid = model['context']['a_grid']
        
        # Distribution over skills and assets
        distribution_skills_and_assets = model['steady_state']['distributions'][0]
        
        # Distribution over assets
        distribution_assets = np.column_stack([a_grid, 
                                               100*jnp.sum(distribution_skills_and_assets, 
                                                           axis = 0)])
        distribution_assets_df = pd.DataFrame(distribution_assets, 
                                              columns = ['grid', 'distribution'])
        
        a_grid = distribution_assets_df['grid']
        y = distribution_assets_df['distribution']

    # Calculate the widths of the bars based on the intervals between grid points
    bar_widths = np.diff(a_grid)
    bar_widths = np.append(bar_widths, bar_widths[-1])  # Add last width for consistency

    # Calculate the positions of the bars
    bar_positions = np.array(a_grid) - np.array(bar_widths) / 2
    
    # Generate a color sequence using a colormap
    cmap = cm.get_cmap('twilight_shifted')  # Choose a colormap
    colours = [cmap(i / len(a_grid)) for i in range(len(a_grid))]
    
    # Get lowest grid point with positive frequency 
    pos_a_grid = a_grid[y>0]
    pos_a_grid.reset_index(drop=True, inplace=True)
    
    # Get the frequency of that grid point
    pos_y = y[y>0]
    pos_y.reset_index(drop=True, inplace=True)
    
    fig = go.Figure() # Create plot
    for i in range(len(a_grid)):
        if y[i] > y_threshold: # replace value over y-axis threshold by threshold
            fig.add_trace(go.Bar(
                x=[bar_positions[i]],
                y=np.array(y_threshold),
                width=bar_widths[i],
                marker=dict(color=colours[i])))
        else:
            fig.add_trace(go.Bar(
                x=[bar_positions[i]],
                y=[y[i]],
                width=bar_widths[i],
                marker=dict(color=colours[i])))
        
    fig.update_layout(xaxis_title='Bond Holdings', 
                      yaxis_title='Share',
                      plot_bgcolor='whitesmoke', 
                      font=dict(family="Times New Roman",
                                size=20,
                                color="black"),
                      margin=dict(l=15, r=15, t=5, b=5), 
                      legend_title='',
                      showlegend=False,
                      annotations=[dict(x=(pos_a_grid[0]+4), 
                                        y=y_threshold-1,
                                        text=f'Pr(b={round(pos_a_grid[0],2)}) = {round(pos_y[0],2)}',
                                        showarrow=False,
                                        arrowhead=1,
                                        arrowcolor='black',
                                        arrowsize=2,
                                        arrowwidth=1,
                                        ax=210,
                                        ay=0,
                                        font=dict(family="Times New Roman",
                                                  size=20,
                                                  color="black"))])
    fig.update_yaxes(range=[0., y_threshold]) # Fix range of y-axis
    fig.show()
    
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
                       legend=dict(orientation="h", # For horizontal legend
                                   yanchor="bottom", y=1.02, 
                                   xanchor="right", x=1), 
                       legend_title=None, 
                       plot_bgcolor='whitesmoke', 
                       margin=dict(l=15, r=15, t=5, b=5),
                       font=dict(family="Times New Roman",
                                 size=20,
                                 color="black"))
    fig.update_traces(line=dict(width=3))
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