#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Andreas Koundouros, University of Bonn
Date: 04.05.2023

This file contains custom functions for plotting results from the main file.
"""

###############################################################################
###############################################################################
# Import packages
import os
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from grgrlib import grbar3d
import matplotlib.cm as cm
import plotly.graph_objects as go

###############################################################################
###############################################################################
# Function for plotting the functions of a single policy
def plot_single_policy(model,
                       policy,
                       policy_name,
                       save_plots,
                       path,
                       borr_cutoff=None,
                       x_threshold=None):
    # Preliminaries
    a_grid = model['context']['a_grid'] # Get asset grid
    policy_arr = np.array(a_grid) # Initialise container for policies
    policy_columns = ['grid'] # Initialise name constainer
    
    # Loop through skill grid to get policy functions for each skill level
    for no_states in range(model['distributions']['dist']['skills']['n']):
        one_policy = model['steady_state']['decisions'][f'{policy}'][no_states]
        
        if policy_arr.size == 0:
            policy_arr = one_policy
        else:
            policy_arr = np.column_stack((policy_arr, one_policy))
            
        policy_columns.append(f'\u03B8_{no_states}')
    
    # Create data frames for plotting
    policy_df = pd.DataFrame(policy_arr, 
                             columns = policy_columns)
    
    # Cutoff if borrowing limit does not coincide with lowest asset grid point
    if borr_cutoff != None:
        policy_df.loc[policy_df['grid'] < borr_cutoff, :] = np.nan
    
    # Cutoff x axis if desired
    if x_threshold != None:
        policy_df.loc[policy_df['grid'] > x_threshold, :] = np.nan
        
    fig_policy =  px.line(policy_df, # Create plot
                          x = 'grid',
                          y = policy_columns,
                          title = '',
                          color_discrete_sequence=px.colors.qualitative.G10[:policy_df.shape[1]]) 
    fig_policy.update_layout(xaxis_title='Bonds Holdings Today', 
                             yaxis_title=f'{policy_name}',
                             plot_bgcolor = 'whitesmoke', 
                             font=dict(family="Times New Roman",
                                       size=20,
                                       color="black"),
                             margin=dict(l=15, r=15, t=50, b=5), 
                             legend_title='')
    fig_policy.show()
    
    if save_plots == True:
        path_plot = os.path.join(path, f'stst_policies_{policy}.svg')
        fig_policy.write_image(path_plot)
    

###############################################################################
###############################################################################
# Function for plotting selected policy functions
def plot_selected_policies(list_of_policies,
                            model,
                            save_plots,
                            path,
                            borr_cutoff=None,
                            x_threshold=None):
    # Loop through list of selected policies
    for sublist in list_of_policies:
        if len(sublist) == 2:
            policy = sublist[0] # extract policy
            policy_name = sublist[1] # extract policy name
            
            plot_single_policy(model, policy, policy_name, 
                               save_plots, path, 
                               borr_cutoff, x_threshold)
            
        else: 
            print('Error with the dimensions of the variable list.')
        
        
###############################################################################
###############################################################################
# Function for plotting steady state distributions
def make_stst_dist_plots(model, 
                         plot_dist_skills_and_assets=True, 
                         plot_dist_assets=True,
                         percent=100):
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
                                           percent*jnp.sum(distribution_skills_and_assets, 
                                                           axis = 0)])
    distribution_assets_df = pd.DataFrame(distribution_assets, 
                                          columns = ['grid', 'distribution'])
    
    if plot_dist_assets == True:
        fig_distr_assets = px.line(distribution_assets_df, # Create plot
                                  x = 'grid', 
                                  y = 'distribution',
                                  title='Bond Distribution',
                                  color_discrete_sequence=[px.colors.qualitative.G10[0]])
        fig_distr_assets.update_layout(xaxis_title='Bond Holdings', 
                                      yaxis_title='Share',
                                      plot_bgcolor = 'whitesmoke', 
                                      font=dict(family="Times New Roman",
                                                size=20,
                                                color="black"),
                                      margin=dict(l=15, r=15, t=50, b=5), 
                                      legend_title='')
        fig_distr_assets.show()

        
def shorten_asset_dist(model, 
                       x_threshold,
                       percent=100):
    # Get asset grid
    a_grid = model['context']['a_grid']
    
    # Distribution over skills and assets
    distribution_skills_and_assets = model['steady_state']['distributions'][0]
    
    # Distribution over assets
    distribution_assets = np.column_stack([a_grid, 
                                           percent*jnp.sum(distribution_skills_and_assets, 
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
                        x_threshold=None, 
                        y_threshold=None,
                        percent=100):
    # Create data frame for plotting depending on whether shortening is required
    if x_threshold != None:
        short_distribution_assets_df = shorten_asset_dist(model, x_threshold)
        
        a_grid = short_distribution_assets_df['grid']
        y = short_distribution_assets_df['distribution']
        
    elif x_threshold == None:
        a_grid = model['context']['a_grid']
        
        # Distribution over skills and assets
        distribution_skills_and_assets = model['steady_state']['distributions'][0]
        
        # Distribution over assets
        distribution_assets = np.column_stack([a_grid, 
                                               percent*jnp.sum(distribution_skills_and_assets, 
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
    
    fig = go.Figure() # make plot
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
                                        text=f'Pr[b={round(pos_a_grid[0],2)}] = {round(pos_y[0],2)}',
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
def plot_all(x_trans, 
             var_names, 
             horizon=30):
    for i,v in enumerate(var_names):
        plt.figure()
        plt.plot(x_trans[:horizon, i])
        plt.title(v)

###############################################################################
###############################################################################
# Function for plotting the transition of a single variable
def plot_single_transition(model, 
                           x_trans, 
                           variable, 
                           var_name, 
                           horizon, 
                           percent=100):   
    time = list(range(0, horizon, 1)) # Time vector
    
    variable = [variable] # Make variable a list
    var_index = [model['variables'].index(v) for v in variable] # Find variable index
    
    stst = x_trans[-1, var_index] # Find steady state (by definition, last
    # value in transition is the steady state value)
    
    if variable in ['R', 'Rn', 'Rr']:
        x_single_transition = np.column_stack([time, # Concatenate IRF and time vector
                                               (x_trans[:horizon,var_index] - stst)])
    elif variable in ['D_o_Y', 'lower_bound_a']:
        x_single_transition = np.column_stack([time, # Concatenate IRF and time vector
                                               x_trans[:horizon,var_index]])
    else:
        x_single_transition = np.column_stack([time, # Concatenate IRF and time vector
                                               percent*((x_trans[:horizon,var_index] - stst)/stst)])
        
    x_single_transition_df = pd.DataFrame(x_single_transition, # Turn into data frame
                                          columns = ['Quarters', f'{var_name}'])
    
    fig = px.line(x_single_transition_df, # make plot
                  x = 'Quarters',
                  y = f'{var_name}',
                  color_discrete_map={f'{var_name}': [px.colors.qualitative.G10[0]]})
    fig.update_layout(title='', # empty title
                       xaxis_title='Quarters', # x-axis labeling
                       yaxis_title=f'{var_name}', # y-axis labeling
                       legend=dict(orientation="h", # horizontal legend
                                   yanchor="bottom", y=1.02, 
                                   xanchor="right", x=1), 
                       legend_title=None, 
                       plot_bgcolor='whitesmoke', 
                       margin=dict(l=15, r=15, t=5, b=5),
                       font=dict(family="Times New Roman", # adjust font
                                 size=20,
                                 color="black"))
    fig.update_traces(line=dict(width=3))
    fig.show() # Show plot
    
###############################################################################
###############################################################################
# Function for plotting the transition of a list of selected variables
def plot_selected_transition(list_of_variables, 
                             model, 
                             x_trans, 
                             horizon, 
                             percent=100):
    # Loop through list of selected variables
    for sublist in list_of_variables:
        if len(sublist) == 2:
            variable = sublist[0] # extract variable
            variable_name = sublist[1] # extract variable name
        
            plot_single_transition(model, x_trans, # plot single transition of a 
                                   # given variable from the list
                                   variable, variable_name, 
                                   horizon, percent)
        elif len(sublist) == 4:
            pass # TO DO
        else: 
            print('Error with the dimensions of the variable list.')