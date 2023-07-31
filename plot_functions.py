#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Andreas Koundouros
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
import jax
import jax.numpy as jnp
from grgrlib import grbar3d
from grgrlib import figurator, grplot
import matplotlib.cm as cm
import plotly.graph_objects as go

# Import custom functions
from custom_functions import make_policy_df

###############################################################################
###############################################################################
# Function for plotting features of a steady state
def plot_full_stst(hank_model, 
                   settings,
                   shock_model_parameters,
                   save_results,
                   exact_path,
                   state,
                   borr_cutoff=None):
    # Plot 2D steady state distribution over assets
    bar_plot_asset_dist(hank_model, 
                        save_results,
                        exact_path,
                        state,
                        x_threshold=65, 
                        y_threshold=1.5)
    
    # Plot steady state policies over assets depending on skill
    policies_to_plot = [['a', 'Asset Holdings'],
                        ['c', 'Consumption']]
    if settings['Model'] == 'end_L':
        policies_to_plot.append(['n', 'Labour Supply'])
        
    plot_selected_policies(hank_model, 
                           policies_to_plot,
                           shock_model_parameters,
                           save_results,
                           exact_path,
                           state,
                           borr_cutoff,
                           x_threshold=None)
    
    # Plot MPCs over assets depending on skill
    plot_single_policy(hank_model, 'mpc', 'MPC',
                       shock_model_parameters,
                        save_results, exact_path, state,
                        borr_cutoff, x_threshold = 2, 
                        y_range=[0.,1.], 
                        group_mpc=True,
                        constraint_mpc=shock_model_parameters['terminal_borrowing_limit']) # shock_model_parameters['terminal_borrowing_limit']
    
    # Plot asset accumulation over assets depending on skill
    asset_acc = make_policy_df(hank_model, 'a', borr_cutoff)
    a_grid = hank_model['context']['a_grid']
    asset_acc[asset_acc.columns[1:]] = asset_acc[asset_acc.columns[1:]].sub(a_grid, axis='rows')
    plot_single_policy(hank_model, 'a', 'Asset Accumulation',
                        save_results, exact_path, state,
                        borr_cutoff, policy_df=asset_acc)
    
###############################################################################
###############################################################################
# Function for plotting the functions of a single policy
def plot_single_policy(hank_model, 
                       policy,
                       policy_name,
                       shock_model_parameters,
                       save_results,
                       exact_path,
                       state,
                       borr_cutoff=None,
                       x_threshold=None,
                       y_range=None,
                       group_mpc=False,
                       policy_df=None,
                       constraint_mpc=None):
    # Get policies 
    if not isinstance(policy_df, pd.DataFrame):
        policy_df = make_policy_df(hank_model, 
                                   policy, 
                                   borr_cutoff,
                                   x_threshold)
    
    fig_policy =  px.line(policy_df, # Create plot
                          x = 'grid',
                          y = policy_df.columns.tolist(),
                          title = '',
                          color_discrete_sequence=px.colors.qualitative.D3[:policy_df.shape[1]]) 
    fig_policy.update_layout(xaxis_title='Bond Holdings Today', 
                              yaxis_title=f'{policy_name}',
                              plot_bgcolor = 'whitesmoke', 
                              font=dict(family="Times New Roman",
                                        size=20,
                                        color="black"),
                              margin=dict(l=15, r=15, t=5, b=5), 
                              legend_title='') 
    fig_policy.update_traces(line=dict(width=3))
    
    if y_range != None:
        fig_policy.update_yaxes(range=y_range) # Fix range of y-axis
    
    if policy == 'mpc' and group_mpc == True:
        a_grid = hank_model['context']['a_grid']
        dist = hank_model['steady_state']['distributions'][0]
        mpc = hank_model['steady_state']['decisions']['mpc']
        
        borr_mpc = jnp.sum(jnp.where(a_grid<0,dist*mpc,0), axis=(0,1)) / jnp.sum(jnp.where(a_grid<0, dist, 0))
        
        lend_mpc = jnp.sum(jnp.where(a_grid>=0,dist*mpc,0), axis=(0,1)) / jnp.sum(jnp.where(a_grid>=0, dist, 0))
        
        fig_policy.update_layout(annotations=[dict(x=-0.8, 
                                                   y=0.7,
                                                   text=f'MPC(b<0) = {round(borr_mpc,2)}',
                                                   font=dict(family="Times New Roman",
                                                             size=20,
                                                             color="black")),
                                              dict(x=1, 
                                                   y=0.7,
                                                   text=f'MPC(b≥0) = {round(lend_mpc,2)}',
                                                   font=dict(family="Times New Roman",
                                                             size=20,
                                                             color="black"))])
        
        if constraint_mpc != None:
            const_mpc = jnp.sum(jnp.where(a_grid<constraint_mpc,dist*mpc,0), axis=(0,1)) / jnp.sum(jnp.where(a_grid<constraint_mpc, dist, 0))
            fig_policy.update_layout(annotations=[dict(x=-0.8, 
                                                       y=0.7,
                                                       text=f'MPC(b<0) = {round(borr_mpc,2)}',
                                                       font=dict(family="Times New Roman",
                                                                 size=20,
                                                                 color="black")),
                                                  dict(x=1, 
                                                       y=0.7,
                                                       text=f'MPC(b≥0) = {round(lend_mpc,2)}',
                                                       font=dict(family="Times New Roman",
                                                                 size=20,
                                                                 color="black")),
                                                  dict(x=a_grid[0]+0.8, 
                                                       y=0.9,
                                                       text=f'MPC(b< \u03C6 \') = {round(const_mpc,2)}',
                                                       font=dict(family="Times New Roman",
                                                                 size=20,
                                                                 color="black"))])
    
    fig_policy.show()
    
    # Save plot
    if save_results == True:
        path_plot = os.path.join(os.getcwd(),
                                 'Results',
                                 f'stst_policies_{policy}_{exact_path}_{state}.svg')
        fig_policy.write_image(path_plot)
    
###############################################################################
###############################################################################
# Function for plotting selected policy functions
def plot_selected_policies(hank_model, 
                           policies_to_plot,
                           shock_model_parameters,
                           save_results,
                           exact_path,
                           state,
                           borr_cutoff=None,
                           x_threshold=None):
    # Loop through list of selected policies
    for sublist in policies_to_plot:
        if len(sublist) == 2:
            policy = sublist[0] # extract policy
            policy_name = sublist[1] # extract policy name
            
            plot_single_policy(hank_model, 
                               policy,
                               policy_name,
                               shock_model_parameters,
                               save_results,
                               exact_path,
                               state,
                               borr_cutoff,
                               x_threshold,
                               policy_df=None)
            
        else: 
            print('Error with the dimensions of the variable list.')
        
###############################################################################
###############################################################################
def plot_stst_dist_3d(model,
                      percent=100):
    """Plot the steady state distribution in 3D.
    
    This function plots the steady state distribution of a given model over 
    skills and assets in three dimensions.

    Parameters
    ----------
    model                           :
    percent                         :

    Returns
    -------
    fig_dist_skills_and_assets      : 3D steady state distribution plot

    """
    
    # Get asset grid and the distribution over skills and assets
    a_grid = model['context']['a_grid']
    full_dist = model['steady_state']['distributions'][0]
    
    # 3D plot of distribution
    fig_dist_skills_and_assets, _ = grbar3d(percent*full_dist,
                                            xedges=jnp.arange(1, 
                                                              (len(full_dist)+1)), 
                                            yedges=a_grid, 
                                            figsize=(9,7), 
                                            depth=.5)
    
    # Label axes
    fig_dist_skills_and_assets.set_xlabel('Productivity')
    fig_dist_skills_and_assets.set_ylabel('Bond Holdings')
    fig_dist_skills_and_assets.set_zlabel('Share')
    
    # Adjust perspective
    fig_dist_skills_and_assets.view_init(azim=120)

###############################################################################
###############################################################################
def shorten_asset_dist(hank_model, 
                       x_threshold,
                       percent=100):
    # Get asset grid
    a_grid = hank_model['context']['a_grid']
    
    # Distribution over skills and assets
    distribution_skills_and_assets = hank_model['steady_state']['distributions'][0]
    
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
    
    # Return shortened distribution
    return short_asset_dist

###############################################################################
###############################################################################
def bar_plot_asset_dist(hank_model, 
                        save_results,
                        exact_path,
                        state,
                        x_threshold, 
                        y_threshold,
                        percent=100):
    # Create data frame for plotting depending on whether shortening is required
    if x_threshold != None:
        short_distribution_assets_df = shorten_asset_dist(hank_model, 
                                                          x_threshold)
        
        a_grid = short_distribution_assets_df['grid']
        y = short_distribution_assets_df['distribution']
        
    elif x_threshold == None:
        a_grid = hank_model['context']['a_grid']
        
        # Distribution over skills and assets
        distribution_skills_and_assets = hank_model['steady_state']['distributions'][0]
        
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
                      annotations=[dict(x=(pos_a_grid[0]+(bar_positions[i]/7)),
                                        y=y_threshold*(9/10),
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
                                                  color="black")),
                                   dict(x=(pos_a_grid.tail(1).iloc[0]-(bar_positions[i]/7)),
                                                     y=y_threshold*(1/8),
                                                     text=f'Pr[b≥{round(pos_a_grid.tail(1).iloc[0],2)}] = {round(pos_y.tail(1).iloc[0],2)}',
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
    
    # Save plot
    if save_results == True:
        path_plot = os.path.join(os.getcwd(),
                                 'Results',
                                 f'stst_dist_2d_bar_{exact_path}_{state}.svg')
        fig.write_image(path_plot)

###############################################################################
###############################################################################
# Function for plotting a comparison of two different steady states
def plot_compare_stst(hank_model_initial,
                      hank_model_terminal,
                      save_results,
                      exact_path,
                      phi,
                      x_threshold=None,
                      percent=100):
    ###########################################################################
    # Compare the steady state distribution over assets
    
    # Distribution over assets in the initial steady state
    a_grid_init = hank_model_initial['context']['a_grid']
    distribution_skills_and_assets_init = hank_model_initial['steady_state']['distributions'][0]
    distribution_assets_init = np.column_stack([a_grid_init, 
                                                percent*jnp.sum(distribution_skills_and_assets_init, 
                                                                axis = 0)])
    distribution_assets_init_df = pd.DataFrame(distribution_assets_init, 
                                               columns = ['grid', 'Initial'])
    
    # Distribution over assets in the terminal steady state
    a_grid_term = hank_model_terminal['context']['a_grid']
    distribution_skills_and_assets_term = hank_model_terminal['steady_state']['distributions'][0]
    distribution_assets_term = np.column_stack([a_grid_term, 
                                                percent*jnp.sum(distribution_skills_and_assets_term, 
                                                                axis = 0)])
    distribution_assets_term_df = pd.DataFrame(distribution_assets_term, 
                                               columns = ['grid', 'Terminal'])
    
    # Merge data frames
    dists_df = pd.merge(distribution_assets_init_df, 
                        distribution_assets_term_df,
                        on = 'grid', how = 'left')
    
    if x_threshold != None:
        dists_df.loc[dists_df['grid'] > x_threshold, :] = np.nan
    
    fig_dists = px.line(dists_df, # Create plot
                        x = 'grid', 
                        y = ['Initial', 'Terminal'],
                        title='', 
                        color_discrete_sequence=[px.colors.qualitative.D3[0], 
                                                 px.colors.qualitative.D3[1]])
    fig_dists.update_layout(xaxis_title='Bond Holdings',
                            yaxis_title='Share', 
                            plot_bgcolor = 'whitesmoke', 
                            font=dict(family="Times New Roman",
                                             size=20,
                                             color="black"), 
                            margin=dict(l=15, r=15, t=5, b=5), 
                            legend=dict(yanchor="top", y=0.99, 
                                        xanchor="right", x=0.99), 
                            legend_title=None)
    fig_dists.update_traces(line=dict(width=3))
    fig_dists.show()

    if save_results == True:
        path_plot = os.path.join(os.getcwd(),
                                 'Results',
                                 f'stst_dist_2d_comparison_{exact_path}.svg')
        fig_dists.write_image(path_plot)
    
    ###########################################################################
    # Compare steady state asset accumulation, averaged over productivity 
    # levels
    
    # Get stationary distribution over productivity levels 
    skills_stationary = hank_model_initial['context']['skills_stationary']
    
    # Asset accumulation in the initial steady state
    asset_acc_init = make_policy_df(hank_model_initial, 'a', 
                                    borr_cutoff=None, x_threshold=x_threshold)
    asset_acc_init[asset_acc_init.columns[1:]] = asset_acc_init[asset_acc_init.columns[1:]].sub(a_grid_init, 
                                                                                                axis='rows')
    asset_acc_long = asset_acc_init.iloc[:, 1:]
    df_array = asset_acc_long.to_numpy()
    dot_product_init = jnp.dot(df_array, skills_stationary)
    asset_acc_init['Initial'] = dot_product_init
    
    # Asset accumulation in the terminal steady state
    asset_acc_term = make_policy_df(hank_model_terminal, 'a', 
                                    borr_cutoff=phi, 
                                    x_threshold=x_threshold)
    asset_acc_term[asset_acc_term.columns[1:]] = asset_acc_term[asset_acc_term.columns[1:]].sub(a_grid_init, 
                                                                                                axis='rows')
    asset_acc_long = asset_acc_term.iloc[:, 1:]
    df_array = asset_acc_long.to_numpy()
    dot_product_term = jnp.dot(df_array, skills_stationary)
    asset_acc_term['Terminal'] = dot_product_term
    
    # Merge data frames
    acc_df = pd.merge(asset_acc_init, 
                      asset_acc_term,
                      on = 'grid', how = 'left')
    
    fig_acc = px.line(acc_df, # Create plot
                      x = 'grid', 
                      y = ['Initial', 'Terminal'],
                      title='', 
                      color_discrete_sequence=[px.colors.qualitative.D3[0], 
                                               px.colors.qualitative.D3[1]])
    fig_acc.update_layout(xaxis_title='Bond Holdings',
                            yaxis_title='Asset Accumulation', 
                            plot_bgcolor = 'whitesmoke', 
                            font=dict(family="Times New Roman",
                                             size=20,
                                             color="black"), 
                            margin=dict(l=15, r=15, t=5, b=5), 
                            legend=dict(yanchor="top", y=0.99, 
                                        xanchor="right", x=0.99), 
                            legend_title=None)
    fig_acc.update_traces(line=dict(width=3))
    fig_acc.show() # Show plot
    
    # Save plot
    if save_results == True:
        path_plot = os.path.join(os.getcwd(),
                                 'Results',
                                 f'stst_asset_acc_comparison_{exact_path}.svg')
        fig_acc.write_image(path_plot)
    
###############################################################################
###############################################################################
# Function for plotting the transition of all variables
def plot_all(x_trans, 
             var_names, 
             bunch=True,
             horizon=30):
    # The single plots can be bunched together to plots of four
    if bunch == True:
        grplot(x_trans[:horizon], labels=var_names)
    
    # Single full plots for each plot
    elif bunch == False:
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
                           unit,
                           horizon, 
                           save_results, 
                           exact_path,
                           percent=100):   
    time = list(range(0, horizon, 1)) # Time vector
    
    variable = [variable] # Make variable a list
    var_index = [model['variables'].index(v) for v in variable] # Find variable index
    
    if variable[0] in ['R', 'Rn', 'Rr', 'Rrminus']:
        x_single_transition = np.column_stack([time, # Concatenate IRF and time vector
                                               4*percent*(x_trans[:horizon,var_index] - 1.0)])
    elif variable[0] in ['Rbar']:
        x_single_transition = np.column_stack([time, # Concatenate IRF and time vector
                                               4*percent*x_trans[:horizon,var_index]])
    elif variable[0] in ['beta', 'D', 'DY', 'phi', 'gr_liquid', 'Top10C', 'Top10A', 'Top1C', 'Top1A', 'Top25C', 'Top25A', 'Bot25A', 'Bot25C']:
        x_single_transition = np.column_stack([time, # Concatenate IRF and time vector
                                               x_trans[:horizon,var_index]])
    else:
        stst = x_trans[0, var_index] # Find (initial) steady state
        x_single_transition = np.column_stack([time, # Concatenate IRF and time vector
                                               percent*((x_trans[:horizon,var_index] - stst)/stst)])
        
    x_single_transition_df = pd.DataFrame(x_single_transition, # Turn into data frame
                                          columns = ['Quarters', f'{var_name}'])
    
    fig = px.line(x_single_transition_df, # make plot
                  x = 'Quarters',
                  y = f'{var_name}',
                  color_discrete_sequence=[px.colors.qualitative.D3[0]])
    fig.update_layout(title=f'{var_name}', # title
                       xaxis_title='Quarters', # x-axis labeling
                       yaxis_title=f'{unit}', # y-axis labeling
                       legend=dict(orientation="h", # horizontal legend
                                   yanchor="bottom", y=1.02, 
                                   xanchor="right", x=1), 
                       legend_title=None, 
                       plot_bgcolor='whitesmoke', 
                       margin=dict(l=15, r=15, t=5, b=5),
                       font=dict(family="Times New Roman", # adjust font
                                 size=20,
                                 color="black"))
    if var_name != '':
        fig.update_layout(margin=dict(l=15, r=15, t=50, b=5))
    fig.update_traces(line=dict(width=3))
    fig.show() # Show plot
    
    # Save plot
    if save_results == True:
        path_plot = os.path.join(os.getcwd(),
                                 'Results',
                                 f'transition_{variable[0]}_{exact_path}.svg')
        fig.write_image(path_plot)
    
###############################################################################
###############################################################################
# Function for plotting the transition of two selected variables in the same 
# plot
def plot_double_transition(model, 
                           x_trans, 
                           variables, 
                           var_names, 
                           unit,
                           horizon, 
                           save_results, 
                           exact_path,
                           percent=100):
    time = list(range(0, horizon, 1)) # Time vector
    
    var_indices = [model['variables'].index(v) for v in variables] # Find variable index
    
    if variables[0] in ['R', 'Rn', 'Rr', 'Rminus'] and variables[1] in ['R', 'Rn', 'Rr', 'Rminus']:
        x_double_transition = np.column_stack([time, # Concatenate IRFs and time vector
                                               4*percent*(x_trans[:horizon,var_indices] - 1.0)])
    else:
        stst = x_trans[0, var_indices] # Find (initial) steady state
        x_double_transition = np.column_stack([time, # Concatenate IRF and time vector
                                               percent*((x_trans[:horizon,var_indices] - stst)/stst)])
        
    x_double_transition_df = pd.DataFrame(x_double_transition, # Turn into data frame
                                          columns = ['Quarters', 
                                                     f'{var_names[0]}', 
                                                     f'{var_names[1]}'])
    
    fig = px.line(x_double_transition_df, # make plot
                  x = 'Quarters',
                  y = [f'{var_names[0]}', f'{var_names[1]}'],
                  color_discrete_map={f'{var_names[0]}': px.colors.qualitative.D3[0],
                                      f'{var_names[1]}': px.colors.qualitative.D3[1]}).update_traces(selector={"name": f'{var_names[1]}'}, 
                                                                                                     line={"dash": "dash"})
    fig.update_layout(title='', # empty title
                       xaxis_title='Quarters', # x-axis labeling
                       yaxis_title=f'{unit}', # y-axis labeling
                       legend=dict(yanchor="top", y=0.99, 
                                   xanchor="right", x=0.99), 
                       legend_title=None, 
                       plot_bgcolor='whitesmoke', 
                       margin=dict(l=15, r=15, t=5, b=5),
                       font=dict(family="Times New Roman", # adjust font
                                 size=20,
                                 color="black"))
    fig.update_traces(line=dict(width=3))
    fig.show() # Show plot
    
    # Save plot
    if save_results == True:
        path_plot = os.path.join(os.getcwd(),
                                 'Results',
                                 f'transition_{variables[0]}_{variables[1]}_{exact_path}.svg')
        fig.write_image(path_plot)
    
###############################################################################
###############################################################################
# Function for plotting the transition of a list of selected variables
def plot_selected_transition(list_of_variables, 
                             model, 
                             x_trans, 
                             horizon, 
                             save_results, 
                             exact_path,
                             percent=100,
                             title=True):
    # Loop through list of selected variables
    for sublist in list_of_variables:
        if 'R' in sublist:
            percent=100
        
        if len(sublist) == 3:
            variable = sublist[0] # extract variable
            
            if title == True:
                variable_name = sublist[1] # extract variable name
            if title != True:
                variable_name = ''
            
            unit = sublist[2] # extract unit
        
            plot_single_transition(model, x_trans, # plot single transition of a 
                                   # given variable from the list
                                   variable, variable_name, unit,
                                   horizon, 
                                   save_results, 
                                   exact_path,
                                   percent)
        elif len(sublist) == 5:
            variables = [sublist[0], sublist[2]] # extract variable
            variable_names = [sublist[1], sublist[3]] # extract variable name
            unit = sublist[4] # extract unit
            
            plot_double_transition(model, x_trans, # plot double transition of a 
                                   # given variable from the list
                                   variables, variable_names, unit,
                                   horizon, 
                                   save_results, 
                                   exact_path,
                                   percent)
        else: 
            print('Error with the dimensions of the variable list.')

###############################################################################
###############################################################################
# Function for plotting the impact of the shock on household decisions over the 
# distribution
def plot_policy_on_impact_over_dist(hank_model_initial, 
                                    hank_model_terminal,
                                    x_transition,
                                    policy, 
                                    policy_name, 
                                    save_results,
                                    exact_path,
                                    x_threshold=None,
                                    borr_cutoff=True,
                                    borr_lim=None,
                                    percent=100):
    # Get disaggregated dynamics
    hank_model_initial_dist = hank_model_initial['steady_state']['distributions'].copy()
    dist_transition = hank_model_terminal.get_distributions(trajectory = x_transition,
                                                            init_dist = hank_model_initial_dist)
    
    # Get steady state policy, averaged over productivity
    stst_policy = hank_model_initial['steady_state']['decisions'][f'{policy}']
    av_stst_policy = jnp.dot(hank_model_terminal['context']['skills_stationary'], stst_policy)
    
    # Get policy in period 1, averaged over productivity
    period_1_policy = dist_transition[f'{policy}'][:, :, 0]
    av_period_1_policy = jnp.dot(hank_model_terminal['context']['skills_stationary'], period_1_policy)
    
    # Calculate differences in percent
    diff_stst = percent*(av_period_1_policy - av_stst_policy)/(abs(av_stst_policy))
    
    if policy == 'a':
        # Get a list of whether - at a given asset grid point - the sign of
        # the policy changed
        sign_change = jnp.not_equal(jnp.sign(av_period_1_policy), jnp.sign(av_stst_policy)) 
        
        # Loop over the percentage change in asset policies to leave out the 
        # cases where the sign of the policy changed (rendering percentage 
        # change calculation difficult)
        for ii in range(len(sign_change)):
            if sign_change[ii] == True:
                diff_stst = diff_stst.at[ii].set(np.nan)
    
    # Make data frame
    impact = {'grid': np.array(hank_model_terminal['context']['a_grid']),
              'impact': diff_stst}
    impact_df = pd.DataFrame(impact)
    
    # Cut off x axis at borrowing limit in period 1
    if borr_cutoff == True:
        cutoff = x_transition[:,hank_model_terminal['variables'].index('phi')][1] # Find borrowing limit which applies period 1
        impact_df.loc[impact_df['grid'] < (round(float(cutoff),8)), :] = np.nan

    # Cut off x axis at threshold
    if x_threshold != None:
        impact_df.loc[impact_df['grid'] > x_threshold, :] = np.nan

    # Plot
    fig_impact=px.line(impact_df, 
                       x = 'grid', 
                       y = 'impact', 
                       title='', 
                       color_discrete_sequence=[px.colors.qualitative.D3[0]])
    fig_impact.update_layout(xaxis_title='Bond Holdings', 
                              yaxis_title=f'{policy_name}',
                              plot_bgcolor = 'whitesmoke', 
                              font=dict(family="Times New Roman",
                                        size=20,
                                        color="black"),
                              margin=dict(l=15, r=15, t=5, b=5), 
                              legend_title='') 
    fig_impact.update_traces(line=dict(width=3))
    
    # Add terminal borrowing limit as vertical line
    if borr_lim != None:
        fig_impact.add_vline(x=borr_lim,
                             line_width=3, line_dash="dash", line_color="red")

    fig_impact.show()
    
    # Save plot
    if save_results == True:
        path_plot = os.path.join(os.getcwd(),
                                 'Results',
                                 f'distr_impact_{policy}_{exact_path}.svg')
        fig_impact.write_image(path_plot)
        
###############################################################################
###############################################################################
# Function for plotting the impact of the shock on all household decisions over 
# the distribution
def plot_policy_impact(hank_model_initial, hank_model_terminal, 
                       x_transition,
                       save_results, exact_path,
                       borr_lim,
                       x_threshold=150, borr_cutoff=True):
    
    # Change in consumption choice on impact over the distribution of assets
    plot_policy_on_impact_over_dist(hank_model_initial, hank_model_terminal, 
                                    x_transition, 'c', 'Consumption', 
                                    save_results, exact_path,
                                    x_threshold=x_threshold, borr_cutoff=borr_cutoff, 
                                    borr_lim=borr_lim)

    # Change in asset choice on impact over the distribution of assets
    plot_policy_on_impact_over_dist(hank_model_initial, hank_model_terminal, 
                                    x_transition, 'a', 'Assets', 
                                    save_results, exact_path,
                                    x_threshold=x_threshold, borr_cutoff=borr_cutoff, 
                                    borr_lim=borr_lim)

    # Change in labour choice on impact over the distribution of assets
    try: 
        plot_policy_on_impact_over_dist(hank_model_initial, hank_model_terminal, 
                                        x_transition, 'n', 'Labour Supply', 
                                        save_results, exact_path,
                                        x_threshold=x_threshold, borr_cutoff=borr_cutoff, 
                                        borr_lim=borr_lim)
    except KeyError:
        pass
        
            
###############################################################################
###############################################################################
# Function to compare the transitions of some selected variables 
def compare_selected_transition(variables_to_plot, 
                                x_1, 
                                x_2, 
                                horizon, 
                                legend,
                                save_results, 
                                comparison,
                                percent=100,
                                title=True):
    time = list(range(0, horizon, 1)) # Time vector
    
    # Loop through list of selected variables
    for sublist in variables_to_plot:
        variable = sublist[0] # extract variable

        if title == True:
            variable_name = sublist[1] # extract variable name
        if title != True:
            variable_name = ''
            
        unit = sublist[2] # extract unit
        
        if variable in ['R', 'Rn', 'Rr', 'Rminus']:
            x_double_transition = np.column_stack([time, # Concatenate IRFs and time vector
                                                   4*percent*(x_1[f'{variable}'][:horizon] - 1.0),
                                                   4*percent*(x_2[f'{variable}'][:horizon] - 1.0)])
        elif variable in ['Rbar']:
            x_double_transition = np.column_stack([time, # Concatenate IRF and time vector
                                                   4*percent*x_1[f'{variable}'][:horizon],
                                                   4*percent*x_2[f'{variable}'][:horizon]])
        elif variable in ['beta', 'D', 'DY', 'phi', 'gr_liquid']:
            x_double_transition = np.column_stack([time, # Concatenate IRF and time vector
                                                   x_1[f'{variable}'][:horizon],
                                                   x_2[f'{variable}'][:horizon]])
        else:
            stst_1 = x_1[f'{variable}'][0] # Find (initial) steady state
            stst_2 = x_2[f'{variable}'][0] # Find (initial) steady state
            x_double_transition = np.column_stack([time, # Concatenate IRF and time vector
                                                   percent*((x_1[f'{variable}'][:horizon] - stst_1)/stst_1),
                                                   percent*((x_2[f'{variable}'][:horizon] - stst_2)/stst_2)])
        
        x_double_transition_df = pd.DataFrame(x_double_transition, # Turn into data frame
                                              columns = ['Quarters', 
                                                         f'{legend[0]}',
                                                         f'{legend[1]}'])
        
        # Plot
        fig = px.line(x_double_transition_df,
                      x = 'Quarters',
                      y = [f'{legend[0]}', f'{legend[1]}'],
                      color_discrete_map={f'{legend[0]}': px.colors.qualitative.D3[7],
                                          f'{legend[1]}': px.colors.qualitative.D3[0]}).update_traces(selector={"name": f'{legend[0]}'}, 
                                                                                                         line={"dash": "dash"})
        
        if variable == 'C':
            fig.update_layout(title=f'{variable_name}', # empty title
                               xaxis_title='Quarters', # x-axis labeling
                               yaxis_title=f'{unit}', # y-axis labeling
                               legend=dict(yanchor="top", y=0.99, 
                                           xanchor="right", x=0.99), 
                               legend_title=None, 
                               plot_bgcolor='whitesmoke', 
                               margin=dict(l=15, r=15, t=5, b=5),
                               font=dict(family="Times New Roman", # adjust font
                                         size=20,
                                         color="black"))
        if variable != 'C':
            fig.update_layout(title='', # empty title
                               xaxis_title='Quarters', # x-axis labeling
                               yaxis_title=f'{variable_name}', # y-axis labeling
                               showlegend=False,
                               legend_title=None, 
                               plot_bgcolor='whitesmoke', 
                               margin=dict(l=15, r=15, t=5, b=5),
                               font=dict(family="Times New Roman", # adjust font
                                         size=20,
                                         color="black"))
        fig.update_traces(line=dict(width=3))
        fig.show() # Show plot
        
        # Save plot
        if save_results == True:
            key1 = comparison['transition_1']
            key2 = comparison['transition_2']
            path_plot = os.path.join(os.getcwd(),
                                     'Results',
                                     f'comparison_{variable}_{key1}_{key2}.svg')
            fig.write_image(path_plot)
            