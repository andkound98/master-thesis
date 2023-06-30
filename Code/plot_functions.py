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
import jax.numpy as jnp
from grgrlib import grbar3d
import matplotlib.cm as cm
import plotly.graph_objects as go

# Import custom functions
from custom_functions import make_policy_df

###############################################################################
###############################################################################
# Function for plotting features of a steady state
def plot_full_stst(model,
                   model_type,
                   save_results,
                   full_path_results,
                   state,
                   borr_limit=None):
    # Plot 3D steady state distribution over skills and assets
    plot_stst_dist_3d(model, 
                      save_results,
                      full_path_results,
                      state)
    
    # Plot 2D steady state distribution over assets
    bar_plot_asset_dist(model, 
                        save_results,
                        full_path_results,
                        state,
                        x_threshold = 30, 
                        y_threshold = 8)
    
    # Plot steady state policies over assets depending on skill
    policies_to_plot = [['a', 'Asset Holdings'],
                        ['c', 'Consumption']]
    if model_type == 'End_labour':
        policies_to_plot.append(['n', 'Labour Supply'])
        
    plot_selected_policies(model, policies_to_plot,
                           save_results, full_path_results, state,
                           borr_cutoff=borr_limit, x_threshold=None)
    
    # Plot MPCs over assets depending on skill
    plot_single_policy(model, 'mpc', 'MPC',
                       save_results, full_path_results, state,
                       borr_cutoff=borr_limit, x_threshold = 3, 
                       y_range=[0.,1.])
    
    # Plot asset accumulation over assets depending on skill
    asset_acc = make_policy_df(model, 'a', borr_cutoff=borr_limit)
    a_grid = model['context']['a_grid']
    asset_acc[asset_acc.columns[1:]] = asset_acc[asset_acc.columns[1:]].sub(a_grid, axis='rows')
    plot_single_policy(model, 'a', 'Asset Accumulation',
                       save_results, full_path_results, state,
                       borr_cutoff=borr_limit, policy_df=asset_acc)
    
###############################################################################
###############################################################################
# Function for plotting the functions of a single policy
def plot_single_policy(model,
                       policy,
                       policy_name,
                       save,
                       path,
                       state,
                       borr_cutoff=None,
                       x_threshold=None,
                       y_range=None,
                       policy_df=None):
    # Get policies 
    if not isinstance(policy_df, pd.DataFrame):
        policy_df = make_policy_df(model, policy, borr_cutoff, x_threshold)
        
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
    if y_range != None:
        fig_policy.update_yaxes(range=y_range) # Fix range of y-axis
    fig_policy.show()
    
    if save == True:
        path_plot = os.path.join(path, f'stst_policies_{policy}_{state}.svg')
        fig_policy.write_image(path_plot)
    

###############################################################################
###############################################################################
# Function for plotting selected policy functions
def plot_selected_policies(model,
                           list_of_policies,
                           save_plots,
                           path,
                           state,
                           borr_cutoff=None,
                           x_threshold=None):
    # Loop through list of selected policies
    for sublist in list_of_policies:
        if len(sublist) == 2:
            policy = sublist[0] # extract policy
            policy_name = sublist[1] # extract policy name
            
            plot_single_policy(model, policy, policy_name, 
                               save_plots, path, state,
                               borr_cutoff, x_threshold, policy_df=None)
            
        else: 
            print('Error with the dimensions of the variable list.')
        
        
###############################################################################
###############################################################################
def plot_stst_dist_3d(model,
                      save_results,
                      full_path_results,
                      state,
                      percent=100):
    """Plot the steady state distribution in 3D.
    
    This function plots the steady state distribution of a given model over 
    skills and assets in three dimensions.

    Parameters
    ----------
    model                           :
    save_results                    :
    full_path_results               :
    state                           :
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
    fig_dist_skills_and_assets.set_xlabel('Productivity')
    fig_dist_skills_and_assets.set_ylabel('Bond Holdings')
    fig_dist_skills_and_assets.set_zlabel('Share')
    fig_dist_skills_and_assets.view_init(azim=120)
        
    if save_results == True:
        path_plot = os.path.join(full_path_results, 'stst_dist_3d_{state}.png')
        #fig_dist_skills_and_assets.savefig(path_plot) # TO DO

###############################################################################
###############################################################################
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

###############################################################################
###############################################################################
def bar_plot_asset_dist(model, 
                        save_results,
                        full_path_results,
                        state,
                        x_threshold, 
                        y_threshold,
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
    
    if save_results == True:
        path_plot = os.path.join(full_path_results, 
                                 f'stst_dist_2d_bar_{state}.svg')
        fig.write_image(path_plot)

###############################################################################
###############################################################################
def plot_compare_stst(hank_model_initial,
                      hank_model_terminal,
                      save_results,
                      full_path_results,
                      borr_limit,
                      x_threshold=None,
                      percent=100):
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
    fig_dists.show()

    if save_results == True:
        path_plot = os.path.join(full_path_results, 'stst_dist_2d_comp.svg')
        fig_dists.write_image(path_plot)
    
    # Compare steady state asset accumulation, averaged over productivity levels
    skills_stationary = hank_model_initial['context']['skills_stationary']
    
    # Asset accumulation in the initial steady state
    asset_acc_init = make_policy_df(hank_model_initial, 'a', 
                                    borr_cutoff=None, x_threshold=x_threshold)
    asset_acc_init[asset_acc_init.columns[1:]] = asset_acc_init[asset_acc_init.columns[1:]].sub(a_grid_init, axis='rows')
    asset_acc_long = asset_acc_init.iloc[:, 1:]
    df_array = asset_acc_long.to_numpy()
    dot_product_init = jnp.dot(df_array, skills_stationary)
    asset_acc_init['Initial'] = dot_product_init
    
    # Asset accumulation in the terminal steady state
    asset_acc_term = make_policy_df(hank_model_terminal, 'a', 
                                    borr_cutoff=borr_limit, 
                                    x_threshold=x_threshold)
    asset_acc_term[asset_acc_term.columns[1:]] = asset_acc_term[asset_acc_term.columns[1:]].sub(a_grid_init, axis='rows')
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
    fig_acc.show()
    
    if save_results == True:
        path_plot = os.path.join(full_path_results, 'stst_asset_acc_comp.svg')
        fig_acc.write_image(path_plot)
    
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
                           save_plots, 
                           path,
                           percent=100):   
    time = list(range(0, horizon, 1)) # Time vector
    
    variable = [variable] # Make variable a list
    var_index = [model['variables'].index(v) for v in variable] # Find variable index
    
    if variable[0] in ['R', 'Rn', 'Rr', 'Rrminus']:
        x_single_transition = np.column_stack([time, # Concatenate IRF and time vector
                                               (x_trans[:horizon,var_index] - 1.0)])
    elif variable[0] in ['D', 'DY', 'borr_limit', 'Rbar', 'gr_liquid']:
        x_single_transition = np.column_stack([time, # Concatenate IRF and time vector
                                               x_trans[:horizon,var_index]])
    else:
        stst = x_trans[0, var_index] # Find steady state (by definition, last
        # value in transition is the steady state value)
        x_single_transition = np.column_stack([time, # Concatenate IRF and time vector
                                               percent*((x_trans[:horizon,var_index] - stst)/stst)])
        
    x_single_transition_df = pd.DataFrame(x_single_transition, # Turn into data frame
                                          columns = ['Quarters', f'{var_name}'])
    
    fig = px.line(x_single_transition_df, # make plot
                  x = 'Quarters',
                  y = f'{var_name}',
                  color_discrete_map={f'{var_name}': [px.colors.qualitative.D3[0]]})
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
    
    if save_plots == True:
        path_plot = os.path.join(path, f'transition_{variable[0]}.svg')
        fig.write_image(path_plot)
    
###############################################################################
###############################################################################
def plot_double_transition(model, 
                           x_trans, 
                           variables, 
                           var_names, 
                           horizon, 
                           save_plots, 
                           path,
                           percent=100):
    time = list(range(0, horizon, 1)) # Time vector
    
    var_indices = [model['variables'].index(v) for v in variables] # Find variable index
    
    if variables[0] in ['R', 'Rn', 'Rr', 'Rminus'] and variables[1] in ['R', 'Rn', 'Rr', 'Rminus']:
        x_double_transition = np.column_stack([time, # Concatenate IRFs and time vector
                                               percent*(x_trans[:horizon,var_indices] - 1.0)])
    else:
        stst = x_trans[-1, var_indices] # Find steady state (by definition, last
        # value in transition is the steady state value)
        x_double_transition = np.column_stack([time, # Concatenate IRF and time vector
                                               percent*((x_trans[:horizon,var_indices] - stst)/stst)])
        
    x_double_transition_df = pd.DataFrame(x_double_transition, # Turn into data frame
                                          columns = ['Quarters', f'{var_names[0]}', f'{var_names[1]}'])
    
    fig = px.line(x_double_transition_df, # make plot
                  x = 'Quarters',
                  y = [f'{var_names[0]}', f'{var_names[1]}'],
                  color_discrete_map={f'{var_names[0]}': px.colors.qualitative.D3[0],
                                      f'{var_names[1]}': px.colors.qualitative.D3[1]}).update_traces(selector={"name": f'{var_names[1]}'}, 
                                                                                                      line={"dash": "dash"})
    fig.update_layout(title='', # empty title
                       xaxis_title='Quarters', # x-axis labeling
                       yaxis_title='', # y-axis labeling
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
    
    if save_plots == True:
        path_plot = os.path.join(path, 
                                 f'transition_{variables[0]}_{variables[1]}.svg')
        fig.write_image(path_plot)
    
###############################################################################
###############################################################################
# Function for plotting the transition of a list of selected variables
def plot_selected_transition(list_of_variables, 
                             model, 
                             x_trans, 
                             horizon, 
                             save_results, 
                             full_path_results,
                             percent=100):
    # Loop through list of selected variables
    for sublist in list_of_variables:
        if 'R' in sublist:
            percent=100
        
        if len(sublist) == 2:
            variable = sublist[0] # extract variable
            variable_name = sublist[1] # extract variable name
        
            plot_single_transition(model, x_trans, # plot single transition of a 
                                   # given variable from the list
                                   variable, variable_name, 
                                   horizon, 
                                   save_results, 
                                   full_path_results,
                                   percent)
        elif len(sublist) == 4:
            variables = [sublist[0], sublist[2]] # extract variable
            variable_names = [sublist[1], sublist[3]] # extract variable name
            
            plot_double_transition(model, x_trans, # plot double transition of a 
                                   # given variable from the list
                                   variables, variable_names, 
                                   horizon, 
                                   save_results, 
                                   full_path_results,
                                   percent)
        else: 
            print('Error with the dimensions of the variable list.')

###############################################################################
###############################################################################
# Function for plotting the impact of the shock on household decisions over the 
# distribution
def plot_impact_distr(model, 
                      x_transition, 
                      policy, 
                      policy_name, 
                      save, 
                      path,
                      x_threshold=None,
                      borr_cutoff=True,
                      borr_lim=None,
                      percent=100):
    # Get distributional dynamics from transition
    dist_transition = model.get_distributions(x_transition)
    
    # Get distributional transition of specified policy
    dynamic_dist_policy = dist_transition[f'{policy}']
    
    # Get impact effect and sum over grid
    diff_stst = (dynamic_dist_policy[:, :, 1] - dynamic_dist_policy[:, :, 0])/dynamic_dist_policy[:, :, 0]
    diff_stst = diff_stst.sum(0)
    
    # Get asset grid
    a_grid = model['context']['a_grid']
    
    # Make data frame
    impact = {'grid': np.array(a_grid), 
              'impact': diff_stst}
    impact_df = pd.DataFrame(impact)
    
    # Cut off x axis at borrowing limit
    if borr_cutoff == True:
        var_index = [model['variables'].index(v) for v in ['borr_limit']] # Find variable index
        cutoff = x_transition[:,var_index][1]
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
    if borr_lim != None:
        fig_impact.add_vline(x=borr_lim,
                             line_width=3, line_dash="dash", line_color="red")
    fig_impact.show()
    
    if save == True:
        path_plot = os.path.join(path, f'distr_impact_{policy}.svg')
        fig_impact.write_image(path_plot)
        