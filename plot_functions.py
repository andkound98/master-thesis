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
import os # path management
import pandas as pd
import numpy as np
#import jax
import jax.numpy as jnp
from grgrlib import grplot
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from grgrlib import grbar3d

# Import custom functions
from custom_functions import (make_policy_df,
                              shorten_asset_dist)

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
                        x_threshold=50, 
                        y_threshold=1.5)
    
    # Plot steady state policies over assets depending on skill
    policies_to_plot = [['a', 'Bonds/IOUs'],
                        ['c', 'Consumption Units']]
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
    
    # Plot
    fig_policy =  px.line(policy_df,
                          x = 'grid',
                          y = policy_df.columns.tolist(),
                          color_discrete_sequence=px.colors.qualitative.D3[:policy_df.shape[1]]) 
    fig_policy.update_layout(title=None, 
                             xaxis_title='Bond/IOU Holdings', 
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
    
    # Show plot
    fig_policy.show()
    
    # Save plot
    if save_results == True:
        path_plot = os.path.join(os.getcwd(),
                                 'Results',
                                 f'{exact_path}',
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
    
    # Plot
    fig_dist_skills_and_assets, _ = grbar3d(percent*full_dist,
                                            xedges=jnp.arange(1, 
                                                              (len(full_dist)+1)), 
                                            yedges=a_grid, 
                                            figsize=(9,7), 
                                            depth=.5)
    
    # Label axes
    fig_dist_skills_and_assets.set_xlabel('Productivity')
    fig_dist_skills_and_assets.set_ylabel('Bond/IOU Holdings')
    fig_dist_skills_and_assets.set_zlabel('Percent')
    
    # Adjust perspective
    fig_dist_skills_and_assets.view_init(azim=120)

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
    
    # Colour of the distribution
    colours = [px.colors.qualitative.D3[0]]*len(a_grid)
    
    # Get lowest grid point with positive frequency 
    pos_a_grid = a_grid[y>0]
    ll=pos_a_grid.index
    first_pos_entry_index = ll[0]
    pos_a_grid.reset_index(drop=True, inplace=True)
    
    # Get the frequency of that grid point
    pos_y = y[y>0]
    pos_y.reset_index(drop=True, inplace=True)
    
    # Empty plot
    fig = go.Figure() 
    
    # Fill plot step-by-step
    for i in range(len(a_grid)):
        if i == first_pos_entry_index: # make the very first bar oversized so that it becomes clear in the figure
            x_position = bar_positions[i] - bar_widths[140] / 2
            fig.add_trace(go.Bar(
                x=[x_position],
                y=np.array(y[i]),
                width=bar_widths[140],
                marker=dict(color=colours[i])))
        
        elif y[i] > y_threshold: # replace value over y-axis threshold by threshold
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

    fig.update_layout(title=None, 
                      xaxis_title='Bond/IOU Holdings', 
                      yaxis_title='Percent',
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
                                   dict(x=(pos_a_grid.tail(1).iloc[0]-(bar_positions[i]/8)),
                                                     y=y_threshold*(1/5),
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
    
    # Show plot
    fig.show()
    
    # Save plot
    if save_results == True:
        path_plot = os.path.join(os.getcwd(),
                                 'Results',
                                 f'{exact_path}',
                                 f'stst_dist_2d_bar_{exact_path}_{state}.svg')
        fig.write_image(path_plot)

###############################################################################
###############################################################################
# Function for plotting a comparison of two different steady states
def plot_compare_stst(hank_model_initial,
                      hank_model_terminal,
                      settings,
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
    
    # Plot
    fig_dists = px.line(dists_df, 
                        x = 'grid', 
                        y = ['Initial', 'Terminal'],
                        title='', 
                        color_discrete_sequence=[px.colors.qualitative.D3[0], 
                                                 px.colors.qualitative.D3[1]])#.update_traces(selector={"name": 'Initial'},line={"dash": "dash"})
    fig_dists.update_layout(xaxis_title='Bond/IOU Holdings',
                            yaxis_title='Percent', 
                            plot_bgcolor = 'whitesmoke', 
                            font=dict(family="Times New Roman",
                                             size=20,
                                             color="black"), 
                            margin=dict(l=15, r=15, t=5, b=5), 
                            legend=dict(yanchor="top", y=0.98, 
                                        xanchor="right", x=0.98,
                                        font=dict(size=28)), 
                            legend_title=None)
    fig_dists.update_traces(line=dict(width=2))
    
    # Show plot
    fig_dists.show()
    
    # Save plot
    if save_results == True:
        path_plot = os.path.join(os.getcwd(),
                                 'Results',
                                 f'{exact_path}',
                                 f'stst_dist_2d_comparison_{exact_path}.svg')
        fig_dists.write_image(path_plot)
    
    ###########################################################################
    # Compare steady state asset accumulation, averaged over productivity 
    # levels
    
    # Get stationary distribution over productivity levels 
    skills_stationary = hank_model_initial['context']['skills_stationary']
    
    # Plot steady state policies over assets depending on skill
    policies_to_plot = [['a', 'Asset Accumulation'],
                        ['c', 'Consumption Units']]
    if settings['Model'] == 'end_L':
        policies_to_plot.append(['n', 'Labour Supply'])
        
    for sublist in policies_to_plot:
        policy = sublist[0]
        policy_name = sublist[1]
        # Asset accumulation in the initial steady state
        asset_acc_init = make_policy_df(hank_model_initial, f'{policy}', 
                                        borr_cutoff=None, x_threshold=x_threshold)
        
        if policy == 'a':
            asset_acc_init[asset_acc_init.columns[1:]] = asset_acc_init[asset_acc_init.columns[1:]].sub(a_grid_init, 
                                                                                                        axis='rows')
            
        asset_acc_long = asset_acc_init.iloc[:, 1:]
        df_array = asset_acc_long.to_numpy()
        dot_product_init = jnp.dot(df_array, skills_stationary)
        asset_acc_init['Initial'] = dot_product_init
        
        # Asset accumulation in the terminal steady state
        asset_acc_term = make_policy_df(hank_model_terminal, f'{policy}', 
                                        borr_cutoff=phi, 
                                        x_threshold=x_threshold)
        
        if policy == 'a':
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
    
        # Plot
        fig_acc = px.line(acc_df, 
                          x = 'grid', 
                          y = ['Initial', 'Terminal'],
                          title='', 
                          color_discrete_sequence=[px.colors.qualitative.D3[0], 
                                                   px.colors.qualitative.D3[1]])#.update_traces(selector={"name": 'Initial'}, line={"dash": "dash"})
        
        if policy == 'a':
            fig_acc.update_layout(xaxis_title='Bond/IOU Holdings',
                                    yaxis_title=f'{policy_name}', 
                                    plot_bgcolor = 'whitesmoke', 
                                    font=dict(family="Times New Roman",
                                                     size=20,
                                                     color="black"), 
                                    margin=dict(l=15, r=15, t=5, b=5), 
                                    legend=dict(yanchor="top", y=0.98, 
                                                xanchor="right", x=0.98,
                                                font=dict(size=28)), 
                                    legend_title=None)
            
        if policy != 'a':
            fig_acc.update_layout(xaxis_title='Bond/IOU Holdings',
                                    yaxis_title=f'{policy_name}', 
                                    plot_bgcolor = 'whitesmoke', 
                                    font=dict(family="Times New Roman",
                                                     size=20,
                                                     color="black"), 
                                    margin=dict(l=15, r=15, t=5, b=5), 
                                    showlegend=False)
            
        fig_acc.update_traces(line=dict(width=2))
        fig_acc.show() # Show plot
        
        # Save plot
        if save_results == True:
            path_plot = os.path.join(os.getcwd(),
                                     'Results',
                                     f'{exact_path}',
                                     f'stst_{policy}_{exact_path}.svg')
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
    
    if variable[0] in ['R', 'Rn', 'Rr', 'Rrminus', 'pi']:
        x_single_transition = np.column_stack([time, # Concatenate IRF and time vector
                                               4*percent*(x_trans[:horizon,var_index] - 1.0)])
        yfin = 4*percent*(x_trans[-1,var_index] - 1.0)
    elif variable[0] in ['Rbar']:
        x_single_transition = np.column_stack([time, # Concatenate IRF and time vector
                                               4*percent*x_trans[:horizon,var_index]])
        yfin = 4*percent*x_trans[-1,var_index]
    elif variable[0] in ['beta', 'D', 'DY', 'phi', 'gr_liquid', 'Top10C', 'Top10A', 'Top1C', 'Top1A', 'Top25C', 'Top25A', 'Bot25A', 'Bot25C']:
        x_single_transition = np.column_stack([time, # Concatenate IRF and time vector
                                               x_trans[:horizon,var_index]])
        yfin = x_trans[-1,var_index]
    else:
        stst = x_trans[0, var_index] # Find (initial) steady state
        x_single_transition = np.column_stack([time, # Concatenate IRF and time vector
                                               percent*((x_trans[:horizon,var_index] - stst)/stst)])
        yfin = percent*((x_trans[-1,var_index] - stst)/stst)
        
    x_single_transition_df = pd.DataFrame(x_single_transition, # Turn into data frame
                                          columns = ['Quarters', 
                                                     f'{var_name}'])
    
    # Plot
    fig = px.line(x_single_transition_df, 
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
    
    # More space at heading 
    if var_name != '':
        fig.update_layout(margin=dict(l=15, r=15, t=50, b=5))
    
    fig.update_traces(line=dict(width=4))
    
    # Add line for terminal steady state
    if round(x_trans[0,var_index],5) != round(x_trans[-1,var_index],5): 
        fig.add_hline(y=yfin.item(), line_width=3, line_dash="dash", 
                      line_color="red")
    
    fig.show() # Show plot
    
    # Save plot
    if save_results == True:
        path_plot = os.path.join(os.getcwd(),
                                 'Results',
                                 f'{exact_path}',
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
    
    if variables[0] in ['R', 'Rn', 'Rr', 'Rminus', 'pi'] and variables[1] in ['R', 'Rn', 'Rr', 'Rminus', 'pi']:
        x_double_transition = np.column_stack([time, # Concatenate IRFs and time vector
                                               4*percent*(x_trans[:horizon,var_indices] - 1.0)])
        yfin = 4*percent*(x_trans[-1,var_indices[0]] - 1.0)
    else:
        stst = x_trans[0, var_indices] # Find (initial) steady state
        x_double_transition = np.column_stack([time, # Concatenate IRF and time vector
                                               percent*((x_trans[:horizon,var_indices] - stst)/stst)])
        yfin = percent*((x_trans[-1,var_indices[0]] - stst[0])/stst[0])
        
    x_double_transition_df = pd.DataFrame(x_double_transition, # Turn into data frame
                                          columns = ['Quarters', 
                                                     f'{var_names[0]}', 
                                                     f'{var_names[1]}'])
    
    # Plot
    fig = px.line(x_double_transition_df, 
                  x = 'Quarters',
                  y = [f'{var_names[0]}', f'{var_names[1]}'],
                  color_discrete_map={f'{var_names[0]}': px.colors.qualitative.D3[0],
                                      f'{var_names[1]}': px.colors.qualitative.D3[1]}).update_traces(selector={"name": f'{var_names[1]}'}, 
                                                                                                     line={"dash": "dash"})
    fig.update_layout(title='', # empty title
                       xaxis_title='Quarters', # x-axis labeling
                       yaxis_title=f'{unit}', # y-axis labeling
                       legend=dict(yanchor="bottom", y=0.02, 
                                   xanchor="right", x=0.98,
                                   font=dict(size=28)), 
                       legend_title=None, 
                       plot_bgcolor='whitesmoke', 
                       margin=dict(l=15, r=15, t=5, b=5),
                       font=dict(family="Times New Roman", # adjust font
                                 size=20,
                                 color="black"))
    fig.update_traces(line=dict(width=4))
    
    # Add line for terminal steady state
    if round(x_trans[0,var_indices[0]],5) != round(x_trans[-1,var_indices[0]],5): 
        fig.add_hline(y=yfin.item(), line_width=3, line_dash="dash", 
                      line_color="red")
    
    # Show plot
    fig.show()
    
    # Save plot
    if save_results == True:
        path_plot = os.path.join(os.getcwd(),
                                 'Results',
                                 f'{exact_path}',
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
        
            plot_single_transition(model, x_trans, # plot single transition of
                                   # a given variable from the list
                                   variable, variable_name, unit,
                                   horizon, 
                                   save_results, 
                                   exact_path,
                                   percent)
            
        elif len(sublist) == 5:
            variables = [sublist[0], sublist[2]] # extract variable
            variable_names = [sublist[1], sublist[3]] # extract variable name
            unit = sublist[4] # extract unit
            
            plot_double_transition(model, x_trans, # plot double transition of
                                   # a given variable from the list
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
        sign_change = jnp.not_equal(jnp.sign(av_period_1_policy), 
                                    jnp.sign(av_stst_policy)) 
        
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

    # Cut off x axis at threshold
    if x_threshold != None:
        impact_df.loc[impact_df['grid'] > x_threshold, :] = np.nan
        
        if policy == 'a':
            condition = (-0.5 < impact_df['grid']) & (impact_df['grid'] < 0.5)
            impact_df.loc[condition, :] = np.nan

    # Plot
    fig_impact=px.line(impact_df, 
                       x = 'grid', 
                       y = 'impact', 
                       title='', 
                       color_discrete_sequence=[px.colors.qualitative.D3[0]])
    fig_impact.update_layout(xaxis_title='Bond/IOU Holdings', 
                              yaxis_title=f'{policy_name}',
                              plot_bgcolor = 'whitesmoke', 
                              font=dict(family="Times New Roman",
                                        size=20,
                                        color="black"),
                              margin=dict(l=15, r=15, t=5, b=5), 
                              legend_title='') 
    fig_impact.update_traces(line=dict(width=4))
    
    # Add terminal borrowing limit as vertical line
    if borr_lim != None:
        fig_impact.add_vline(x=borr_lim,
                             line_width=3, line_dash="dash", line_color="red")
    
    # Show plot
    fig_impact.show()
    
    # Save plot
    if save_results == True:
        path_plot = os.path.join(os.getcwd(),
                                 'Results',
                                 f'{exact_path}',
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
                       x_threshold=150):
    
    # Change in consumption choice on impact over the distribution of assets
    plot_policy_on_impact_over_dist(hank_model_initial, hank_model_terminal, 
                                    x_transition, 'c', 'Consumption', 
                                    save_results, exact_path,
                                    x_threshold=x_threshold,
                                    borr_lim=borr_lim)

    # Change in asset choice on impact over the distribution of assets
    plot_policy_on_impact_over_dist(hank_model_initial, hank_model_terminal, 
                                    x_transition, 'a', 'Assets', 
                                    save_results, exact_path,
                                    x_threshold=x_threshold,
                                    borr_lim=borr_lim)

    # Change in labour choice on impact over the distribution of assets
    try: 
        plot_policy_on_impact_over_dist(hank_model_initial, 
                                        hank_model_terminal, 
                                        x_transition, 'n', 'Labour Supply', 
                                        save_results, exact_path,
                                        x_threshold=x_threshold,
                                        borr_lim=borr_lim)
    except KeyError: # If there is no labour supply policy, pass
        pass

###############################################################################
###############################################################################
# Function to visualise the evolution of the asset distribution over time
def visualise_dist_over_time(initial_model,
                             terminal_model,
                             x_trans,
                             horizon,
                             y_threshold,
                             x_threshold=None,
                             percent=100):
    dist_dyn = terminal_model.get_distributions(trajectory = x_trans,
                                                init_dist = initial_model['steady_state']['distributions'])
    dist_init = jnp.sum(initial_model['steady_state']['distributions'][0],axis=0)
    dist_term = jnp.sum(terminal_model['steady_state']['distributions'][0],axis=0)
    
    for tt in range(0,horizon):
        new_dist = jnp.sum(dist_dyn['dist'][..., tt], axis=0)
        dist_df = pd.DataFrame({'Bond/IOU Holdings': terminal_model['context']['a_grid'],
                           'Initial Distribution': percent*dist_init,
                           'Terminal Distribution': percent*dist_term,
                           f'Distribution t = {tt}': percent*new_dist})
        
        if x_threshold != None:
            dist_df = dist_df[dist_df['Bond/IOU Holdings'] < x_threshold]
        
        fig=px.line(dist_df,
                    x='Bond/IOU Holdings',
                    y=['Initial Distribution',
                       f'Distribution t = {tt}',
                       'Terminal Distribution'],
                    title=f't={tt}',
                    color_discrete_sequence=px.colors.qualitative.D3[:3]).update_traces(line=dict(width=3))
        fig.update_layout(xaxis_title='Bond/IOU Holdings', 
                          yaxis_title='%',
                          plot_bgcolor = 'whitesmoke', 
                          font=dict(family="Times New Roman",
                                            size=20,
                                            color="black"),
                          margin=dict(l=15, r=15, t=50, b=5), 
                          legend_title='',
                          legend=dict(yanchor="top", y=0.98, 
                                      xanchor="right", x=0.98,
                                      font=dict(size=28)), ) 
        fig.add_vline(x=x_trans[tt,terminal_model['variables'].index('phi')],
                      line_width=3, line_dash="dash", line_color="red")
        fig.update_yaxes(range=[0,y_threshold])
        fig.show()


def plot_percentile_transitions_C(hank_model_terminal,
                                  x_transition,
                                  percentiles,
                                  horizon,
                                  save_results, 
                                  exact_path,
                                  title=True,
                                  percent=100):
    time = list(range(0, horizon, 1)) # Time vector
    
    df = pd.DataFrame({'Quarters': time})
    
    if title == False:
        fig_title = ''
        top_space = 5
    elif title == True:
        fig_title = 'Consumption Response by Percentile'
        top_space = 50
    
    for pp in percentiles:
        pp_var = pp[0]
        pp_name = pp[1]
        var_index = hank_model_terminal['variables'].index(f'{pp_var}') # Find variable index
        
        disagg_transition = x_transition[:horizon,var_index] * x_transition[:horizon,hank_model_terminal['variables'].index('C')]
        disagg_transition_per = percent*((disagg_transition - disagg_transition[0])/ disagg_transition[0])
        
        df[f'{pp_name}'] = disagg_transition_per
    
    # Plot
    fig = px.line(df,
                  x = 'Quarters',
                  y = df.columns.tolist(), 
                  color_discrete_sequence=px.colors.qualitative.D3[:len(percentiles)]).update_traces(line=dict(width=3))
    fig.update_layout(title=f'{fig_title}', # empty title
                       xaxis_title='Quarters', # x-axis labeling
                       yaxis_title='Percent Deviation', # y-axis labeling
                       legend=dict(yanchor="bottom", y=0.02, 
                                   xanchor="right", x=0.98,
                                   font=dict(size=28)), 
                       legend_title='', 
                       plot_bgcolor='whitesmoke', 
                       margin=dict(l=15, r=15, t=top_space, b=5),
                       font=dict(family="Times New Roman", # adjust font
                                 size=20,
                                 color="black"))
    fig.show() # Show plot
    
    if save_results == True:
        path_plot = os.path.join(os.getcwd(),
                                 'Results',
                                 f'{exact_path}',
                                 'percentile_transitions_C.svg')
        fig.write_image(path_plot)
    

###############################################################################
###############################################################################
# Function to compare the transitions of some selected variables 
def compare_selected_transitions(list_of_transition_dfs,
                                 variables_to_plot, 
                                 horizon, 
                                 legend,
                                 save_results, 
                                 comparison,
                                 percent=100,
                                 title=True):
    time = list(range(0, horizon, 1)) # Time vector

    # Loop through list of selected variables
    for sublist in variables_to_plot:
        if len(sublist) == 3:
        
            variable = sublist[0] # extract variable
    
            if title == True:
                variable_name = sublist[1] # extract variable name
            if title != True:
                variable_name = ''
                
            unit = sublist[2] # extract unit
            
            transition_df = pd.DataFrame({'Quarters': time})
            
            if variable in ['R', 'Rn', 'Rr', 'Rrminus', 'pi']:
                for i, df in enumerate(list_of_transition_dfs):
                    col_name = f'{legend[i]}'
                    new_col = 4*percent*(df[f'{variable}'][:horizon] - 1.0)
                    transition_df[col_name] = new_col.reset_index(drop=True)
    
            elif variable in ['Rbar']:
                for i, df in enumerate(list_of_transition_dfs):
                    col_name = f'{legend[i]}'
                    new_col = 4*percent*(df[f'{variable}'][:horizon])
                    transition_df[col_name] = new_col.reset_index(drop=True)
                
            elif variable in ['beta', 'D', 'DY', 'phi', 'gr_liquid', 'Top10C', 'Top10A', 'Top1C', 'Top1A', 'Top25C', 'Top25A', 'Bot25A', 'Bot25C']:
                for i, df in enumerate(list_of_transition_dfs):
                    col_name = f'{legend[i]}'
                    new_col = df[f'{variable}'][:horizon]
                    transition_df[col_name] = new_col.reset_index(drop=True)
                
            else:
                for i, df in enumerate(list_of_transition_dfs):
                    col_name = f'{legend[i]}' #col_name = f'{variable}_{legend[i]}'
                    new_col = percent*((df[f'{variable}'][:horizon] - df[f'{variable}'][0]) / df[f'{variable}'][0])
                    transition_df[col_name] = new_col.reset_index(drop=True)
        
        elif len(sublist) == 5:
            
            variable1 = sublist[0] # extract variable
            variable2 = sublist[2] # extract variable
            
            variable_name1 = sublist[1] # extract variable name
            variable_name2 = sublist[3] # extract variable name
                
            unit = sublist[-1] # extract unit
            
            transition_df = pd.DataFrame({'Quarters': time})
            
            if variable1 in ['R', 'Rn', 'Rr', 'Rrminus', 'pi'] and variable2 in ['R', 'Rn', 'Rr', 'Rrminus', 'pi']:
                for i, df in enumerate(list_of_transition_dfs):
                    col_name1 = f'{variable_name1}; {legend[i]}'
                    new_col1 = 4*percent*(df[f'{variable1}'][:horizon] - 1.0)
                    transition_df[col_name1] = new_col1.reset_index(drop=True)
                    
                    col_name2 = f'{variable_name2}; {legend[i]}'
                    new_col2 = 4*percent*(df[f'{variable2}'][:horizon] - 1.0)
                    transition_df[col_name2] = new_col2.reset_index(drop=True)
            
            else:
                for i, df in enumerate(list_of_transition_dfs):
                    col_name1 = f'{variable_name1}; {legend[i]}'
                    new_col1 = percent*((df[f'{variable1}'][:horizon] - df[f'{variable1}'][0]) / df[f'{variable1}'][0])
                    transition_df[col_name1] = new_col1.reset_index(drop=True)
                    
                    col_name2 = f'{variable_name2}; {legend[i]}'
                    new_col2 = percent*((df[f'{variable2}'][:horizon] - df[f'{variable2}'][0]) / df[f'{variable2}'][0])
                    transition_df[col_name2] = new_col2.reset_index(drop=True)
        
        # Plot
        fig = px.line(transition_df,
                      x = 'Quarters',
                      y = transition_df.columns.tolist(), 
                      color_discrete_sequence=px.colors.qualitative.D3[:transition_df.shape[1]-1]).update_traces(line=dict(width=4))
        if len(sublist) == 3 and transition_df.shape[1] == 3:
            fig.update_traces(selector={"name": f'{legend[0]}'},
                              line={"dash": "dash"})
        elif len(sublist) == 3 and transition_df.shape[1] == 4:
            fig.update_traces(selector={"name": f'{legend[0]}'},
                              line={"dash": "dash"})
            fig.update_traces(selector={"name": f'{legend[2]}'},
                              line={"dash": "dot"})
        elif len(sublist) == 5 and transition_df.shape[1] == 5:
            fig.update_traces(selector={"name": f'{variable_name2}; {legend[0]}'},
                              line={"dash": "dash"})
            fig.update_traces(selector={"name": f'{variable_name2}; {legend[1]}'},
                              line={"dash": "dash"})
            variable_name = ''
        elif len(sublist) == 5 and transition_df.shape[1] == 7:
            fig.update_traces(selector={"name": f'{variable_name2}; {legend[0]}'},
                              line={"dash": "dash"})
            fig.update_traces(selector={"name": f'{variable_name2}; {legend[1]}'},
                              line={"dash": "dash"})
            fig.update_traces(selector={"name": f'{variable_name2}; {legend[2]}'},
                              line={"dash": "dash"})
            variable_name = ''
            
        if variable_name != '':
            top_space = 50
        elif variable_name == '':
            top_space = 5
        
        if 'C' in sublist:
            fig.update_layout(title=f'{variable_name}', # empty title
                               xaxis_title='Quarters', # x-axis labeling
                               yaxis_title=f'{unit}', # y-axis labeling
                               legend=dict(yanchor="bottom", y=0.02, 
                                           xanchor="right", x=0.98,
                                           font=dict(size=28)), 
                               legend_title='', 
                               plot_bgcolor='whitesmoke', 
                               margin=dict(l=15, r=15, t=top_space, b=5),
                               font=dict(family="Times New Roman", # adjust font
                                         size=20,
                                         color="black"))

        elif 'C' not in sublist and transition_df.shape[1] != 5:
            fig.update_layout(title=f'{variable_name}', # empty title
                               xaxis_title='Quarters', # x-axis labeling
                               yaxis_title=f'{unit}', # y-axis labeling
                               showlegend=False,
                               plot_bgcolor='whitesmoke', 
                               margin=dict(l=15, r=15, t=top_space, b=5),
                               font=dict(family="Times New Roman", # adjust font
                                         size=20,
                                         color="black"))
        
        if transition_df.shape[1] == 5 or transition_df.shape[1] == 7:
            fig.update_layout(title=None, # empty title
                               xaxis_title='Quarters', # x-axis labeling
                               yaxis_title=f'{unit}', # y-axis labeling
                               legend=dict(yanchor="bottom", y=0.02, 
                                           xanchor="right", x=0.98), 
                               legend_title='', 
                               plot_bgcolor='whitesmoke', 
                               margin=dict(l=15, r=15, t=top_space, b=5),
                               font=dict(family="Times New Roman", # adjust font
                                         size=20,
                                         color="black"))
    
        # Show plot
        fig.show()
        
        # Save plot
        if save_results == True:
            # Define path 
            path = os.path.join(os.getcwd(), 
                                'Results', 
                                'compare_transitions')
            
            # Check if the folder exists
            if not os.path.exists(path):
            # Create the folder if it doesn't exist
                os.makedirs(path)
            
            key1 = comparison['transition_1']
            key2 = comparison['transition_2']
            
            if len(sublist) == 3:
                path_plot = os.path.join(path,
                                         f'comparison_{variable}_{key1}_{key2}.svg')
            elif len(sublist) ==5:
                path_plot = os.path.join(path,
                                         f'comparison_{variable1}_{variable2}_{key1}_{key2}.svg')
            fig.write_image(path_plot)
            