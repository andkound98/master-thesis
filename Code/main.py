#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Andreas Koundouros
Date: 26.04.2023

This file is the main code file for my master thesis.
"""

###############################################################################
###############################################################################
# Import packages
import os
import time as tm # for timing
import econpizza as ep # econpizza
import pandas as pd # for data wrangling
import numpy as np # for data wrangling
import jax.numpy as jnp 
import plotly.io as pio # for plotting
import matplotlib.pyplot as plt # for plotting
from grgrlib import grbar3d # for plotting

###############################################################################
###############################################################################
# Set paths
absolute_path = os.getcwd() # Get current working directory
if absolute_path.endswith('Code'):
    full_path_code = absolute_path
else:
    relative_path_code = os.path.join('Documents', 
                                      'Uni-Masterarbeit', 
                                      'master-thesis',
                                      'Code')
    full_path_code = os.path.join(absolute_path, relative_path_code)
    os.chdir(full_path_code) # Set working directory 

###############################################################################
###############################################################################
# Import custom functions
from custom_functions import (path_results, 
                              make_models, 
                              make_policy_df,
                              make_stst_comparison,) # make comparison of steady states

from plot_functions import (plot_single_policy, 
                            plot_selected_policies, # plot steady state policy functions
                            make_stst_dist_plots, # plot steady state distributions
                            bar_plot_asset_dist, # plot steady state distribution in bar plot
                            #plot_all, # plot all IRFs
                            plot_selected_transition) # plot selected IRFs

###############################################################################
###############################################################################
# Preliminaries
start = tm.time() # Start timer

pio.renderers.default = "svg" # For plotting in the Spyder window
pd.set_option('display.max_columns', None) # Show all columns
pd.set_option('display.expand_frame_repr', False) # Prevent line wrapping

save_results = True # If true, results (tables and plots) are stored

###############################################################################
###############################################################################
# Choose HANK model and shock

# HANK without endogenous labour supply
full_path_hank = os.path.join(full_path_code, 'hank_without_end_labour.yml')
# HANK with endogenous labour supply
#full_path_hank = os.path.join(full_path_code, 'hank_with_end_labour.yml')  

# Shock; True: shock to borrowing limit, False: shock to borrowing wedge
shock_limit = True

###############################################################################
###############################################################################
# Fix borrowing limits according to model used
hank_model_initial, hank_model_terminal, terminal_borrowing_limit = make_models(full_path_hank, 
                                                                                shock_limit)

###############################################################################
###############################################################################
# Steady States

# Save asset and income grids for later convenience
a_grid = hank_model_initial['context']['a_grid']
#skills_grid = hank_model_initial['context']['skills_grid']

# Select steady state policies to plot
policies_to_plot = [['a', 'Asset Holdings'],
                    ['c', 'Consumption']]
if full_path_hank.endswith('hank_with_end_labour.yml'):
    policies_to_plot.append(['n', 'Labour Supply'])

# Get path in which to store the results
full_path_results = path_results(full_path_hank, 
                                 full_path_code, 
                                 shock_limit)

###############################################################################
###############################################################################
# Initial Steady State

# Find initial steady state
_ = hank_model_initial.solve_stst()

# Plot steady state policy functions
plot_selected_policies(policies_to_plot, hank_model_initial, 
                       save_results, full_path_results, 'initial',
                       borr_cutoff=None, x_threshold=None)

# Plot asset accumulation
asset_policy_df_initial = make_policy_df(hank_model_initial, 'a')
asset_policy_df_initial[asset_policy_df_initial.columns[1:]] = asset_policy_df_initial[asset_policy_df_initial.columns[1:]].sub(a_grid, axis='rows')
plot_single_policy(hank_model_initial, 'a', 'Asset Accumulation',
                   save_results, full_path_results, 'initial',
                   policy_df=asset_policy_df_initial, borr_cutoff=None)

# Plot marginal propensities to consume
plot_single_policy(hank_model_initial, 'mpc', 'MPC',
                   save_results, full_path_results, 'initial',
                   borr_cutoff=None, x_threshold = 3)

# Plot steady state distributions
make_stst_dist_plots(hank_model_initial)
bar_plot_asset_dist(hank_model_initial,
                    x_threshold = 30, y_threshold = 8)

###############################################################################
###############################################################################
# Terminal Steady State

# Find terminal steady state
_ = hank_model_terminal.solve_stst()

# Plot steady state policy functions
plot_selected_policies(policies_to_plot, hank_model_terminal, 
                       save_results, full_path_results, 'terminal',
                       borr_cutoff=terminal_borrowing_limit, x_threshold=None)

# Plot asset accumulation
asset_policy_df_terminal = make_policy_df(hank_model_terminal, 'a', borr_cutoff=terminal_borrowing_limit)
asset_policy_df_terminal[asset_policy_df_terminal.columns[1:]] = asset_policy_df_terminal[asset_policy_df_terminal.columns[1:]].sub(a_grid, axis='rows')
plot_single_policy(hank_model_initial, 'a', 'Asset Accumulation',
                   save_results, full_path_results, 'initial',
                   policy_df=asset_policy_df_terminal)

# Plot marginal propensities to consume
plot_single_policy(hank_model_terminal, 'mpc', 'MPC',
                   save_results, full_path_results, 'terminal',
                   borr_cutoff=terminal_borrowing_limit, x_threshold=3)

# Plot steady state distributions
make_stst_dist_plots(hank_model_terminal)
bar_plot_asset_dist(hank_model_terminal,
                    x_threshold = 30, y_threshold = 8)

###############################################################################
###############################################################################
# Comparison of Steady States
stst_comparison = make_stst_comparison(hank_model_initial, 
                                       hank_model_terminal,
                                       save_results, full_path_results)
print(stst_comparison)

for ii in range(1,9):
    compare_asset_acc = {'grid': a_grid,
                         'Initial StSt': asset_policy_df_initial[asset_policy_df_initial.columns[ii]], 
                         'Terminal StSt': asset_policy_df_terminal[asset_policy_df_terminal.columns[ii]]}
    compare_asset_acc_df = pd.DataFrame(compare_asset_acc)
    plot_single_policy(hank_model_initial, 'a', 'Asset Accumulation',
                       False, full_path_results, 'initial',
                       policy_df=compare_asset_acc_df)

###############################################################################
###############################################################################
# Transition to New Steady State

# Save old seady state as starting point of transition
hank_model_initial_stst = hank_model_initial['stst']

# Find perfect foresight transition to new steady state
x_transition, _ = hank_model_terminal.find_path(init_state = hank_model_initial_stst.values())

# Fix horizon for plotting
horizon = 12

###############################################################################
###############################################################################
# Aggregate transitional dynamics

# If desired, plot transition of all variables       
#plot_all(x_transition, hank_model_initial['variables'], horizon)

# Select variables to plot (with descriptions)
variables_to_plot = [['C', 'Consumption'], 
                     ['y', 'Output'], 
                     ['y_prod', 'Production'], 
                     ['pi', 'Inflation'], 
                     ['w', 'Wage'], 
                     ['R', 'Nominal Interest Rate', 'Rn', 'Notional Interest Rate'], 
                     ['Rr', 'Ex-Post Real Interest Rate'], 
                     ['div', 'Dividends'],
                     ['tau', 'Tax Rate'],
                     ['lower_bound_a', 'Borrowing Limit'],
                     ['D', 'Household Debt'], 
                     ['DY', 'Household Debt-to-Output']]

# Depending on the specific HANK model, add aggregate labour hours
if full_path_hank.endswith('hank_without_end_labour.yml'):
    variables_to_plot.append(['n', 'Labour Hours'])
elif full_path_hank.endswith('hank_with_end_labour.yml'):
    variables_to_plot.append(['N', 'Labour Hours'])

# Plot transition of some selected variables
plot_selected_transition(variables_to_plot, 
                         hank_model_initial, x_transition, horizon,
                         save_results, full_path_results)

###############################################################################
###############################################################################
# Disaggregated transitional dynamics

# Get disaggregated responses of the distribution, of consumption and asset 
# holdings
dist_transition = hank_model_terminal.get_distributions(x_transition)

# Store distributional dynamics
dynamic_dist = dist_transition['dist']

# Plot 
ax, _ = grbar3d(100*dynamic_dist[...,:horizon].sum(0),
                xedges=hank_model_terminal['context']['a_grid'], 
                yedges=jnp.arange(horizon), 
                figsize=(9,7), 
                depth=.5, 
                width=.5, 
                alpha=.5)
ax.set_xlabel('wealth')
ax.set_ylabel('time')
ax.set_zlabel('share')
ax.view_init(azim=50)

###############################################################################
###############################################################################
# Print run time
print('It took', round((tm.time()-start)/60, 2), 
      'minutes to execute this script.')
