#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Andreas Koundouros, University of Bonn
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
absolute_path = os.getcwd() # Set working directory accordingly

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
from custom_functions import (make_paths, 
                              make_models,
                              make_stst_comparison) # make comparison of steady states

from plot_functions import (plot_single_policy, 
                            plot_selected_policies, # plot steady state policy functions
                            make_stst_dist_plots, # plot steady state distributions
                            bar_plot_asset_dist, # plot steady state distribution in bar plot
                            plot_all, # plot all IRFs
                            plot_selected_transition) # plot selected IRFs

###############################################################################
###############################################################################
# Preliminaries
start = tm.time() # Start timer

pio.renderers.default = "svg" # For plotting in the Spyder window
pd.set_option('display.max_columns', None) # Show all columns
pd.set_option('display.expand_frame_repr', False) # Prevent line wrapping

save_results = False # If true, results (tables and plots) are stored

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
policies_to_plot = [['a', 'Assets'],
                    ['c', 'Consumption']]
if full_path_hank.endswith('hank_with_end_labour.yml'):
    policies_to_plot.append(['n', 'Labour Supply'])

full_path_results = make_paths(full_path_hank, 
                               full_path_code, 
                               shock_limit)

###############################################################################
###############################################################################
# Initial Steady State

# Find initial steady state
stst = hank_model_initial.solve_stst()

# Plot steady state policy functions
plot_selected_policies(policies_to_plot, hank_model_initial, 
                       save_results, full_path_results, 'initial',
                       borr_cutoff=None, x_threshold=None)

# Plot marginal propensities to consume
plot_single_policy(hank_model_initial, 'mpc', 'MPC',
                   save_results, full_path_results, 'initial',
                   borr_cutoff=None, x_threshold=3)

# Plot steady state distributions
make_stst_dist_plots(hank_model_initial)
bar_plot_asset_dist(hank_model_initial,
                    x_threshold = 30, y_threshold = 8)


###############################################################################
###############################################################################
# Transitory Shocks
shock = ('e_a', -0.05)

# Find perfect foresight transition to new steady state
x_transition, _ = hank_model_initial.find_path(shock)

# Fix horizon for plotting
horizon = 15

###############################################################################
###############################################################################
# Aggregate transitional dynamics

# If desired, plot transition of all variables       
plot_all(x_transition, hank_model_initial['variables'], horizon)

# Select variables to plot (with descriptions)
variables_to_plot = [['C', 'Consumption'], 
                     ['y', 'Output'], 
                     ['pi', 'Inflation'], 
                     ['w', 'Wage'], 
                     ['R', 'Nominal Interest Rate', 'Rn', 'Notional Interest Rate'], 
                     ['Rr', 'Ex-Post Real Interest Rate'], 
                     ['div', 'Dividends'],
                     ['tax', 'Taxes'],
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
dist_transition = hank_model_initial.get_distributions(x_transition)

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
