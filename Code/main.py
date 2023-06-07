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
# Fix paths
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
    
path_save_tables = '/Users/andreaskoundouros/Documents/Uni-Masterarbeit/master-thesis/Tables'

###############################################################################
###############################################################################
# Import custom functions
from custom_functions import make_stst_comparison # quick steady state comparison

from plot_functions import (make_stst_policiy_plots, # plot steady state policies
                            make_stst_dist_plots, # plot steady state distributions
                            plot_all, # plot all IRFs
                            plot_selected_transition, # plot selected IRFs
                            bar_plot_asset_dist) # plot steady state distribution in bar plot

###############################################################################
###############################################################################
# Preliminaries
start = tm.time() # Start timer

pio.renderers.default = "svg" # For plotting in the Spyder window
pd.set_option('display.max_columns', None) # Show all columns
pd.set_option('display.expand_frame_repr', False)  # Prevent line wrapping

save_tables = True # If true, it saves the tables 
save_plots = False # If true, it saves the plots 

###############################################################################
###############################################################################
# Choose HANK model

# HANK without endogenous labour supply
full_path_hank = os.path.join(full_path_code, 'hank_without_end_labour.yml')
# HANK with endogenous labour supply
#full_path_hank = os.path.join(full_path_code, 'hank_with_end_labour.yml')  

###############################################################################
###############################################################################
# Fix borrowing limits
if full_path_hank.endswith('hank_without_end_labour.yml'):
    initial_borrowing_limit = -1 # initial borrowing limit
    terminal_borrowing_limit = -0.8 # terminal borrowing limit
elif full_path_hank.endswith('hank_with_end_labour.yml'):
    initial_borrowing_limit = -2 # initial borrowing limit
    terminal_borrowing_limit = -1 # terminal borrowing limit

# Fix persistence in borrowing limit shock
persistence_borrowing_limit = 0.3  

###############################################################################
###############################################################################
# Get model as dictionary
hank_dict = ep.parse(full_path_hank)

# Create model with initial borrowing limit
hank_dict['steady_state']['fixed_values']['lower_bound_a'] = initial_borrowing_limit
hank_dict['definitions'] = hank_dict['definitions'].replace('amin = -1', f'amin = {initial_borrowing_limit}')
hank_dict['definitions'] = hank_dict['definitions'].replace('amin_terminal = 0', f'amin_terminal = {terminal_borrowing_limit}')

hank_dict['steady_state']['fixed_values']['rho_a'] = persistence_borrowing_limit
hank_dict['definitions'] = hank_dict['definitions'].replace('rho_a = 0.3', f'rho_a = {persistence_borrowing_limit}')

hank_model_initial = ep.load(hank_dict) # Load into model instance

# Create model with terminal borrowing limit
hank_dict['steady_state']['fixed_values']['lower_bound_a'] = terminal_borrowing_limit
hank_model_terminal = ep.load(hank_dict) # Load into model instance

###############################################################################
###############################################################################
# Analysis of Steady States

# Save asset and income grids for later convenience
a_grid = hank_model_initial['context']['a_grid']
skills_grid = hank_model_initial['context']['skills_grid']

# Calculate initial steady state
_ = hank_model_initial.solve_stst()

# Plot features of the initial steady state  
make_stst_policiy_plots(hank_model_initial)
make_stst_dist_plots(hank_model_initial)
bar_plot_asset_dist(hank_model_initial, shorten=True, 
                    x_threshold = 30, y_threshold = 8)

# Calculate terminal steady state
_ = hank_model_terminal.solve_stst()

# Plot features of the terminal steady state 
make_stst_policiy_plots(hank_model_terminal,cutoff=True)
make_stst_dist_plots(hank_model_terminal)
bar_plot_asset_dist(hank_model_terminal, shorten=True, 
                    x_threshold = 30, y_threshold = 8)

# Compare steady states
stst_comparison = make_stst_comparison(hank_model_initial, hank_model_terminal,
                                       save_tables, path_save_tables)
print(stst_comparison)

###############################################################################
###############################################################################
# Transition to New Steady State

# Save old seady state as starting point of transition
hank_model_initial_stst = hank_model_initial['stst']

# Find perfect foresight transition to new steady state
x_transition, _ = hank_model_terminal.find_path(init_state = hank_model_initial_stst.values())

###############################################################################
###############################################################################
# Aggregate transitional dynamics

# Fix horizon for plotting
horizon = 15

# If desired, plot transition of all variables       
plot_all(x_transition, hank_model_initial['variables'], horizon)

# Select variables to plot (with descriptions)
variables_to_plot = [['C', 'Consumption'], 
                     ['y', 'Output'], 
                     ['pi', 'Inflation'], 
                     ['w', 'Wage'], 
                     ['R', 'Nominal Interest Rate'], 
                     ['Rn', 'Notional Interest Rate'], 
                     ['Rr', 'Ex-Post Real Interest Rate'], 
                     ['div', 'Dividends'],
                     ['tax', 'Taxes'],
                     ['lower_bound_a', 'Borrowing Limit']]

# Depending on the specific HANK model, add aggregate labour hours
if full_path_hank.endswith('hank_without_end_labour.yml'):
    variables_to_plot.append(['n', 'Labour Hours'])
elif full_path_hank.endswith('hank_with_end_labour.yml'):
    variables_to_plot.append(['N', 'Labour Hours'])

# Plot transition of some selected variables
plot_selected_transition(variables_to_plot, 
                         hank_model_initial, x_transition, horizon)

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
