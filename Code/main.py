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
    os.chdir(full_path_code)

###############################################################################
###############################################################################
# Import custom functions
from custom_functions import (find_closest_grid_point, 
                              find_stable_time)
from plot_functions import (make_stst_policiy_plots, 
                            make_stst_dist_plots,
                            plot_all,
                            plot_selected_transition,
                            bar_plot_asset_dist) 

###############################################################################
###############################################################################
# Preliminaries
start = tm.time() # Start timer

pio.renderers.default = "svg" # For plotting in the Spyder window
pd.set_option('display.max_columns', None) # Show all columns
pd.set_option('display.expand_frame_repr', False)  # Prevent line wrapping

#save_plot_yes = False # If true, it saves the plots after creating them
round_func_4 = lambda x: round(float(x), 4) # Rounding function used throughout
round_func_2 = lambda x: round(float(x), 2) # Rounding function used throughout

###############################################################################
###############################################################################
# Fix initial and terminal borrowing limits
initial_borrowing_limit = -1
terminal_borrowing_limit = -0.5

# Fix persistence in borrowing limit shock
persistence_borrowing_limit = 0.3

###############################################################################
###############################################################################
# Get HANK model dictionary
full_path_hank = os.path.join(full_path_code, 'hank_without_end_labour.yml')
hank_dict = ep.parse(full_path_hank)

# Create model with initial borrowing limit
hank_dict['steady_state']['fixed_values']['lower_bound_a'] = initial_borrowing_limit
hank_dict['definitions'] = hank_dict['definitions'].replace('amin = -1', f'amin = {initial_borrowing_limit}')
hank_dict['definitions'] = hank_dict['definitions'].replace('amin_terminal = 0', f'amin_terminal = {terminal_borrowing_limit}')

hank_dict['steady_state']['fixed_values']['rho_a'] = persistence_borrowing_limit
hank_dict['definitions'] = hank_dict['definitions'].replace('rho_a = 0.3', f'rho_a = {persistence_borrowing_limit}')

hank_model_initial = ep.load(hank_dict)

# Create model with terminal borrowing limit
hank_dict['steady_state']['fixed_values']['lower_bound_a'] = terminal_borrowing_limit
hank_model_terminal = ep.load(hank_dict)

###############################################################################
###############################################################################
# Analysis of Steady States

# Save asset and income grids for later convenience
a_grid = hank_model_initial['context']['a_grid']
skills_grid = hank_model_initial['context']['skills_grid']

# Calculate initial steady state
_ = hank_model_initial.solve_stst()
hank_stst_df = pd.DataFrame(hank_model_initial['stst'].items(), 
                            columns = ['Variable', 'Initial Steady State'])
hank_stst_df['Initial Steady State'] = hank_stst_df['Initial Steady State'].apply(round_func_4)

# Plot initial steady state features 
make_stst_policiy_plots(hank_model_initial)
make_stst_dist_plots(hank_model_initial)
bar_plot_asset_dist(hank_model_initial, shorten=True, 
                    x_threshold = 30, y_threshold = 8)

# Calculate terminal steady state
_ = hank_model_terminal.solve_stst()
hank_stst_df_terminal = pd.DataFrame(hank_model_terminal['stst'].items(),
                                     columns = ['Variable', 'Terminal Steady State'])
hank_stst_df_terminal['Terminal Steady State'] = hank_stst_df_terminal['Terminal Steady State'].apply(round_func_4)

# Plot terminal steady state features 
make_stst_policiy_plots(hank_model_terminal,cutoff=True)
make_stst_dist_plots(hank_model_terminal)
bar_plot_asset_dist(hank_model_terminal, shorten=True, 
                    x_threshold = 30, y_threshold = 8)

# Compare steady states
full_stst_analysis = pd.merge(hank_stst_df, hank_stst_df_terminal, 
                              on = 'Variable', how = 'left')
full_stst_analysis['Percent Change'] = (100*(hank_stst_df_terminal['Terminal Steady State']-hank_stst_df['Initial Steady State'])/hank_stst_df['Initial Steady State']).apply(round_func_2)
print(full_stst_analysis)

# Save result as LaTeX table
#print(full_stst_analysis.to_latex(index=False))

###############################################################################
###############################################################################
# Transition to New Steady State

# Save old seady state as starting point of transition
hank_model_initial_stst = hank_model_initial['stst']

# Find perfect foresight transition to new steady state
x_transition, flag_transition = hank_model_terminal.find_path(init_state = hank_model_initial_stst.values())

###############################################################################
###############################################################################
# Transitional Dynamics

# Fix horizon for plotting
horizon = 15

# Plot transition of all variables       
plot_all(x_transition, hank_model_initial['variables'], 200)

# Plot transition of some selected variables
variables_to_plot = [['C', 'Consumption'], 
                     #['N', 'Labour Hours'],
                     ['y', 'Output'], 
                     ['pi', 'Inflation'],
                     ['w', 'Wage'],
                     ['R', 'Nominal Interest Rate'],
                     ['Rn', 'Notional Interest Rate'],
                     ['div', 'Dividends'],
                     ['tax', 'Taxes']]

plot_selected_transition(variables_to_plot, 
                         hank_model_initial, x_transition, horizon)

###############################################################################
###############################################################################
# Transitional Dynamics of the distribution
dist_transition = hank_model_terminal.get_distributions(x_transition)

dist = dist_transition['dist']

ax, _ = grbar3d(dist[...,:horizon].sum(0), 
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
