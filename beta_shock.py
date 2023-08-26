#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Andreas Koundouros
Date: 26.08.2023

This file implements the permanent shock to the household discount factor in 
the baseline model, as described in appendix E.3 of the thesis.
"""

###############################################################################
###############################################################################
###############################################################################
# Packages
import os # path management
import time as tm # timing
import plotly.io as pio # plot settings
import econpizza as ep # Econpizza

###############################################################################
###############################################################################
###############################################################################
# Imports

# Custom functions
from custom_functions import get_agg_and_dist_transitions_and_check_c

# Custom functions for plotting
from plot_functions import plot_all, plot_selected_transition

###############################################################################
# Set working directory to folder 'master_thesis' to execute this file
if not os.getcwd().endswith('master-thesis'):
    raise Exception(f'Set working directory to folder \'master_thesis\'! Currently it is \'{os.getcwd()}\'')

###############################################################################
# Preliminaries
start = tm.time() # Start timer

save_results = True # True: save results 
show_titles_in_plots = False # True: show plot titles

pio.renderers.default = 'svg' # For plotting in the Spyder window

###############################################################################
###############################################################################
###############################################################################
# Obtain initial and terminal models 

# Path to the model and parse the model
model_path = os.path.join(os.getcwd(), 'Models', 
                          'hank_baseline_beta.yml') # baseline model with shock to household discount factor
hank_dict = ep.parse(model_path)

# Set borrowing limits from baseline (for identical asset grid)
init = -2.3485
term = -2.1775
hank_dict['definitions'] = hank_dict['definitions'].replace('amin = 0', f'amin = {init}')
hank_dict['steady_state']['fixed_values']['phi'] = init
hank_dict['definitions'] = hank_dict['definitions'].replace('amin_terminal = 0', f'amin_terminal = {term}')

# Load initial model
hank_model_initial = ep.load(hank_dict)

# Create model with terminal discount factor
hank_dict['steady_state']['fixed_values']['beta'] = 0.992 

# Load terminal model
hank_model_terminal = ep.load(hank_dict)

###############################################################################
# STEADY STATES
###############################################################################

_ = hank_model_initial.solve_stst() # Solve for steady state
_ = hank_model_terminal.solve_stst() # Solve for steady state

###############################################################################
# TRANSITION
###############################################################################

# Initial seady state and distribution as starting point of transition
hank_model_initial_stst = hank_model_initial['stst'].copy()
hank_model_initial_dist = hank_model_initial['steady_state']['distributions'].copy()

# Get transition and check for negative values in individual 
# consumption responses
x_transition, _ = get_agg_and_dist_transitions_and_check_c(hank_model_terminal,
                                                           hank_model_initial_stst,
                                                           hank_model_initial_dist)
    
###############################################################################
###############################################################################
###############################################################################
# Plot impulse responses to shock

# Fix horizon for plotting
horizon = 100 # long horizon

###############################################################################
# Check for convergence and plot relevant variables 
   
plot_all(x_transition, hank_model_initial['variables'], 
          bunch=True, 
          horizon=200)

plot_selected_transition([['beta', 'Discount Factor', 'Model Units'], 
                          ['C', 'Consumption', 'Percent Deviation'], 
                          ['D', 'Household Debt', 'Model Units'], 
                          ['DY', 'Household Debt-to-GDP', 'Percent of Output']], 
                         hank_model_terminal, 
                         x_transition, horizon, 
                         save_results, 'baseline_beta_permanent', 
                         title=show_titles_in_plots)

###############################################################################
###############################################################################
###############################################################################
# Print run time
print(f'It took {round((tm.time()-start)/60, 2)} minutes to execute this script.')
