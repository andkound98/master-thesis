#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Andreas Koundouros
Date: 21.06.2023

This file contains the code which investigates asymmetry.
"""

###############################################################################
###############################################################################
###############################################################################
# Import packages
import os # path management
import time as tm # timing
import plotly.io as pio # plotting
import econpizza as ep
    
###############################################################################
# Set working directory to folder 'master_thesis' to execute this file
if not os.getcwd().endswith('master-thesis'):
    raise Exception('Set working directory to folder \'master_thesis\'!')

###############################################################################
###############################################################################
###############################################################################
# Import functions

# Custom functions for various purposes
from custom_functions import (get_model_path, 
                              get_exact_results_path,
                              get_parametrisation,
                              return_models_permanent,
                              stst_overview,
                              save_transition)

# Custom functions for plotting
from plot_functions import (plot_full_stst, 
                            plot_compare_stst,
                            plot_all,
                            plot_selected_transition,
                            plot_policy_impact)

save_results = False
exact_path = ''
settings = ''

hank_dict = ep.parse('/Users/andreaskoundouros/Documents/Uni-Masterarbeit/master-thesis/Models/hank_baseline_init.yml')
hank_model_initial = ep.load(hank_dict)
hank_dict = ep.parse('/Users/andreaskoundouros/Documents/Uni-Masterarbeit/master-thesis/Models/hank_baseline_term.yml')
hank_model_terminal = ep.load(hank_dict)

#######################################################################
# STEADY STATES
#######################################################################

# Initial steady state
_ = hank_model_initial.solve_stst() # Solve for steady state
# plot_full_stst(hank_model_initial, # Visualise steady state
#                settings, shock_model_parameters,
#                save_results, exact_path, 'initial')

# Terminal steady state
_ = hank_model_terminal.solve_stst() # Solve for steady state
# plot_full_stst(hank_model_terminal, # Visualise steady state
#                settings, shock_model_parameters,
#                save_results, exact_path, 'terminal', 
#                borr_cutoff=shock_model_parameters['terminal_borrowing_limit'])

# Compare steady states
stst_comparison = stst_overview([hank_model_initial, hank_model_terminal], 
                                save_results, exact_path)
print(stst_comparison)

# plot_compare_stst(hank_model_initial, hank_model_terminal,
#                   save_results, exact_path,
#                   shock_model_parameters['terminal_borrowing_limit'], 
#                   x_threshold=25)

#######################################################################
# TRANSITION
#######################################################################

# Initial seady state as starting point of transition
hank_model_initial_stst = hank_model_initial['stst'].copy()
hank_model_initial_dist = hank_model_initial['steady_state']['distributions'].copy()

# Find perfect foresight transition to terminal steady state
x_transition, _ = hank_model_terminal.find_path(init_state = hank_model_initial_stst.values(),
                                                init_dist = hank_model_initial_dist)
    
#######################################################################
#######################################################################
#######################################################################
# Plot impulse responses to shock

# Fix horizon for plotting
horizon = 12

#######################################################################
# Aggregate dynamics

# If desired, plot transition of all aggregate variables (over entire horizon)       
plot_all(x_transition, hank_model_initial['variables'], 
          bunch=True, horizon=200)

# Plot transitions of selected aggregate variables
from list_variables_to_plot import dict_of_variables # Import lists of variables to plot
plot_selected_transition(dict_of_variables['aggregate'], hank_model_terminal, 
                         x_transition, horizon, 
                         save_results, exact_path, title=True)

#######################################################################
# Distributional dynamics

# Plot transitions of selected distributional variables
# plot_selected_transition(dict_of_variables['cross_sec'], hank_model_terminal, 
#                          x_transition, horizon, 
#                          save_results, exact_path, title=True)

# Plot policies on impact
# plot_policy_impact(hank_model_initial, hank_model_terminal, 
#                    x_transition,
#                    save_results, exact_path,
#                    borr_lim=shock_model_parameters['terminal_borrowing_limit'],
#                    x_threshold=150, borr_cutoff=False)

# plot_policy_impact(hank_model_initial, hank_model_terminal, 
#                    x_transition,
#                    save_results, exact_path,
#                    borr_lim=shock_model_parameters['terminal_borrowing_limit'],
#                    x_threshold=1, borr_cutoff=False)

#######################################################################
# Save transition as pickle for convenience
#store_transition(hank_model_terminal, x_transition, exact_path)