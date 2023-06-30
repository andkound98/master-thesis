#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Andreas Koundouros
Date: 21.06.2023

This file runs the main codes of my master thesis.
"""

###############################################################################
###############################################################################
# Import packages
import os # path creation
import pickle # saving files
import time as tm # timing
#import econpizza as ep # econpizza
#import pandas as pd # data wrangling
#import numpy as np # data wrangling
#import jax.numpy as jnp # data wrangling
import plotly.io as pio # plotting
#import matplotlib.pyplot as plt # plotting

###############################################################################
###############################################################################
# Set working directory
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
from custom_functions import (get_model, 
                              path_to_results, 
                              return_models_permanent,
                              return_models_transitory,
                              stst_overview)

from plot_functions import (plot_full_stst, 
                            plot_compare_stst,
                            #plot_all,
                            plot_selected_transition,
                            plot_impact_distr)

###############################################################################
###############################################################################
# Preliminaries
start = tm.time() # Start timer

save_results = True # If true, results (tables and plots) are stored

pio.renderers.default = "svg" # For plotting in the Spyder window

###############################################################################
###############################################################################
# Choose model

# Set 'baseline' to True for the baseline one-asset HANK model w/o endogenous 
# labour supply and to False for the one-asset HANK model with endogenous 
# labour supply
baseline = True

# Get path to model and model type
full_path_hank, model_type = get_model(full_path_code, baseline)

###############################################################################
###############################################################################
# Choose shock

# Set 'shock_to_borrowing_constraint' to True for a shock to the borrowing 
# constraint and to False for a shock to the interest rate wedge
shock_to_borrowing_constraint = False 

# Set 'shock_permanent' to True for a permanent shock and to False for a 
# transitory shock
shock_permanent = True 

# Fix path to results according to which shock is chosen
full_path_results = path_to_results(full_path_code, 
                                    model_type, 
                                    shock_to_borrowing_constraint, 
                                    shock_permanent)

###############################################################################
###############################################################################
# Fix settings of shocks (note that some choices affect also the other shock)

# Permanent shocks

# Settings of shock to borrowing limit
persistence_borrowing_limit = 0.3
initial_borrowing_limit = -1.025  #-0.6 
terminal_borrowing_limit = -0.76 #-0.45

# Settings of shock to interest rate wedge
persistence_borrowing_wedge = 0.3
initial_wedge = 1e-8
terminal_wedge = 0.01

# Transitory shocks

#### TO DO

###############################################################################
###############################################################################
# Analysis of steady states and calculation of impulse responses to shock

# Make analysis according to which shock is chosen
if shock_permanent == True: # Case of permanent shock
    hank_model_initial, hank_model_terminal = return_models_permanent(full_path_hank, 
                                                                      shock_to_borrowing_constraint, 
                                                                      persistence_borrowing_limit,
                                                                      initial_borrowing_limit,
                                                                      terminal_borrowing_limit,
                                                                      persistence_borrowing_wedge,
                                                                      initial_wedge,
                                                                      terminal_wedge)
    
    if shock_to_borrowing_constraint == False:
        terminal_borrowing_limit = initial_borrowing_limit
    
    # hank_dict = ep.parse('/Users/andreaskoundouros/Documents/Uni-Masterarbeit/master-thesis/Code/hank_baseline_init.yml')
    # hank_model_initial = ep.load(hank_dict)
    # hank_dict = ep.parse('/Users/andreaskoundouros/Documents/Uni-Masterarbeit/master-thesis/Code/hank_baseline_term.yml')
    # hank_model_terminal = ep.load(hank_dict)
    
    # Initial steady state
    _ = hank_model_initial.solve_stst() # Solve for stst
    plot_full_stst(hank_model_initial, # Visualise stst
                   model_type, save_results, full_path_results, 'initial')
    
    # Terminal steady state
    _ = hank_model_terminal.solve_stst() # Solve for stst
    plot_full_stst(hank_model_terminal, # Visualise stst
                   model_type, save_results, full_path_results, 'terminal',
                   borr_limit=terminal_borrowing_limit)
    
    # Compare steady states
    stst_comparison = stst_overview([hank_model_initial, hank_model_terminal], 
                                    save_results, full_path_results)
    print(stst_comparison)
    plot_compare_stst(hank_model_initial, hank_model_terminal,
                      save_results, full_path_results, 
                      terminal_borrowing_limit, x_threshold=15)
    
    # Initial seady state as starting point of transition
    hank_model_initial_stst = hank_model_initial['stst']

    # Find perfect foresight transition to terminal steady state
    x_transition, _ = hank_model_terminal.find_path(init_state = hank_model_initial_stst.values())
    
elif shock_permanent == False: # Case of transitory shock
    hank_model_terminal = return_models_transitory(full_path_hank, 
                                                   shock_to_borrowing_constraint, 
                                                   persistence_borrowing_limit,
                                                   initial_borrowing_limit,
                                                   terminal_borrowing_limit,
                                                   persistence_borrowing_wedge,
                                                   initial_wedge)
    
    # Steady state
    _ = hank_model_terminal.solve_stst()
    plot_full_stst(hank_model_terminal, # Visualise stst
                   model_type, save_results, full_path_results, 'zero')
    
    # Print steady state
    stst = stst_overview([hank_model_terminal], 
                         save_results, full_path_results)
    print(stst)
    
    # Fix shock 
    if shock_to_borrowing_constraint == True:
        shock = ('e_a', -0.3) # (ONLY NEGATIVE VALUES POSSIBLE due to interpolation)
        
        
    elif shock_to_borrowing_constraint == False:
        shock = ('e_Rbar', 0.9)

    # Find perfect foresight transition to new steady state
    x_transition, _ = hank_model_terminal.find_path(shock)
    
    #x_transition, _ = hank_model_terminal.find_path(init_state=x0.values())

###############################################################################
###############################################################################
# Plot impulse responses to shock

# Fix horizon for plotting
horizon = 12

###############################################################################
###############################################################################
# Aggregate dynamics

# If desired, plot transition of all aggregate variables       
#plot_all(x_transition, hank_model_initial['variables'], horizon)

# Select variables to plot (with descriptions)
variables_to_plot = [['C', 'Consumption'], 
                     ['y', 'GDP'], 
                     ['y_prod', 'Production'], 
                     ['mc', 'Marginal Costs'],
                     ['pi', 'Inflation'], 
                     ['w', 'Wage'], 
                     ['borr_limit', 'Borrowing Limit'],
                     ['Rbar', 'Interest Rate Wedge'],
                     ['Rrminus', 'Interest Rate on Neg. Assets'],
                     ['R', 'Nominal Interest Rate', 'Rn', 'Notional Interest Rate'], 
                     ['Rr', 'Ex-Post Real Interest Rate'], 
                     ['div', 'Dividends'],
                     ['tau', 'Taxes'],
                     ['D', 'Household Debt'], 
                     ['DY', 'Household Debt-to-GDP'],
                     ['gr_liquid', 'Gross Liquidity']]

# Depending on the specific HANK model, add aggregate labour hours
if model_type == 'Baseline':
    variables_to_plot.append(['n', 'Labour Hours'])
elif model_type == 'End_labour':
    variables_to_plot.append(['N', 'Labour Hours'])

# Plot transition of some selected variables
plot_selected_transition(variables_to_plot, hank_model_initial, x_transition, 
                         horizon, save_results, full_path_results)

###############################################################################
###############################################################################
# Distributional dynamics

# Plot consumption on impact over wealth distribution
plot_impact_distr(hank_model_terminal, x_transition, 'c', 'Consumption', 
                  save_results, full_path_results, 
                  x_threshold=0, borr_cutoff=True, 
                  borr_lim=terminal_borrowing_limit)

plot_impact_distr(hank_model_terminal, x_transition, 'a', 'Assets', 
                  save_results, full_path_results, 
                  x_threshold=0, borr_cutoff=True, 
                  borr_lim=terminal_borrowing_limit)

##############################################################################
###############################################################################
# Save transition as pickle for later convenience
with open(os.path.join(full_path_results, 'x_trans.pkl'), 'wb') as file:
    pickle.dump(x_transition, file)

##############################################################################
###############################################################################
# Print run time
print('It took', round((tm.time()-start)/60, 2), 
      'minutes to execute this script.')
