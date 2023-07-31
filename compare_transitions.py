#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Andreas Koundouros
Date: 21.07.2023

This file creates plots that compare the transitions produced by different
models and shocks.
"""

###############################################################################
###############################################################################
###############################################################################
# Import packages
import os # path management
import time as tm # timing
import plotly.io as pio # plotting

###############################################################################
# Set working directory to folder 'master_thesis' to execute this file
if not os.getcwd().endswith('master-thesis'):
    raise Exception('Set working directory to folder \'master_thesis\'!')
    
###############################################################################
###############################################################################
###############################################################################
# Import functions

# Custom functions for various purposes
from custom_functions import get_transitions

# Custom functions for plotting
from plot_functions import compare_selected_transition

###############################################################################
# Preliminaries
start = tm.time() # Start timer

save_results = True # True: results (tables and plots) are saved

pio.renderers.default = 'svg' # For plotting in the Spyder window

###############################################################################
###############################################################################
###############################################################################
# Choose comparison

# Choose model-shock combinations to comapre
### NOTE: in order to comapre the transition of shocks, you must have 
# implemented them first through the main.py file
comparison = {'transition_1': 'baseline_limit_permanent',
              'transition_2': 'baseline_wedge_permanent'} 

# Set informative legend names for plots
#legend = ['Perm. Shock to \u03C6', 'Perm. Shock to \u03B2']
legend = ['Perm. Shock to \u03C6', 'Perm. Shock to \u03BA']

# Get transitions from data frames as pickles
x_1, x_2 = get_transitions(comparison)

# Fix horizon for plotting
horizon = 12

# Select variables to plot (with descriptions)
variables_to_plot = [['phi', 'Borrowing Limit'],
                     ['Rbar', 'Interest Rate Wedge'],
                     ['beta', 'Discount Factor'],
                     ['C', 'Consumption'], 
                     ['y', 'GDP'], 
                     ['y_prod', 'Production'], 
                     ['N', 'Labour Hours'],
                     ['w', 'Wage'], 
                     ['mc', 'Marginal Costs'],
                     ['pi', 'Inlfation'],
                     ['Rr', 'Ex-Post Real Interest Rate'],
                     ['Rrminus', 'Interest Rate on Neg. Assets'],
                     ['R', 'Nominal Interest Rate'], 
                     ['Rn', 'Notional Interest Rate'],  
                     ['div', 'Dividends'],
                     ['tau', 'Taxes'],
                     ['D', 'Household Debt'], 
                     ['DY', 'Household Debt-to-GDP'],
                     ['gr_liquid', 'Gross Liquidity']#,
                     # ['Top10C', 'Top10% Consumption Share'],
                     # ['Bot25C', 'Bottom25% Consumption Share'],
                     # ['Top10A', 'Top10% Assets Share'],
                     # ['Bot25A', 'Bottom25% Assets Share']
                     ]

# Plot the comparison of transitions of selected variables
compare_selected_transition(variables_to_plot, x_1, x_2, 
                            horizon, legend,
                            save_results, comparison)
