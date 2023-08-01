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
from custom_functions import (get_transitions,
                              get_labels)

# Custom functions for plotting
from plot_functions import compare_selected_transitions

###############################################################################
# Preliminaries
start = tm.time() # Start timer

save_results = False # True: results (tables and plots) are saved

pio.renderers.default = 'svg' # For plotting in the Spyder window

###############################################################################
###############################################################################
###############################################################################
# Choose model-shock combinations to comapre

### NOTE: in order to comapre the transition of shocks, you must have 
# implemented them first through the main.py file
comparison = {'transition_1': 'baseline_limit_permanent',
              'transition_2': 'end_L_limit_permanent'} # baseline_wedge_permanent

# Get informative legend names for plots
legend = get_labels(comparison)

# Get transitions from data frames as pickles
list_of_transitions = get_transitions(comparison)

# Fix horizon for plotting
horizon = 12

# Select variables to plot (with descriptions)
from list_variables_to_plot import dict_of_variables # Import lists of variables to plot

# Plot the comparison of transitions of selected variables
compare_selected_transitions(list_of_transitions,
                             dict_of_variables['aggregate'],
                             horizon,
                             legend,
                             save_results,
                             comparison,
                             title=True)
