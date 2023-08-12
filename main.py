#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Andreas Koundouros
Date: 25.08.2023

This file contains the main code for my master thesis with the title:
    
   "Financial Constraints and Household Heterogeneity in the Macroeconomy", 

supervised by Prof. Dr. Keith Kuester and Dr. Gregor Böhl at the University of 
Bonn and submitted on 25th August 2023.

The code herein reproduces the results found in the thesis. It allows the user
to select one or more of the possible combinations of models and shocks. Having
set the respective choices, the code loads, adjusts and solves the initial
and terminal models, computes the fully non-linear perfect-foresight transition 
paths and plots various informative plots about the steady states and
transitions. If desired, the results are stored in the folder 'Results'.
"""

###############################################################################
###############################################################################
###############################################################################
# Import packages
import os # path management
import time as tm # timing
import plotly.io as pio # plot settings
    
###############################################################################
# Set working directory to folder 'master_thesis' to execute this file
if not os.getcwd().endswith('master-thesis'):
    raise Exception(f'Set working directory to folder \'master_thesis\'! Currently it is \'{os.getcwd()}\'')

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
                              get_agg_and_dist_transitions_and_check_c,
                              save_transition)

# Custom functions for plotting
from plot_functions import (plot_full_stst, 
                            plot_compare_stst,
                            plot_all,
                            plot_selected_transition,
                            visualise_dist_over_time,
                            plot_percentile_transitions_C)

# Import lists of variables to plot
from list_variables_to_plot import dict_of_variables_to_plot

###############################################################################
# Preliminaries
start = tm.time() # Start timer

save_results = True # True: save results (tables and plots)

show_titles_in_plots = False # True: show plot titles

pio.renderers.default = 'svg' # For plotting in the Spyder window

###############################################################################
###############################################################################
###############################################################################
# Settings

# Choose model(s)
models = ['baseline', # baseline model (section 3)
          'slow_shock', # baseline model with slow deleveraging (section 6.1)
          'fast_shock', # baseline model with fast deleveraging (section 6.1)
          'end_L', # extended model with endogenous labour supply (section 6.2)
          'no_ZLB', # baseline model with a low beta calibration (appendix E.1)
          'low_B' # baseline model with a low B calibration (appendix E.2)
          ]

# Choose shock(s)
shocks = ['limit_permanent', # permanent shock to the borrowing limit (section 4.1)
         'wedge_permanent', # permanent shock to the interest rate wedge (section 4.2)
         ]

# Choose asymmetry 
asymmetry = False # True: credit easing

# Loop thorugh model-shock combinations to obtain results
for model in models:
    # Set model
    set_model = model
    
    for shock in shocks:
        # Set shock
        set_shock = shock
        
        # Get current settings
        settings = {'Model': set_model, 
                    'Shock': set_shock,
                    'Asymmetry': asymmetry}

        # Get path to model based on settings
        model_path = get_model_path(settings)
        
        # Get path for saving results based on settings
        exact_path = get_exact_results_path(settings)
        
        #######################################################################
        # Fix parameters
        
        # Get shock and model parameters based on settings
        shock_model_parameters = get_parametrisation(settings)
        
        #######################################################################
        # Print informative message
        print(f'You implement a shock of type \'{set_shock}\' in the model of type \'{set_model}\' with the following parameter values:', 
              shock_model_parameters)
        
        #######################################################################
        #######################################################################
        #######################################################################
        # Analysis of steady states and calculation of impulse responses to shock
        
        # Make analysis according to which shock is chosen
        hank_model_initial, hank_model_terminal = return_models_permanent(model_path,
                                                                          settings,
                                                                          shock_model_parameters,
                                                                          asym = asymmetry)
        
        if not settings['Shock'].startswith('limit') == True:
            shock_model_parameters['terminal_borrowing_limit'] = None
        
        #######################################################################
        # STEADY STATES
        #######################################################################
        
        # Initial steady state
        _ = hank_model_initial.solve_stst() # Solve for steady state
        plot_full_stst(hank_model_initial, # Visualise steady state
                       settings, shock_model_parameters,
                       save_results, exact_path, 'initial')
        
        # Terminal steady state
        _ = hank_model_terminal.solve_stst() # Solve for steady state
        plot_full_stst(hank_model_terminal, # Visualise steady state
                       settings, shock_model_parameters,
                       save_results, exact_path, 'terminal', 
                       borr_cutoff=shock_model_parameters['terminal_borrowing_limit'])
        
        # Compare steady states
        stst_comparison = stst_overview([hank_model_initial, hank_model_terminal], 
                                        save_results, exact_path)
        print(stst_comparison)
        
        plot_compare_stst(hank_model_initial, hank_model_terminal, settings,
                          save_results, exact_path,
                          shock_model_parameters['terminal_borrowing_limit'], 
                          x_threshold=30)
        
        #######################################################################
        # TRANSITION
        #######################################################################
        
        # Initial seady state as starting point of transition
        hank_model_initial_stst = hank_model_initial['stst'].copy()
        hank_model_initial_dist = hank_model_initial['steady_state']['distributions'].copy()
        
        x_transition, _ = get_agg_and_dist_transitions_and_check_c(hank_model_terminal,
                                                                   hank_model_initial_stst,
                                                                   hank_model_initial_dist)
            
        #######################################################################
        #######################################################################
        #######################################################################
        # Plot impulse responses to shock
        
        # Fix horizon for plotting
        horizon = 12
        
        #######################################################################
        # Aggregate dynamics
        
        # If desired, plot transition of all aggregate variables     
        plot_all(x_transition, hank_model_initial['variables'], 
                  bunch=True, # Bunch plots together 
                  horizon=200) # Entire horizon to check convergence
        
        # Plot transitions of selected aggregate variables
        plot_selected_transition(dict_of_variables_to_plot['aggregate'], 
                                 hank_model_terminal, 
                                 x_transition, horizon, 
                                 save_results, exact_path, 
                                 title=show_titles_in_plots)
        
        #######################################################################
        # Distributional dynamics
        
        # Plot transitions of selected distributional variables
        plot_selected_transition(dict_of_variables_to_plot['cross_sec'], 
                                 hank_model_terminal, 
                                 x_transition, horizon, 
                                 save_results, exact_path, 
                                 title=show_titles_in_plots)

        # Plot asset distribution over time
        visualise_dist_over_time(hank_model_initial, hank_model_terminal,
                                 x_transition, horizon=30,
                                 y_threshold=10, 
                                 x_threshold=-1.4)
        
        # Plot consumption responses of certain percentiles
        plot_percentile_transitions_C(hank_model_terminal, x_transition,
                                      [['Bot25C','Bottom-25%'], 
                                       ['Bot50C','Bottom-50%'], 
                                       ['Top25C','Top-25%']], horizon, 
                                      save_results, exact_path,
                                      title=show_titles_in_plots)
        
        #######################################################################
        # Save transition as pickle for convenience
        save_transition(hank_model_terminal, x_transition, 
                        save_results, exact_path)

###############################################################################
##############################################################################
###############################################################################
# Print run time
print(f'It took {round((tm.time()-start)/60, 2)} minutes to execute this script.')
