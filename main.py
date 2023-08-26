#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Andreas Koundouros
Date: 26.08.2023

This file contains the main code for my master thesis with the title:
    
   "Financial Constraints and Household Heterogeneity in the Macroeconomy", 

supervised by Prof. Dr. Keith Kuester and Dr. Gregor Böhl at the University of 
Bonn and submitted on 25th August 2023.

The code herein produces the core results of my thesis. Some of the results 
are used by other code files, such as compare_transitions.py. The code here
allows the user to select one or more of the possible combinations of models 
and shocks. Having set the respective choices, the code loads, adjusts and 
solves the initial and terminal models, computes the fully non-linear perfect-
foresight transition paths and prints various (hopefully) informative plots
and tables about the steady states and the transitions. If desired, these 
results are stored in the folder 'Results'. The transitions can also be stored
as pickle files in the end, for easy re-use by compare_transitions.py.
"""

###############################################################################
###############################################################################
###############################################################################
# Packages
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
# Imports

# Custom functions
from custom_functions import (get_model_path, # path to models
                              get_exact_results_path, # specific path for results
                              get_parametrisation, # obtain parameters
                              return_models_permanent, # final models
                              stst_overview, # table comparing steady states 
                              get_agg_and_dist_transitions_and_check_c, # obtain transition
                              # of aggregates and cross-section and check for 
                              # non-negativity of consumption
                              save_transition) # save transition as pickle

# Custom functions for plotting
from plot_functions import (plot_full_stst, # plot characteristics of single steady state
                            plot_compare_stst, # plot to compare steady states
                            plot_all, # quickly plot all variables 
                            plot_selected_transition, # plot selected variables 
                            visualise_dist_over_time, # plot distribution over time
                            plot_percentile_transitions, # plot responses by percentiles
                            plot_assets_on_impact_over_dist) # plot asset policy change 
                            # over the distribution

# Dictionary with variables to plot
from list_variables_to_plot import dict_of_variables_to_plot

###############################################################################
# Preliminaries
start = tm.time() # Start timer

save_results = False # True: save results 
show_titles_in_plots = True # True: show plot titles

pio.renderers.default = 'svg' # For plotting in the Spyder window

###############################################################################
###############################################################################
###############################################################################
# Settings

# Choose model(s)
models = ['baseline', # baseline model
          'slow_shock', # model with high persistence in shock processes
          'fast_shock', # model with low persistence in shock processes
          'end_L', # extended model with CRRA preferences and endogenous labour supply
          'very_slow_phi', # model with very high persistence in borrowing limit
          'no_ZLB', # model without zero-lower bound
          'low_B' # model with low calibration of liquid assets
          ]

# Choose shock(s)
shocks = ['limit_permanent', # permanent shock to the borrowing limit 
          'wedge_permanent', # permanent shock to the interest rate wedge
         ]

# Choose asymmetry 
asymmetry = False # True: credit easing, i.e. reverse transition

# Loop thorugh model-shock combinations to obtain results
for model in models:
    # Set model
    set_model = model
    
    for shock in shocks:
        # Set shock
        set_shock = shock
        
        # Dictionary of current settings
        settings = {'Model': set_model, 
                    'Shock': set_shock,
                    'Asymmetry': asymmetry}

        # Get path to model based on settings
        model_path = get_model_path(settings)
        
        # Get path for specific results
        exact_path = get_exact_results_path(settings)
        
        #######################################################################
        # Fix parameters
        
        # Get shock and model parameters based on settings (can be adjusted in 
        # the custom_functions.py file)
        shock_model_parameters = get_parametrisation(settings)
        
        #######################################################################
        # Print informative message
        print(f'You implement a shock of type \'{set_shock}\' in the model of type \'{set_model}\' with the following parameter values:', 
              shock_model_parameters)
        
        #######################################################################
        #######################################################################
        #######################################################################
        # Analysis of steady states and calculation of transition
        
        # Obtain initial and terminal models according to settings
        hank_model_initial, hank_model_terminal = return_models_permanent(model_path,
                                                                          settings,
                                                                          shock_model_parameters,
                                                                          asym = asymmetry)
        
        # Reset terminal borrowing limit in case of shock to interest rate 
        # wedge or in case of asymmetry
        if not settings['Shock'].startswith('limit') == True or asymmetry == True:
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
        
        # Plot to compare steady states
        plot_compare_stst(hank_model_initial, hank_model_terminal, settings,
                          save_results, exact_path,
                          shock_model_parameters['terminal_borrowing_limit'], 
                          x_threshold=30)
        
        #######################################################################
        # TRANSITION
        #######################################################################
        
        # Initial seady state and distribution as starting point of transition
        hank_model_initial_stst = hank_model_initial['stst'].copy()
        hank_model_initial_dist = hank_model_initial['steady_state']['distributions'].copy()
        
        # Get transition and check for negative values in individual 
        # consumption responses
        x_transition, dist_transition = get_agg_and_dist_transitions_and_check_c(hank_model_terminal,
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
        
        # Plot long-term debt dynamics separately
        plot_selected_transition(dict_of_variables_to_plot['debt'], 
                                 hank_model_terminal, 
                                 x_transition, 80, # long horizon
                                 save_results, 
                                 exact_path + '_' + 'long_run_debt', 
                                 title=show_titles_in_plots)
        
        #######################################################################
        # Distributional dynamics
        
        # Plot transitions of selected cross-sectional variables
        plot_selected_transition(dict_of_variables_to_plot['cross_sec'], 
                                 hank_model_terminal, 
                                 x_transition, horizon, 
                                 save_results, exact_path, 
                                 title=show_titles_in_plots)
        
        # Plot consumption responses by percentiles
        plot_percentile_transitions(['C', 'Consumption'], 
                                    hank_model_terminal, x_transition,
                                    [['Bot25C','Bottom-25%'], 
                                     ['Bot50C','Bottom-50%'], 
                                     ['Top25C','Top-25%']], horizon, 
                                    save_results, exact_path,
                                    title=show_titles_in_plots)
        
        # If applicable (hank_end_L), plot labour supply responses by percentiles 
        if settings['Model'] == 'end_L':
            plot_percentile_transitions(['N', 'Labour'], 
                                        hank_model_terminal, x_transition,
                                        [['Bot25N','Bottom-25%'], 
                                         ['Bot50N','Bottom-50%'], 
                                         ['Top25N','Top-25%']], horizon, 
                                        save_results, exact_path,
                                        title=show_titles_in_plots)
        
        # Plot asset changes over borrowers
        plot_assets_on_impact_over_dist(hank_model_initial,hank_model_terminal,
                                        dist_transition,
                                        save_results,exact_path,
                                        x_threshold=0, # focus on debt
                                        borr_lim=shock_model_parameters['terminal_borrowing_limit'])
        
        # Plot asset changes over entire distribution
        plot_assets_on_impact_over_dist(hank_model_initial,hank_model_terminal,
                                        dist_transition,
                                        save_results,exact_path)
        
        # Visualise asset distribution over time
        visualise_dist_over_time(hank_model_initial, hank_model_terminal,
                                 x_transition, dist_transition,
                                 horizon=30,
                                 y_threshold=10, x_threshold=-1.4) # focus close to borrowing limit
        
        #######################################################################
        # Save transition as pickle for convenience
        save_transition(hank_model_terminal, x_transition, 
                        save_results, exact_path)

###############################################################################
##############################################################################
###############################################################################
# Print run time
print(f'It took {round((tm.time()-start)/60, 2)} minutes to execute this script.')
