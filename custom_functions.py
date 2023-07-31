#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Andreas Koundouros
Date: 15.05.2023

This file contains custom functions used throughout the project.
"""

###############################################################################
###############################################################################
# Import packages
import os # path management
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import econpizza as ep # Econpizza

###############################################################################
###############################################################################
# Function to obtain the path to the model
def get_model_path(settings):
    """Path to models.
    
    This function returns, for a chosen model, the full path to the model.
    
    Parameters:
    ----------
    settings            : dictionary which under 'Model' stores which model is
                          chosen ('baseline', 'end_L', 'low_beta', 'high_B', 
                                  'shock_beta')
    
    Returns:
    ----------
    full path to the model dictionary
    """
    # Extract chosen model from settings dictionary
    exact_model = settings['Model']
    
    # Depending on the chosen model, set the path to the model
    return os.path.join(os.getcwd(), 'Models', f'hank_{exact_model}.yml')

###############################################################################
###############################################################################
# Function to obtain a string for easy saving of results 
def get_exact_results_path(settings):
    # Get sub-components of string
    comp1 = settings['Model']
    comp2 = settings['Shock']
    
    # Return exact path to results  
    return f'{comp1}_{comp2}'

###############################################################################
###############################################################################
# Set model and shock parameters according to settings
def get_parametrisation(settings):
    # Parametrisation in case of baseline model of section 3
    if settings['Model'] == 'baseline' or settings['Model'] == 'slow_shock' or settings['Model'] == 'fast_shock':
        shock_model_parameters = {'initial_borrowing_limit': -2.353,
                                  'terminal_borrowing_limit': -2.18,
                                  'initial_wedge': 1e-8,
                                  'terminal_wedge': 0.00206,
                                  'terminal_beta': 0.992}
    
    # Parametrisation in case of extended model with CRRA preferences of section 6.2
    if settings['Model'] == 'end_L':
        shock_model_parameters = {'initial_borrowing_limit': -1.51,
                                  'terminal_borrowing_limit': -1.39,
                                  'initial_wedge': 1e-8,
                                  'terminal_wedge': 0.0025,
                                  'terminal_beta': 0.9906}
        
    # Parametrisation in case of baseline model with a low beta calibration
    if settings['Model'] == 'low_beta':
        shock_model_parameters = {'initial_borrowing_limit': -2.362,
                                  'terminal_borrowing_limit': -2.1883,
                                  'initial_wedge': 1e-8,
                                  'terminal_wedge': 0.0025,
                                  'terminal_beta': 0.9906}
    
    # Parametrisation in case of baseline model with a high B calibration
    if settings['Model'] == 'low_B':
        shock_model_parameters = {'initial_borrowing_limit': -1.55,
                                  'terminal_borrowing_limit': -1.425,
                                  'initial_wedge': 1e-8,
                                  'terminal_wedge': 0.002,
                                  'terminal_beta': 0.99}
        
    # Return shock and model parametrisation
    return shock_model_parameters

###############################################################################
###############################################################################
# Function to return two models, where the first one corresponds to the initial
# steady state and the second one corresponds to the terminal steady state
def return_models_permanent(model_path, 
                            settings, 
                            shock_model_parameters):
    # Get model as dictionary
    hank_dict = ep.parse(model_path)
    
    # Load models depending on the chosen shock and settings
    
    # Models for permanent shock to the borrowing limit
    if settings['Shock'].startswith('limit') == True:
        # Settings for interest rate wedge
        hank_dict['steady_state']['fixed_values']['Rbar'] = shock_model_parameters['initial_wedge']
        
        # Create model with initial borrowing limit
        init = shock_model_parameters['initial_borrowing_limit']
        term = shock_model_parameters['terminal_borrowing_limit']
        
        hank_dict['steady_state']['fixed_values']['phi'] = init
        hank_dict['definitions'] = hank_dict['definitions'].replace('amin = 0', f'amin = {init}')
        hank_dict['definitions'] = hank_dict['definitions'].replace('amin_terminal = 0', f'amin_terminal = {term}')
        
        # Load initial model
        hank_model_initial = ep.load(hank_dict)
        
        # Create model with terminal borrowing limit
        hank_dict['steady_state']['fixed_values']['phi'] = term
        
        # Load terminal model
        hank_model_terminal = ep.load(hank_dict)
        
    # Models for permanent shock to the interest rate wedge
    if settings['Shock'].startswith('wedge') == True:
        # Settings for borrowing limit
        init = shock_model_parameters['initial_borrowing_limit']
        term = shock_model_parameters['terminal_borrowing_limit']
        
        hank_dict['steady_state']['fixed_values']['phi'] = init
        hank_dict['definitions'] = hank_dict['definitions'].replace('amin = 0', f'amin = {init}')
        hank_dict['definitions'] = hank_dict['definitions'].replace('amin_terminal = 0', f'amin_terminal = {term}')
        
        # Create model with initial borrowing wedge
        hank_dict['steady_state']['fixed_values']['Rbar'] = shock_model_parameters['initial_wedge']
        
        # Load initial model
        hank_model_initial = ep.load(hank_dict)
        
        # Create model with terminal borrowing wedge
        hank_dict['steady_state']['fixed_values']['Rbar'] = shock_model_parameters['terminal_wedge']
        
        # Load terminal model
        hank_model_terminal = ep.load(hank_dict)
    
    # Models for permanent shock to the discount factor
    if settings['Shock'].startswith('beta') == True:
        # Settings for borrowing limit
        init = shock_model_parameters['initial_borrowing_limit']
        term = shock_model_parameters['terminal_borrowing_limit']
        
        hank_dict['steady_state']['fixed_values']['phi'] = init
        hank_dict['definitions'] = hank_dict['definitions'].replace('amin = 0', f'amin = {init}')
        hank_dict['definitions'] = hank_dict['definitions'].replace('amin_terminal = 0', f'amin_terminal = {term}')
        
        # Settings for interest rate wedge
        hank_dict['steady_state']['fixed_values']['Rbar'] = shock_model_parameters['initial_wedge']
        
        # Load initial model
        hank_model_initial = ep.load(hank_dict)
        
        # Create model with terminal borrowing wedge
        hank_dict['steady_state']['fixed_values']['beta'] = shock_model_parameters['terminal_beta']
        
        # Load terminal model
        hank_model_terminal = ep.load(hank_dict)
        
    # Return initial model and terminal model
    return hank_model_initial, hank_model_terminal

###############################################################################
###############################################################################
def return_models_transitory(model_path, 
                            settings, 
                            shock_model_parameters):
    # Get model as dictionary
    hank_dict = ep.parse(model_path)
    
    # Load model depending on the chosen shock
    
    hank_dict['steady_state']['fixed_values']['Rbar'] = shock_model_parameters['initial_wedge']
    
    init = shock_model_parameters['initial_borrowing_limit']
    term = shock_model_parameters['terminal_borrowing_limit']
    
    hank_dict['steady_state']['fixed_values']['phi'] = init
    hank_dict['definitions'] = hank_dict['definitions'].replace('amin = 0', f'amin = {init}')
    hank_dict['definitions'] = hank_dict['definitions'].replace('amin_terminal = 0', f'amin_terminal = {term}')
    
    hank_model = ep.load(hank_dict)
        
    # Return model
    return hank_model

###############################################################################
###############################################################################
# Function to find the closest on-grid asset grid point for a given off-grid 
# value
def find_closest_grid_point(ar_borrowing_limit, 
                            asset_grid):
    # Create an array of distances between the grid points and a given value
    array_distances = jnp.abs(asset_grid - ar_borrowing_limit)
    
    # Get the index of the minimum distance in array 
    indx_min_distance = jnp.argmin(array_distances)
    
    # Identify the grid point with the minimum distance to the given value
    closest_grid_point = asset_grid[indx_min_distance]
    
    # Return the closest grid point and its index in the asset grid
    return closest_grid_point, indx_min_distance

###############################################################################
###############################################################################
# Function to get policies as data frane
def make_policy_df(hank_model,
                   policy,
                   borr_cutoff=None,
                   x_threshold=None):
    # Preliminaries
    a_grid = hank_model['context']['a_grid'] # Get asset grid
    policy_arr = np.array(a_grid) # Initialise container for policies
    policy_columns = ['grid'] # Initialise name constainer
    
    # Loop through skill grid to get policy functions for each skill level
    for no_states in range(hank_model['distributions']['dist']['skills']['n']): # Note: here, n refers to the number of skill grid points
        one_policy = hank_model['steady_state']['decisions'][f'{policy}'][no_states]
        
        if policy_arr.size == 0:
            policy_arr = one_policy
        else:
            policy_arr = np.column_stack((policy_arr, one_policy))
            
        policy_columns.append(f'\u03B8_{no_states}')
    
    # Create data frame
    policy_df = pd.DataFrame(policy_arr, columns = policy_columns)
    
    # Cut off x axis at borrowing limit
    if borr_cutoff != None:
        policy_df.loc[policy_df['grid'] < borr_cutoff, :] = np.nan
    
    # Cut off x axis at threshold
    if x_threshold != None:
        policy_df.loc[policy_df['grid'] > x_threshold, :] = np.nan
    
    # Return resulting data frame
    return policy_df

###############################################################################
###############################################################################
# Function to compare initial and terminal steady state
def stst_overview(models,
                  save_results, 
                  exact_path,
                  percent=100):
    """Steady state overview.
    
    This function creates a data frame which holds the steady state values of
    the one or two models passed to the function.
    
    If one model is passed, the function creates a table with the steady state 
    values.
    
    If two models are passed, the function creates a table with both steady 
    states side-by-side and adds a column which calculates the difference 
    between the two steady states. It can also distinguish (manually) between 
    variables for which it needs to take the change in absolute terms or in 
    percentage terms.
    """
    # Create a rounding function for convenience
    round_func_4 = lambda x: round(float(x), 4)
        
    # Distinguish between the case of one model...:
    if len(models) == 1:
        model = models[0]
        stst_df = pd.DataFrame(model['stst'].items(), 
                                    columns = ['Variable', 'Steady State'])
        stst_df['Steady State'] = stst_df['Steady State'].apply(round_func_4)
        
        # Add some more features of the steady states
        a_grid = model['context']['a_grid']
        
        # Distributions
        distribution_skills_and_assets = model['steady_state']['distributions'][0]
        distribution_assets = 100*jnp.sum(distribution_skills_and_assets, 
                                          axis = 0)
        
        # Add fraction of indebted households
        row_share_indebted = {'Variable': 'Frac. of Borrowers',
                              'Steady State': jnp.sum(jnp.where(a_grid < 0, 
                                                                        distribution_assets, 
                                                                        0)).round(2).item()}
        row_share_indebted_df = pd.DataFrame([row_share_indebted])
        stst_df = pd.concat([stst_df, row_share_indebted_df], 
                                 ignore_index=True)
        
        # Add fraction of households at borrowing limit    
        row_share_limit = {'Variable': 'Frac. at Borrowing Limit',
                           'Steady State': distribution_assets[0].round(2).item()}
        row_share_limit_df = pd.DataFrame([row_share_limit])
        stst_df = pd.concat([stst_df, row_share_limit_df], 
                                       ignore_index=True)
        
        # Add fraction of households at 0 assets 
        row_share_zero = {'Variable': 'Frac. at Zero Assets',
                          'Steady State': jnp.sum(jnp.where(a_grid == 0, 
                                                            distribution_assets, 
                                                            0)).round(2).item()}
        row_share_zero_df = pd.DataFrame([row_share_zero])
        stst_df = pd.concat([stst_df, row_share_zero_df], 
                                       ignore_index=True)
    
    # ...and case of two models:
    elif len(models) == 2:
        hank_model_init = models[0]
        hank_model_term = models[1]
        
        stst_df = pd.DataFrame(hank_model_init['stst'].items(), 
                                    columns = ['Variable', 'Initial'])
        stst_df['Initial'] = stst_df['Initial'].apply(round_func_4)
        
        term_stst_df = pd.DataFrame(hank_model_term['stst'].items(),
                                             columns = ['Variable', 'Terminal'])
        term_stst_df['Terminal'] = term_stst_df['Terminal'].apply(round_func_4)
        
        # Merge steady states into one data frame
        stst_df = pd.merge(stst_df, term_stst_df, 
                           on = 'Variable', how = 'left')
        
        # Add some more features of the steady states
        a_grid_init = hank_model_init['context']['a_grid']
        
        # Initial steady state 
        distribution_skills_and_assets_initial = hank_model_init['steady_state']['distributions'][0]
        distribution_assets_initial = percent*jnp.sum(distribution_skills_and_assets_initial, 
                                          axis = 0)
        
        # Terminal steady state 
        distribution_skills_and_assets_terminal = hank_model_term['steady_state']['distributions'][0]
        distribution_assets_terminal = percent*jnp.sum(distribution_skills_and_assets_terminal, 
                                          axis = 0)
        
        # Add fraction of indebted households
        row_share_indebted = {'Variable': 'Frac. of Borrowers',
                              'Initial': jnp.sum(jnp.where(a_grid_init < 0, 
                                                                        distribution_assets_initial, 
                                                                        0)).round(2).item(),
                              'Terminal': jnp.sum(jnp.where(a_grid_init < 0, 
                                                                        distribution_assets_terminal, 
                                                                        0)).round(2).item()}
        row_share_indebted_df = pd.DataFrame([row_share_indebted])
        stst_df = pd.concat([stst_df, row_share_indebted_df], 
                                       ignore_index=True)
        
        # Add fraction of households at borrowing limit    
        row_share_limit = {'Variable': 'Frac. at Borrowing Limit',
                           'Initial': distribution_assets_initial[0].round(2).item(),
                           'Terminal': distribution_assets_terminal[distribution_assets_terminal>0][0].round(2).item()}
        row_share_limit_df = pd.DataFrame([row_share_limit])
        stst_df = pd.concat([stst_df, row_share_limit_df], 
                                       ignore_index=True)
        
        # Add fraction of households at 0 assets 
        row_share_zero = {'Variable': 'Frac. at Zero Assets',
                          'Initial': jnp.sum(jnp.where(a_grid_init == 0, 
                                                                    distribution_assets_initial, 
                                                                    0)).round(2).item(),
                          'Terminal': jnp.sum(jnp.where(a_grid_init == 0, 
                                                                     distribution_assets_terminal, 
                                                                     0)).round(2).item()}
        row_share_zero_df = pd.DataFrame([row_share_zero])
        stst_df = pd.concat([stst_df, row_share_zero_df], 
                                       ignore_index=True)
        
        # Add column which calculates changes in percent between the steady states
        stst_df['Change'] = 0

        # Calculate changes based on variable type
        for index, row in stst_df.iterrows():
            try:
                if row['Variable'] in ['beta', 'tau', 'D', 'DY', 'gr_liquid', 'phi', 'MPC', 'R', 'Rbar', 'Rn', 'Rr', 'Rminus', 'Frac. of Borrowers', 'Frac. at Borrowing Limit', 'Frac. at Zero Assets']:
                    # Absolute change for specific variables
                    stst_df.at[index, 'Change'] = row['Terminal'] - row['Initial']
                else:
                    # Percentage change for other variables
                    stst_df.at[index, 'Change'] = percent * ((row['Terminal'] - row['Initial']) / row['Initial'])
            # In case of division by zero, insert NaN
            except ZeroDivisionError:
                stst_df.at[index, 'Change'] = np.nan

        # Round changes
        stst_df['Change'] = stst_df['Change'].apply(round_func_4)
    
    # Save table in TeX
    if save_results == True:
        stst_table_path = os.path.join(os.getcwd(),
                                       'Results',
                                       f'stst_comparison_{exact_path}.tex')
        stst_df.to_latex(stst_table_path, 
                         label = f'tab:stst_comparison_{exact_path}', 
                         index = False)
    
    # Return resulting data frame
    return stst_df

###############################################################################
###############################################################################
# Function to pickle transitions
def store_transition(model, 
                     x_trans, 
                     exact_path):
    # Transform JAX array into Numpy array
    nparr = jax.device_get(x_trans)
    
    # Transform into Pandas data frame
    transition_df = pd.DataFrame(nparr)
    
    # Give columns names 
    transition_df.columns = model['variables']
    
    # Get path to save pickle file
    path = os.path.join(os.getcwd(),
                        'Results',
                        f'x_trans_{exact_path}.pkl')
    
    # Save data frame as pickle
    transition_df.to_pickle(path)

###############################################################################
###############################################################################
# Function to obtain pickled transitions
def get_transitions(comparison):
    # Get first transition
    transition_1 = comparison['transition_1']
    path_1 = os.path.join(os.getcwd(),
                          'Results',
                          f'x_trans_{transition_1}.pkl')
    try:
        x_1_df = pd.read_pickle(path_1)
    except FileNotFoundError:
        raise FileNotFoundError(f'No transition yet saved under the name {transition_1}.')
    
    # Get second transition
    transition_2 = comparison['transition_2']
    path_2 = os.path.join(os.getcwd(),
                          'Results',
                          f'x_trans_{transition_2}.pkl')
    try:
        x_2_df = pd.read_pickle(path_2)
    except FileNotFoundError:
        raise FileNotFoundError(f'No transition yet saved under the name {transition_2}.')
    
    # Return the data frames of the chosen transitions
    return x_1_df, x_2_df


def check_for_negative_entries(array_impl_obj):
    # Convert input into a numpy array
    array = jnp.asarray(array_impl_obj)

    # Check for negative entries
    negative_indices = jnp.argwhere(array < 0)

    if negative_indices.size > 0:
        # Collect all negative value occurrences
        negative_values = [(index[0], index[1], index[2], array[tuple(index)]) for index in negative_indices]

        # Print warning message
        warning_messages = ['Warning: Negative value(s) found.' for index, _, _, value in negative_values]
        for warning_message in warning_messages:
            print(warning_message)

        return negative_values
    
    else:
        print('No negative values found in the array.')
        return None