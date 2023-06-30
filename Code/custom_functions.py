#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Andreas Koundouros
Date: 15.05.2023

This file contains some custom functions used throughout the project.
"""

###############################################################################
###############################################################################
# Import packages
import os
import pandas as pd
import numpy as np
import jax.numpy as jnp
import econpizza as ep

###############################################################################
###############################################################################
def get_model(full_path_code, baseline):
    """Paths to models.
    
    This function returns, for the given path where the models are and the 
    type of model chosen the full path to the model and the model type. The 
    latter is useful for creating paths to the results.
    
    
    Parameters:
    ----------
    full_path_code          : full path to the codes
    baseline                : True if baseline HANK w/o endogenous labour 
                              supply is chosen and False if the HANK w/ 
                              endogenous labour supply is chosen
    
    
    Returns:
    ----------
    full_path_hank          : full path to the model dictionary
    model_type              : string indicating whether the model is the 
                              baseline model or the one w/ endogenous labour
                              supply
    """
    os.chdir('..') 
    full_path_ms = os.getcwd()
    
    # Baseline model
    if baseline == True:
        full_path_hank = os.path.join(full_path_ms, 
                                      'Models', 
                                      'hank_baseline.yml')
        model_type = 'Baseline'
    
    # Model with endogenous labour supply
    elif baseline == False:
        full_path_hank = os.path.join(full_path_ms, 
                                      'Models', 
                                      'hank_end_labour.yml')
        model_type = 'End_labour'
        
    os.chdir(full_path_code) # Reset the working directory correctly 
        
    # Return path to model and model type
    return full_path_hank, model_type

###############################################################################
###############################################################################
# Function to create path to results depending on the model and on the shock
def path_to_results(full_path_code,
                    model_type,
                    shock_to_borrowing_constraint,
                    shock_permanent):
    os.chdir('..') 
    full_path_ms = os.getcwd()
    
    # Differentiate according to chosen model and shock
    if shock_to_borrowing_constraint == True:
        if shock_permanent == True:
            relative_path_results = os.path.join('Results', 
                                                 model_type,
                                                 'Shock_Limit',
                                                 'Permanent')
        elif shock_permanent == False:
            relative_path_results = os.path.join('Results', 
                                                 model_type,
                                                 'Shock_Limit',
                                                 'Transitory')
            
    if shock_to_borrowing_constraint == False:
        if shock_permanent == True:
            relative_path_results = os.path.join('Results', 
                                                 model_type,
                                                 'Shock_Wedge',
                                                 'Permanent')
        elif shock_permanent == False:
            relative_path_results = os.path.join('Results', 
                                                 model_type,
                                                 'Shock_Wedge',
                                                 'Transitory')

    full_path_results = os.path.join(full_path_ms, relative_path_results)
    os.chdir(full_path_code) # Reset the working directory correctly 
    
    # Return path to results
    return full_path_results

###############################################################################
###############################################################################
def return_models_permanent(full_path_hank, 
                            shock_to_borrowing_constraint, 
                            persistence_borrowing_limit=None,
                            initial_borrowing_limit=None,
                            terminal_borrowing_limit=None,
                            persistence_borrowing_wedge=None,
                            initial_wedge=None,
                            terminal_wedge=None):
    """Models with Permanent Shocks.
    
    This function returns two models, one with the initial conditions and one
    with the final conditions, depending on the shock chosen (shock to the 
    borrowing limit or to the interest rate wedge) and depending on parameters.
    
    Parameters:
    ----------
    full_path_hank          : full path to the model dictionary
    
    
    Returns:
    ----------
    hank_model_initial      : initial model
    hank_model_terminal     : terminal model
    """
    # Get model as dictionary
    hank_dict = ep.parse(full_path_hank)
    
    if shock_to_borrowing_constraint == True:
        # Settings for interest rate wedge
        hank_dict['steady_state']['fixed_values']['rho_Rbar'] = persistence_borrowing_wedge
        hank_dict['steady_state']['fixed_values']['Rbar'] = initial_wedge
        
        # Set persistence of shock to the borrowing limit
        hank_dict['steady_state']['fixed_values']['rho_a'] = persistence_borrowing_limit
        hank_dict['definitions'] = hank_dict['definitions'].replace('rho_a = 0.3', f'rho_a = {persistence_borrowing_limit}')
        
        # Create model with initial borrowing limit
        hank_dict['steady_state']['fixed_values']['borr_limit'] = initial_borrowing_limit
        hank_dict['definitions'] = hank_dict['definitions'].replace('amin = 0', f'amin = {initial_borrowing_limit}')
        hank_dict['definitions'] = hank_dict['definitions'].replace('amin_terminal = 0', f'amin_terminal = {terminal_borrowing_limit}')
        
        # Load initial model
        hank_model_initial = ep.load(hank_dict)

        # Create model with terminal borrowing limit
        hank_dict['steady_state']['fixed_values']['borr_limit'] = terminal_borrowing_limit
        
        ############
        ############ If desired, set new bond supply in terminal steady state
        #hank_dict['steady_state']['fixed_values']['B'] = 1.65
        
        # Load terminal model
        hank_model_terminal = ep.load(hank_dict)
    
    elif shock_to_borrowing_constraint == False:
        # Settings for borrowing limit
        hank_dict['steady_state']['fixed_values']['rho_a'] = persistence_borrowing_limit
        hank_dict['definitions'] = hank_dict['definitions'].replace('rho_a = 0.3', f'rho_a = {persistence_borrowing_limit}')
        hank_dict['steady_state']['fixed_values']['borr_limit'] = initial_borrowing_limit
        hank_dict['definitions'] = hank_dict['definitions'].replace('amin = 0', f'amin = {initial_borrowing_limit}')
        hank_dict['definitions'] = hank_dict['definitions'].replace('amin_terminal = 0', f'amin_terminal = {terminal_borrowing_limit}')
        
        # Set persistence of shock to the interest rate wedge
        hank_dict['steady_state']['fixed_values']['rho_Rbar'] = persistence_borrowing_wedge
        
        # Create model with initial borrowing wedge
        hank_dict['steady_state']['fixed_values']['Rbar'] = initial_wedge
        
        # Load initial model
        hank_model_initial = ep.load(hank_dict)

        # Create model with terminal borrowing wedge
        hank_dict['steady_state']['fixed_values']['Rbar'] = terminal_wedge
        
        # Load terminal model
        hank_model_terminal = ep.load(hank_dict)
        
    # Return initial and terminal models
    return hank_model_initial, hank_model_terminal

###############################################################################
###############################################################################
def return_models_transitory(full_path_hank, 
                             shock_to_borrowing_constraint, 
                             persistence_borrowing_limit=None,
                             stst_borrowing_limit=None,
                             terminal_borrowing_limit=None,
                             persistence_borrowing_wedge=None,
                             stst_wedge=None):
    # Get model as dictionary
    hank_dict = ep.parse(full_path_hank)
    
    if shock_to_borrowing_constraint == True:
        # Settings for interest rate wedge
        hank_dict['steady_state']['fixed_values']['rho_Rbar'] = persistence_borrowing_wedge
        hank_dict['steady_state']['fixed_values']['Rbar'] = stst_wedge
        
        # Set persistence of shock to the borrowing limit
        hank_dict['steady_state']['fixed_values']['rho_a'] = persistence_borrowing_limit
        hank_dict['definitions'] = hank_dict['definitions'].replace('rho_a = 0.3', f'rho_a = {persistence_borrowing_limit}')
        
        # Create model with initial borrowing limit
        hank_dict['steady_state']['fixed_values']['borr_limit'] = stst_borrowing_limit
        hank_dict['definitions'] = hank_dict['definitions'].replace('amin = 0', f'amin = {stst_borrowing_limit}')
        hank_dict['definitions'] = hank_dict['definitions'].replace('amin_terminal = 0', f'amin_terminal = {terminal_borrowing_limit}')
        
        # Load model
        hank_model = ep.load(hank_dict)
        
    elif shock_to_borrowing_constraint == False:
        # Settings for borrowing limit
        hank_dict['steady_state']['fixed_values']['rho_a'] = persistence_borrowing_limit
        hank_dict['definitions'] = hank_dict['definitions'].replace('rho_a = 0.3', f'rho_a = {persistence_borrowing_limit}')
        hank_dict['steady_state']['fixed_values']['borr_limit'] = stst_borrowing_limit
        hank_dict['definitions'] = hank_dict['definitions'].replace('amin = 0', f'amin = {stst_borrowing_limit}')
        hank_dict['definitions'] = hank_dict['definitions'].replace('amin_terminal = 0', f'amin_terminal = {terminal_borrowing_limit}')
        
        # Set persistence of shock to the interest rate wedge
        hank_dict['steady_state']['fixed_values']['rho_Rbar'] = persistence_borrowing_wedge
        
        # Create model with initial borrowing wedge
        hank_dict['steady_state']['fixed_values']['Rbar'] = stst_wedge
        
        # Load initial model
        hank_model = ep.load(hank_dict)
    
    # Return the model
    return hank_model

###############################################################################
###############################################################################
# Function to find the closest on-grid asset grid point for a given off-grid 
# value
def find_closest_grid_point(ar_borrowing_limit, 
                            asset_grid):
    array_distances = jnp.abs(asset_grid - ar_borrowing_limit)
    
    indx_min_distance = jnp.argmin(array_distances)
    
    closest_grid_point = asset_grid[indx_min_distance]
    
    return closest_grid_point, indx_min_distance

###############################################################################
###############################################################################
# Function to get policies as data frane
def make_policy_df(model,
                   policy,
                   borr_cutoff=None,
                   x_threshold=None):
    # Preliminaries
    a_grid = model['context']['a_grid'] # Get asset grid
    policy_arr = np.array(a_grid) # Initialise container for policies
    policy_columns = ['grid'] # Initialise name constainer
    
    # Loop through skill grid to get policy functions for each skill level
    for no_states in range(model['distributions']['dist']['skills']['n']): # Note: here, n refers to the number of skill grid points
        one_policy = model['steady_state']['decisions'][f'{policy}'][no_states]
        
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
# Function to compare initial and terminal steady state
def stst_overview(models,
                  save_results, 
                  full_path_results,
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
    
    # ...and two models:
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
                if row['Variable'] in ['tau', 'D', 'DY', 'gr_liquid', 'borr_limit', 'MPC', 'R', 'Rbar', 'Rn', 'Rr', 'Rminus', 'Frac. of Borrowers', 'Frac. at Borrowing Limit', 'Frac. at Zero Assets']:
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
        stst_table_path = os.path.join(full_path_results, 
                                       'stst_comparison.tex')
        stst_df.to_latex(stst_table_path, 
                              label = 'tab:stst', index = False)
    
    # Return resulting data frame
    return stst_df
    