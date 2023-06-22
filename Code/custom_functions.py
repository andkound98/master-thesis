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
# Custom functions

###############################################################################
# Function to create path for results depending on the model and the shock
def path_results(full_path_hank, 
                full_path_code,
                shock_limit):
    os.chdir('..') 
    full_path_ms = os.getcwd()
    if full_path_hank.endswith('hank_without_end_labour.yml'):
        if shock_limit == True:
            relative_path_results = os.path.join('Results', 
                                                 'Baseline',
                                                 'Shock_Limit')
        elif shock_limit == False:
            relative_path_results = os.path.join('Results', 
                                                 'Baseline',
                                                 'Shock_Wedge')
    
    elif full_path_hank.endswith('hank_with_end_labour.yml'):
        if shock_limit == True:
            relative_path_results = os.path.join('Results', 
                                                 'Endogenous_L',
                                                 'Shock_Limit')
        elif shock_limit == False:
            relative_path_results = os.path.join('Results', 
                                                 'Endogenous_L',
                                                 'Shock_Wedge')
    
    full_path_results = os.path.join(full_path_ms, relative_path_results)
    os.chdir(full_path_code)
    
    # Return path for results
    return full_path_results

def path_to_results(full_path_code,
                    shock_to_borrowing_constraint,
                    shock_permanent):
    os.chdir('..') 
    full_path_ms = os.getcwd()
    
    if shock_to_borrowing_constraint == True:
        if shock_permanent == True:
            relative_path_results = os.path.join('Results', 
                                                 'Baseline',
                                                 'Shock_Limit',
                                                 'Permanent')
        elif shock_permanent == False:
            relative_path_results = os.path.join('Results', 
                                                 'Baseline',
                                                 'Shock_Limit',
                                                 'Transitory')
            
    if shock_to_borrowing_constraint == False:
        if shock_permanent == True:
            relative_path_results = os.path.join('Results', 
                                                 'Baseline',
                                                 'Shock_Wedge',
                                                 'Permanent')
        elif shock_permanent == False:
            relative_path_results = os.path.join('Results', 
                                                 'Baseline',
                                                 'Shock_Wedge',
                                                 'Transitory')

    full_path_results = os.path.join(full_path_ms, relative_path_results)
    os.chdir(full_path_code)
    
    # Return path for results
    return full_path_results



def return_models_permanent(full_path_hank, 
                            shock_to_borrowing_constraint, 
                            persistence_borrowing_limit=None,
                            initial_borrowing_limit=None,
                            terminal_borrowing_limit=None,
                            persistence_borrowing_wedge=None,
                            initial_wedge=None,
                            terminal_wedge=None):
    # Get model as dictionary
    hank_dict = ep.parse(full_path_hank)
    
    if shock_to_borrowing_constraint == True and initial_borrowing_limit != None and terminal_borrowing_limit != None:
        # Create model with initial borrowing limit
        hank_dict['steady_state']['fixed_values']['rho_a'] = persistence_borrowing_limit
        
        hank_dict['steady_state']['fixed_values']['lower_bound_a'] = initial_borrowing_limit
        hank_dict['definitions'] = hank_dict['definitions'].replace('amin = 0', f'amin = {initial_borrowing_limit}')
        hank_dict['definitions'] = hank_dict['definitions'].replace('amin_terminal = 0', f'amin_terminal = {terminal_borrowing_limit}')

        hank_dict['definitions'] = hank_dict['definitions'].replace('rho_a = 0.3', f'rho_a = {persistence_borrowing_limit}')

        hank_model_initial = ep.load(hank_dict) # Load initial model

        # Create model with terminal borrowing limit
        hank_dict['steady_state']['fixed_values']['lower_bound_a'] = terminal_borrowing_limit
        hank_model_terminal = ep.load(hank_dict) # Load terminal model
    
    elif shock_to_borrowing_constraint == False and initial_wedge != None and terminal_wedge != None:
        # Create model with initial borrowing wedge
        hank_dict['steady_state']['fixed_values']['rho_Rbar'] = persistence_borrowing_wedge
        
        hank_dict['steady_state']['fixed_values']['Rbar'] = initial_wedge
        hank_model_initial = ep.load(hank_dict) # Load initial model

        # Create model with terminal borrowing wedge
        hank_dict['steady_state']['fixed_values']['Rbar'] = terminal_wedge
        hank_model_terminal = ep.load(hank_dict) # Load terminal model
        
    return hank_model_initial, hank_model_terminal





def return_models_transitory(full_path_hank,
                             shock_to_borrowing_constraint, 
                             persistence_borrowing_limit=None,
                             stst_borrowing_limit=None,
                             persistence_borrowing_wedge=None,
                             stst_borrowing_wedge=None):
    # Get model as dictionary
    hank_dict = ep.parse(full_path_hank)
    
    if shock_to_borrowing_constraint == True and stst_borrowing_limit != None:
        # Create model
        hank_dict['steady_state']['fixed_values']['rho_a'] = persistence_borrowing_limit
        
        hank_dict['steady_state']['fixed_values']['lower_bound_a'] = stst_borrowing_limit
        hank_dict['definitions'] = hank_dict['definitions'].replace('amin = 0', f'amin = {stst_borrowing_limit}')
        hank_dict['definitions'] = hank_dict['definitions'].replace('rho_a = 0.3', f'rho_a = {persistence_borrowing_limit}')

        hank_model = ep.load(hank_dict) # Load initial model
        
    elif shock_to_borrowing_constraint == False and stst_borrowing_wedge != None:
        # Create model
        hank_dict['steady_state']['fixed_values']['rho_Rbar'] = persistence_borrowing_wedge
        hank_dict['steady_state']['fixed_values']['Rbar'] = stst_borrowing_wedge
        hank_model = ep.load(hank_dict) # Load initial model
    
    return hank_model



###############################################################################
# Function to make models
def make_models(full_path_hank, shock_limit):
    if shock_limit == True:
        if full_path_hank.endswith('hank_without_end_labour.yml'):
            initial_borrowing_limit = -1.2 # initial borrowing limit
            terminal_borrowing_limit = -0.94 # terminal borrowing limit
        elif full_path_hank.endswith('hank_with_end_labour.yml'):
            initial_borrowing_limit = -1 # initial borrowing limit
            terminal_borrowing_limit = -0.7 # terminal borrowing limit
        
        # Fix persistence in the shock to the borrowing limit
        persistence_borrowing_limit = 0.3
        
        # Get model as dictionary
        hank_dict = ep.parse(full_path_hank)

        # Create model with initial borrowing limit
        hank_dict['steady_state']['fixed_values']['lower_bound_a'] = initial_borrowing_limit
        hank_dict['definitions'] = hank_dict['definitions'].replace('amin = 0', f'amin = {initial_borrowing_limit}')
        hank_dict['definitions'] = hank_dict['definitions'].replace('amin_terminal = 0', f'amin_terminal = {terminal_borrowing_limit}')

        hank_dict['steady_state']['fixed_values']['rho_a'] = persistence_borrowing_limit
        hank_dict['definitions'] = hank_dict['definitions'].replace('rho_a = 0.3', f'rho_a = {persistence_borrowing_limit}')

        hank_model_initial = ep.load(hank_dict) # Load initial model

        # Create model with terminal borrowing limit
        hank_dict['steady_state']['fixed_values']['lower_bound_a'] = terminal_borrowing_limit
        hank_model_terminal = ep.load(hank_dict) # Load terminal model
        
    elif shock_limit == False:
        # Get model as dictionary
        hank_dict = ep.parse(full_path_hank)

        # Create model with initial borrowing wedge
        hank_dict['steady_state']['fixed_values']['Rbar'] = 1e-8
        hank_model_initial = ep.load(hank_dict) # Load initial model

        # Create model with terminal borrowing wedge
        hank_dict['steady_state']['fixed_values']['Rbar'] = 0.01
        hank_model_terminal = ep.load(hank_dict) # Load terminal model
        
        terminal_borrowing_limit = -1.1
    
    return (hank_model_initial, 
            hank_model_terminal, terminal_borrowing_limit)

###############################################################################
# Function to find the closest on-grid asset grid point for a given off-grid 
# value
def find_closest_grid_point(ar_borrowing_limit, 
                            asset_grid):
    import jax.numpy as jnp
    array_distances = jnp.abs(asset_grid - ar_borrowing_limit)
    
    indx_min_distance = jnp.argmin(array_distances)
    
    closest_grid_point = asset_grid[indx_min_distance]
    
    return closest_grid_point, indx_min_distance

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
    for no_states in range(model['distributions']['dist']['skills']['n']):
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
def make_stst_comparison(hank_model_init, # model with initial borrowing limit
                         hank_model_term, # model with terminal borrowing limit
                         save_tables, # if table is supposed to be saved
                         path, # path to save table
                         percent=100):
    round_func_4 = lambda x: round(float(x), 4) # Rounding functions
    
    hank_stst_df = pd.DataFrame(hank_model_init['stst'].items(), 
                                columns = ['Variable', 'Initial'])
    hank_stst_df['Initial'] = hank_stst_df['Initial'].apply(round_func_4)
    
    hank_stst_df_terminal = pd.DataFrame(hank_model_term['stst'].items(),
                                         columns = ['Variable', 'Terminal'])
    hank_stst_df_terminal['Terminal'] = hank_stst_df_terminal['Terminal'].apply(round_func_4)
    
    # Merge steady states into one data frame
    full_stst_analysis = pd.merge(hank_stst_df, hank_stst_df_terminal, 
                                  on = 'Variable', how = 'left')
    
    # Add some more features of the steady states
    a_grid_init = hank_model_init['context']['a_grid']
    
    # Initial steady state 
    distribution_skills_and_assets_initial = hank_model_init['steady_state']['distributions'][0]
    distribution_assets_initial = 100*jnp.sum(distribution_skills_and_assets_initial, 
                                      axis = 0)
    
    # Terminal steady state 
    distribution_skills_and_assets_terminal = hank_model_term['steady_state']['distributions'][0]
    distribution_assets_terminal = 100*jnp.sum(distribution_skills_and_assets_terminal, 
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
    full_stst_analysis = pd.concat([full_stst_analysis, row_share_indebted_df], 
                                   ignore_index=True)
    
    # Add fraction of households at borrowing limit    
    row_share_limit = {'Variable': 'Frac. at Borrowing Limit',
                       'Initial': distribution_assets_initial[0].round(2).item(),
                       'Terminal': distribution_assets_terminal[distribution_assets_terminal>0][0].round(2).item()}
    row_share_limit_df = pd.DataFrame([row_share_limit])
    full_stst_analysis = pd.concat([full_stst_analysis, row_share_limit_df], 
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
    full_stst_analysis = pd.concat([full_stst_analysis, row_share_zero_df], 
                                   ignore_index=True)
    
    # Add column which calculates changes in percent between the steady states
    full_stst_analysis['Change'] = 0

    # Calculate changes based on variable type
    for index, row in full_stst_analysis.iterrows():
        try:
            if row['Variable'] in ['D', 'DY', 'gr_liquid', 'lower_bound_a', 'MPC', 'R', 'Rbar', 'Rn', 'Rr', 'Rminus', 'Frac. of Borrowers', 'Frac. at Borrowing Limit', 'Frac. at Zero Assets']:
                # Absolute change for specific variables
                full_stst_analysis.at[index, 'Change'] = row['Terminal'] - row['Initial']
            else:
                # Percentage change for other variables
                full_stst_analysis.at[index, 'Change'] = percent * ((row['Terminal'] - row['Initial']) / row['Initial'])
        # In case of division by zero, insert NaN
        except ZeroDivisionError:
            full_stst_analysis.at[index, 'Change'] = np.nan

    # Round changes
    full_stst_analysis['Change'] = full_stst_analysis['Change'].apply(round_func_4)
    
    # Save table for LaTeX
    if save_tables == True and 'n' in hank_model_init['steady_state']['decisions'].keys():
        stst_table_path = os.path.join(path, 'stst_comparison_labour.tex')
        full_stst_analysis.to_latex(stst_table_path, 
                                    label = 'tab:stst_labour', index = False)
    elif save_tables == True and 'n' not in hank_model_init['steady_state']['decisions'].keys():
        stst_table_path = os.path.join(path, 'stst_comparison.tex')
        full_stst_analysis.to_latex(stst_table_path, 
                                    label = 'tab:stst', index = False)
    
    # Return resulting data frame
    return full_stst_analysis