#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Andreas Koundouros
Date: 15.05.2023

This file contains custom functions used throughout the project, inter alia in 
the main file and the plot_functions file.
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
    
    if settings['Asymmetry'] == True:
        # Define path 
        path = os.path.join(os.getcwd(), 
                            'Results', 
                            f'{comp1}_{comp2}_asymmetric')
        
        # Check if the folder exists
        if not os.path.exists(path):
        # Create the folder if it doesn't exist
            os.makedirs(path)
            
        return f'{comp1}_{comp2}_asymmetric'
    
    else:
        # Define path 
        path = os.path.join(os.getcwd(), 
                            'Results', 
                            f'{comp1}_{comp2}')
        
        # Check if the folder exists
        if not os.path.exists(path):
        # Create the folder if it doesn't exist
            os.makedirs(path)
            
        return f'{comp1}_{comp2}'

###############################################################################
###############################################################################
# Set model and shock parameters according to settings
def get_parametrisation(settings):
    # Parametrisation in case of baseline model of section 3
    if settings['Model'] == 'baseline' or settings['Model'] == 'slow_shock' or settings['Model'] == 'fast_shock' or settings['Model'] == 'no_ZLB' or settings['Model'] == 'no_omega' or settings['Model'] == 'low_psi':
        shock_model_parameters = {'initial_borrowing_limit': -2.3485,
                                  'terminal_borrowing_limit': -2.1775,
                                  'initial_wedge': 1e-8,
                                  'terminal_wedge': 0.00203}
    
    # Parametrisation in case of extended model with CRRA preferences of section 6.2
    if settings['Model'] == 'end_L':
        shock_model_parameters = {'initial_borrowing_limit': -1.7956,
                                  'terminal_borrowing_limit': -1.655,
                                  'initial_wedge': 1e-8,
                                  'terminal_wedge': 0.00277}
    
    # Parametrisation in case of baseline model with a high B calibration
    if settings['Model'] == 'low_B':
        shock_model_parameters = {'initial_borrowing_limit': -1.54946,
                                  'terminal_borrowing_limit': -1.4238,
                                  'initial_wedge': 1e-8,
                                  'terminal_wedge': 0.00316}
        
    # Return shock and model parametrisation
    return shock_model_parameters

###############################################################################
###############################################################################
# Function to return two models, where the first one corresponds to the initial
# steady state and the second one corresponds to the terminal steady state
def return_models_permanent(model_path, 
                            settings, 
                            shock_model_parameters,
                            asym=False):
    # Get model as dictionary
    hank_dict = ep.parse(model_path)
    
    # Load models depending on the chosen shock and settings
    
    # Models for permanent shock to the borrowing limit
    if settings['Shock'].startswith('limit') == True:
        # Settings for interest rate wedge
        hank_dict['steady_state']['fixed_values']['kappa'] = shock_model_parameters['initial_wedge']
        
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
        hank_dict['steady_state']['fixed_values']['kappa'] = shock_model_parameters['initial_wedge']
        
        # Load initial model
        hank_model_initial = ep.load(hank_dict)
        
        # Create model with terminal borrowing wedge
        hank_dict['steady_state']['fixed_values']['kappa'] = shock_model_parameters['terminal_wedge']
        
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
        hank_dict['steady_state']['fixed_values']['kappa'] = shock_model_parameters['initial_wedge']
        
        # Load initial model
        hank_model_initial = ep.load(hank_dict)
        
        # Create model with terminal borrowing wedge
        hank_dict['steady_state']['fixed_values']['beta'] = shock_model_parameters['terminal_beta']
        
        # Load terminal model
        hank_model_terminal = ep.load(hank_dict)
        
    # Return initial model and terminal model
    if asym == False: # Standard case
        return hank_model_initial, hank_model_terminal
    
    if asym == True: # Asymmetric case
        return hank_model_terminal, hank_model_initial

###############################################################################
###############################################################################
def return_models_transitory(model_path, 
                            settings, 
                            shock_model_parameters):
    # Get model as dictionary
    hank_dict = ep.parse(model_path)
    
    # Load model depending on the chosen shock
    
    hank_dict['steady_state']['fixed_values']['kappa'] = shock_model_parameters['initial_wedge']
    
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
        a_grid_term = hank_model_term['context']['a_grid']
        
        # Initial steady state 
        distribution_skills_and_assets_initial = hank_model_init['steady_state']['distributions'][0]
        distribution_assets_initial = percent*jnp.sum(distribution_skills_and_assets_initial, 
                                          axis = 0)
        
        # Terminal steady state 
        distribution_skills_and_assets_terminal = hank_model_term['steady_state']['distributions'][0]
        distribution_assets_terminal = percent*jnp.sum(distribution_skills_and_assets_terminal, 
                                          axis = 0)
        
        #a_grid = hank_model['context']['a_grid']
        #dist = hank_model['steady_state']['distributions'][0]
        mpc_init = hank_model_init['steady_state']['decisions']['mpc']
        mpc_term = hank_model_term['steady_state']['decisions']['mpc']
        
        borr_mpc_init = jnp.sum(jnp.where(a_grid_init<0,distribution_skills_and_assets_initial*mpc_init,0), axis=(0,1)) / jnp.sum(jnp.where(a_grid_init<0, distribution_skills_and_assets_initial, 0))
        borr_mpc_term = jnp.sum(jnp.where(a_grid_init<0,distribution_skills_and_assets_terminal*mpc_term,0), axis=(0,1)) / jnp.sum(jnp.where(a_grid_init<0, distribution_skills_and_assets_terminal, 0))
        
        current_limit_index = jnp.argmax(distribution_skills_and_assets_initial > 0)
        current_limit = a_grid_init[current_limit_index]

        # Find the value in arr1 which is 10 entries away from 'current_limit'
        target_index = current_limit_index + 10
        target_value = a_grid_init[target_index]
        
        constr_mpc_init = jnp.sum(jnp.where(a_grid_init<target_value,distribution_skills_and_assets_initial*mpc_init,0), axis=(0,1)) / jnp.sum(jnp.where(a_grid_init<target_value, distribution_skills_and_assets_initial, 0))
        
        current_limit_index = jnp.argmax(distribution_skills_and_assets_terminal > 0)
        current_limit = a_grid_term[current_limit_index]

        # Find the value in arr1 which is 10 entries away from 'current_limit'
        target_index = current_limit_index + 10
        target_value = a_grid_term[target_index]
        
        constr_mpc_term = jnp.sum(jnp.where(a_grid_term<target_value,distribution_skills_and_assets_terminal*mpc_term,0), axis=(0,1)) / jnp.sum(jnp.where(a_grid_term<target_value, distribution_skills_and_assets_terminal, 0))
        
        lend_mpc_init = jnp.sum(jnp.where(a_grid_init>=0,distribution_skills_and_assets_initial*mpc_init,0), axis=(0,1)) / jnp.sum(jnp.where(a_grid_init>=0, distribution_skills_and_assets_initial, 0))
        lend_mpc_term = jnp.sum(jnp.where(a_grid_init>=0,distribution_skills_and_assets_terminal*mpc_term,0), axis=(0,1)) / jnp.sum(jnp.where(a_grid_init>=0, distribution_skills_and_assets_terminal, 0))
    
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
        
        # Add MPC of indebted households
        row_mpc_indebted = {'Variable': 'MPC of Borrowers',
                              'Initial': round(borr_mpc_init,2),
                              'Terminal': round(borr_mpc_term,2)}
        row_mpc_indebted_df = pd.DataFrame([row_mpc_indebted])
        stst_df = pd.concat([stst_df, row_mpc_indebted_df], 
                                       ignore_index=True)
        
        # Add MPC of households very close to the constraint
        
        # Add MPC of lending households
        row_mpc_lending = {'Variable': 'MPC of Lenders',
                              'Initial': round(lend_mpc_init,2),
                              'Terminal': round(lend_mpc_term,2)}
        row_mpc_lending_df = pd.DataFrame([row_mpc_lending])
        stst_df = pd.concat([stst_df, row_mpc_lending_df], 
                                       ignore_index=True)
        
        # Add column which calculates changes in percent between the steady states
        stst_df['Change'] = 0

        # Calculate changes based on variable type
        for index, row in stst_df.iterrows():
            try:
                if row['Variable'] in ['beta', 'tau', 'D', 'DY', 'gr_liquid', 'phi', 'MPC', 'R', 'kappa', 'Rn', 'Rr', 'Rrminus', 'spread', 'Frac. of Borrowers', 'Frac. at Borrowing Limit', 'Frac. at Zero Assets', 'MPC of Borrowers', 'MPC of Lenders', 'Top10C', 'Top10A', 'Top1C', 'Top1A', 'Top25C', 'Top25A', 'Bot25A', 'Bot25C']:
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
                                       f'{exact_path}',
                                       f'stst_comparison_{exact_path}.tex')
        stst_df.to_latex(stst_table_path, 
                         label = f'tab:stst_comparison_{exact_path}', 
                         index = False)
    
    # Return resulting data frame
    return stst_df

###############################################################################
###############################################################################
# Function to pickle transitions
def save_transition(model, 
                    x_trans, 
                    save_results,
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
                        f'{exact_path}',
                        f'x_trans_{exact_path}.pkl')
    
    # Save data frame as pickle
    if save_results == True:
        transition_df.to_pickle(path)
    else: 
        pass

###############################################################################
###############################################################################
# Function to obtain pickled transitions
def get_transitions(comparison):
    # Create empty list of data frames with desired transitions
    list_of_transitions = []
    
    # Get the values of the comparison dictionary as a list
    comparison_list = list(comparison.values())
    
    # Iterate through desired transitions to obtain the pickled data frames
    for tt in range(len(comparison_list)):
        transition = comparison_list[tt]
        path = os.path.join(os.getcwd(),
                            'Results',
                            f'{transition}',
                            f'x_trans_{transition}.pkl')
        try:
            x_df = pd.read_pickle(path)
            list_of_transitions.append(x_df)
        except FileNotFoundError:
            raise FileNotFoundError(f'No transition yet saved under the name {transition}.')
            
    # Return the data frames of the chosen transitions as a list of data frames
    return list_of_transitions




def get_labels(comparison):
    correspondence = {'baseline_limit_permanent': 'Baseline; Shock to \u03C6',
                      'baseline_wedge_permanent': 'Baseline; Shock to \u03BA',
                      'end_L_limit_permanent': 'End. LS; Shock to \u03C6',
                      'end_L_wedge_permanent': 'End. LS; Shock to \u03BA',
                      'fast_shock_limit_permanent': 'Fast Shock; Shock to \u03C6',
                      'fast_shock_wedge_permanent': 'Fast Shock; Shock to \u03BA',
                      'low_B_limit_permanent': 'Low B; Shock to \u03C6',
                      'low_B_wedge_permanent': 'Low B; Shock to \u03BA',
                      'no_ZLB_limit_permanent': 'No ZLB; Shock to \u03C6',
                      'no_ZLB_wedge_permanent': 'No ZLB; Shock to \u03BA',
                      'slow_shock_limit_permanent': 'Slow Shock; Shock to \u03C6',
                      'slow_shock_wedge_permanent': 'Slow Shock; Shock to \u03BA'}
    
    # Create empty list of data frames with desired transitions
    list_of_labels = []
    
    # Get the values of the comparison dictionary as a list
    comparison_list = list(comparison.values())
    
    for component in comparison_list:
        label = correspondence[component]
        list_of_labels.append(label)
    
    # Return list of labels
    return list_of_labels
        


def check_for_negative_values(array_impl_obj):
    # Convert input into a numpy array
    array = jnp.asarray(array_impl_obj)

    # Check for negative entries
    negative_indices = jnp.argwhere(array < 0)
    
    # Raise error if negative entries were found
    if negative_indices.size > 0:
        raise ValueError('Warning: Negative values found in the consumption responses.')
    
    else:
        print('No negative values found in the consumption responses.')


def get_agg_and_dist_transitions_and_check_c(terminal_model,
                                             initial_stst,
                                             initial_distr):
    # Get transition of aggregate variables
    agg_x, _ = terminal_model.find_path(init_state = initial_stst.values(),
                                        init_dist = initial_distr)
    
    # Get transition of cross-sectional outcomes
    dist_x = terminal_model.get_distributions(trajectory = agg_x,
                                              init_dist = initial_distr)
    
    # Check cross-sectional dynamics of consumption for negative entries
    check_for_negative_values(dist_x['c'])
    
    # Return aggregate and cross-sectional transitions
    return agg_x, dist_x
    
def shorten_asset_dist(hank_model, 
                       x_threshold,
                       percent=100):
    # Get asset grid
    a_grid = hank_model['context']['a_grid']
    
    # Distribution over skills and assets
    distribution_skills_and_assets = hank_model['steady_state']['distributions'][0]
    
    # Distribution over assets
    distribution_assets = np.column_stack([a_grid, 
                                           percent*jnp.sum(distribution_skills_and_assets, 
                                                           axis = 0)])
    distribution_assets_df = pd.DataFrame(distribution_assets, 
                                          columns = ['grid', 'distribution'])

    # Filter the data frame based on the threshold
    filtered_df = distribution_assets_df[distribution_assets_df['grid'] < x_threshold]

    # Calculate the sum of shares for grid points above or equal to the threshold
    sum_share = distribution_assets_df[distribution_assets_df['grid'] >= x_threshold]['distribution'].sum()

    # Create a new row with the threshold and the sum of shares
    threshold_row = pd.DataFrame({'grid': [x_threshold], 'distribution': [sum_share]})

    # Concatenate the filtered data frame with the threshold row
    short_asset_dist = pd.concat([filtered_df, threshold_row])

    # Reset the index of the new data frame
    short_asset_dist.reset_index(drop=True, inplace=True)
    
    # Return shortened distribution
    return short_asset_dist
