#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Andreas Koundouros
Date: 26.08.2023

This file contains custom functions used throughout the project.
"""

###############################################################################
###############################################################################
###############################################################################
# Packages
import os # path management
import pandas as pd # data wrangling
import numpy as np # data wrangling
import jax 
import jax.numpy as jnp
import econpizza as ep # Econpizza

###############################################################################
###############################################################################
# Function to obtain the path to a model based on settings
def get_model_path(settings):
    """Path to models.
    
    This function returns the full path to the desired model.
    """
    # Extract chosen model from settings dictionary
    exact_model = settings['Model']
    
    # Depending on the chosen model, set the path to the model
    return os.path.join(os.getcwd(), 'Models', f'hank_{exact_model}.yml')

###############################################################################
###############################################################################
# Function to obtain a path to save results
def get_exact_results_path(settings):
    """"Path for results.
    
    This function creates a path specifically asigned to a set of results and 
    creates that path, if it does not yet exist, in the 'Results' folder.
    """
    # Get sub-components of string
    comp1 = settings['Model']
    comp2 = settings['Shock']
    
    # Differentiate along the dimension of whether the shock is asymmetric
    if settings['Asymmetry'] == True:
        # Define path 
        path = os.path.join(os.getcwd(), 
                            'Results', 
                            f'{comp1}_{comp2}_asymmetric')
        
        # Check if the folder exists
        if not os.path.exists(path):
        # Create the folder if it doesn't exist
            os.makedirs(path)
            
        # Return final component of path 
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
        
        # Return final component of path 
        return f'{comp1}_{comp2}'

###############################################################################
###############################################################################
# Set model and shock parameters according to settings
def get_parametrisation(settings):
    """Obtain parameters.
    
    This function returns, for given settings, a pre-specified set of 
    parameters.
    """
    # Baseline calibration (used for 'hank_baseline', 'hank_slow_shock',
    # 'hank_fast_shock', 'hank_very_slow_phi', 'hank_no_ZLB')
    shock_model_parameters = {'initial_borrowing_limit': -2.3485,
                              'terminal_borrowing_limit': -2.1775,
                              'initial_wedge': 1e-8,
                              'terminal_wedge': 0.00203} 
    
    # Calibration for extended model with CRRA preferences
    if settings['Model'] == 'end_L':
        shock_model_parameters = {'initial_borrowing_limit': -1.7956,
                                  'terminal_borrowing_limit': -1.655,
                                  'initial_wedge': 1e-8,
                                  'terminal_wedge': 0.00277}
    
    # Calibration for model with low calibration of liquid assets
    if settings['Model'] == 'low_B':
        shock_model_parameters = {'initial_borrowing_limit': -1.54946,
                                  'terminal_borrowing_limit': -1.4238,
                                  'initial_wedge': 1e-8,
                                  'terminal_wedge': 0.00316}
        
    # Return parametrisation
    return shock_model_parameters

###############################################################################
###############################################################################
# Function to return initial and terminal models
def return_models_permanent(model_path, 
                            settings, 
                            shock_model_parameters,
                            asym=False):
    """Obtain Econpizza model instances
    
    This function parses through the given model path. Then, it adjusts this 
    model dictionary as specified in the setting and parameters passed to the 
    function. The output is one initial model and one terminal model. They of 
    course depend on the chosen shock.
    """
    # Get model as dictionary
    hank_dict = ep.parse(model_path)
    
    # Load models depending on the chosen shock
    
    # Permanent shock to the borrowing limit
    if settings['Shock'].startswith('limit') == True:
        # Settings for interest rate wedge
        hank_dict['steady_state']['fixed_values']['kappa'] = shock_model_parameters['initial_wedge']
        
        # Create model with initial borrowing limit
        init = shock_model_parameters['initial_borrowing_limit']
        term = shock_model_parameters['terminal_borrowing_limit']
        
        hank_dict['steady_state']['fixed_values']['phi'] = init
        hank_dict['definitions'] = hank_dict['definitions'].replace('amin = 0', f'amin = {init}') # replace place holders
        hank_dict['definitions'] = hank_dict['definitions'].replace('amin_terminal = 0', f'amin_terminal = {term}') # replace place holders
        
        # Load initial model
        hank_model_initial = ep.load(hank_dict)
        
        # Create model with terminal borrowing limit
        hank_dict['steady_state']['fixed_values']['phi'] = term
        
        # Load terminal model
        hank_model_terminal = ep.load(hank_dict)
        
    # Permanent shock to the interest rate wedge
    if settings['Shock'].startswith('wedge') == True:
        # Settings for borrowing limit
        init = shock_model_parameters['initial_borrowing_limit']
        term = shock_model_parameters['terminal_borrowing_limit']
        
        hank_dict['steady_state']['fixed_values']['phi'] = init
        hank_dict['definitions'] = hank_dict['definitions'].replace('amin = 0', f'amin = {init}') # replace place holders
        hank_dict['definitions'] = hank_dict['definitions'].replace('amin_terminal = 0', f'amin_terminal = {term}') # replace place holders
        
        # Create model with initial borrowing wedge
        hank_dict['steady_state']['fixed_values']['kappa'] = shock_model_parameters['initial_wedge']
        
        # Load initial model
        hank_model_initial = ep.load(hank_dict)
        
        # Create model with terminal borrowing wedge
        hank_dict['steady_state']['fixed_values']['kappa'] = shock_model_parameters['terminal_wedge']
        
        # Load terminal model
        hank_model_terminal = ep.load(hank_dict)
        
    # Return initial model and terminal model
    if asym == False: # Standard case
        return hank_model_initial, hank_model_terminal
    
    if asym == True: # Asymmetric case
        return hank_model_terminal, hank_model_initial # just swap terminal and initial

###############################################################################
###############################################################################
# Function to find the closest on-grid asset grid point for a given value
def find_closest_grid_point(ar_borrowing_limit, 
                            asset_grid):
    """Get closest on-grid grid point.
    
    This function returns the value and the index of the grid point (ON the 
    grid) which is closest in terms of numerical distance from the number 
    provided (potentially OFF grid).
    """
    # Create an array of distances between the grid points and a given value
    array_distances = jnp.abs(asset_grid - ar_borrowing_limit)
    
    # Get the index of the minimum distance in array 
    indx_min_distance = jnp.argmin(array_distances)
    
    # Get the grid point with the minimum distance to the given value
    closest_grid_point = asset_grid[indx_min_distance]
    
    # Return closest grid point and its index
    return closest_grid_point, indx_min_distance

###############################################################################
###############################################################################
# Function to get policies as data frane
def make_policy_df(hank_model,
                   policy,
                   borr_cutoff=None,
                   x_threshold=None):
    """Household policies as data frame.
    
    Obtain, for the desired household policy, a data frame of these policies 
    across skill levels.
    """
    # Preliminaries
    a_grid = hank_model['context']['a_grid'] # asset grid
    policy_arr = np.array(a_grid) # initialise container for policies
    policy_columns = ['grid'] # initialise name constainer
    
    # Loop through skill grid to get policy functions for each skill level
    for no_states in range(hank_model['distributions']['dist']['skills']['n']): # n refers here to the number of skill grid points
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
    
    # Return data frame
    return policy_df

###############################################################################
###############################################################################
# Function to compare initial and terminal steady states
def stst_overview(models, # list of models
                  save_results, 
                  exact_path,
                  percent=100):
    """Steady state overview.
    
    This function creates a table which compares the steady state values of the 
    two models passed to the function. The function adds a column which 
    calculates the differences between the two steady states. It distinguishes 
    (manually) between variables for which it needs to take the change in 
    absolute terms or in percentage terms.
    """
    # Create a rounding function for convenience
    round_func_4 = lambda x: round(float(x), 4)
        
    # Extract models from input list
    hank_model_init = models[0]
    hank_model_term = models[1]
    
    # Data frame of initial steady state
    stst_df = pd.DataFrame(hank_model_init['stst'].items(), 
                           columns = ['Variable', 'Initial'])
    stst_df['Initial'] = stst_df['Initial'].apply(round_func_4)
    
    # Data frame of terminal steady state
    term_stst_df = pd.DataFrame(hank_model_term['stst'].items(),
                                columns = ['Variable', 'Terminal'])
    term_stst_df['Terminal'] = term_stst_df['Terminal'].apply(round_func_4)
    
    # Merge into one data frame
    stst_df = pd.merge(stst_df, term_stst_df, 
                       on = 'Variable', how = 'left')
    
    # Add some more features of the steady states
    
    # Obtain asset grid and calculate distributions in initial and terminal
    # states
    a_grid_init = hank_model_init['context']['a_grid'] # asset grid
    distribution_skills_and_assets_initial = hank_model_init['steady_state']['distributions'][0] # initial dist over skills and assets
    distribution_assets_initial = percent*jnp.sum(distribution_skills_and_assets_initial, 
                                      axis = 0) # initial dist over assets
    distribution_skills_and_assets_terminal = hank_model_term['steady_state']['distributions'][0] # terminal dist over skills and assets
    distribution_assets_terminal = percent*jnp.sum(distribution_skills_and_assets_terminal, 
                                      axis = 0) # terminal dist over assets 
    
    # Calculate MPCs according to asset status, i.e. aggregate MPC among 
    # lenders and aggregate MPC among borrowers
    mpc_init = hank_model_init['steady_state']['decisions']['mpc']
    mpc_term = hank_model_term['steady_state']['decisions']['mpc']
    borr_mpc_init = jnp.sum(jnp.where(a_grid_init<0,distribution_skills_and_assets_initial*mpc_init,0), axis=(0,1)) / jnp.sum(jnp.where(a_grid_init<0, distribution_skills_and_assets_initial, 0))
    borr_mpc_term = jnp.sum(jnp.where(a_grid_init<0,distribution_skills_and_assets_terminal*mpc_term,0), axis=(0,1)) / jnp.sum(jnp.where(a_grid_init<0, distribution_skills_and_assets_terminal, 0))
    lend_mpc_init = jnp.sum(jnp.where(a_grid_init>=0,distribution_skills_and_assets_initial*mpc_init,0), axis=(0,1)) / jnp.sum(jnp.where(a_grid_init>=0, distribution_skills_and_assets_initial, 0))
    lend_mpc_term = jnp.sum(jnp.where(a_grid_init>=0,distribution_skills_and_assets_terminal*mpc_term,0), axis=(0,1)) / jnp.sum(jnp.where(a_grid_init>=0, distribution_skills_and_assets_terminal, 0))

    # Add MPC of borrowers to data frame
    row_mpc_indebted = {'Variable': 'MPC of Borrowers',
                          'Initial': round(borr_mpc_init,2),
                          'Terminal': round(borr_mpc_term,2)}
    row_mpc_indebted_df = pd.DataFrame([row_mpc_indebted])
    stst_df = pd.concat([stst_df, row_mpc_indebted_df], 
                                   ignore_index=True)
    
    # Add MPC of lenders to data frame
    row_mpc_lending = {'Variable': 'MPC of Lenders',
                          'Initial': round(lend_mpc_init,2),
                          'Terminal': round(lend_mpc_term,2)}
    row_mpc_lending_df = pd.DataFrame([row_mpc_lending])
    stst_df = pd.concat([stst_df, row_mpc_lending_df], 
                                   ignore_index=True)

    # Add fraction of borrowers to data frame
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
    
    # Add fraction of households at borrowing limit to data frame  
    row_share_limit = {'Variable': 'Frac. at Borrowing Limit',
                       'Initial': distribution_assets_initial[0].round(2).item(),
                       'Terminal': distribution_assets_terminal[distribution_assets_terminal>0][0].round(2).item()}
    row_share_limit_df = pd.DataFrame([row_share_limit])
    stst_df = pd.concat([stst_df, row_share_limit_df], 
                                   ignore_index=True)
    
    # Add fraction of households at zero assets to data frame
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
    
    # Add column with percentage changes
    stst_df['Change'] = 0

    # Calculate changes based on variable type
    for index, row in stst_df.iterrows():
        try:
            if row['Variable'] in ['beta', 'tau', 'DY', 'phi', 'MPC', 'R', 'kappa', 'Rn', 'Rr', 'Rrminus', 'spread', 'Frac. of Borrowers', 'Frac. at Borrowing Limit', 'Frac. at Zero Assets', 'MPC of Borrowers', 'MPC of Lenders'] or row['Variable'].startswith('Top') or row['Variable'].startswith('Bot'):
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
    
    # Save data frame as TeX table
    if save_results == True:
        stst_table_path = os.path.join(os.getcwd(),
                                       'Results',
                                       f'{exact_path}',
                                       f'stst_comparison_{exact_path}.tex')
        stst_df.to_latex(stst_table_path, 
                         label = f'tab:stst_comparison_{exact_path}', 
                         index = False)
    
    # Return data frame
    return stst_df

###############################################################################
###############################################################################
# Function to pickle transitions
def save_transition(model, 
                    x_trans, 
                    save_results,
                    exact_path):
    """Pickle transition.
    
    This function saves a given transition as a pickled pandas data frame.
    """
    # Transform JAX array into numpy array
    nparr = jax.device_get(x_trans)
    
    # Transform into pandas data frame
    transition_df = pd.DataFrame(nparr)
    
    # Give columns names 
    transition_df.columns = model['variables']
    
    # Get path to save pickle
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
    """Get pickled transitions.
    
    For a given dictionary with the names of saved transitions (in the 
    'Results' folder), this function obtains the transitions and stores them
    in a list of data frames.
    """
    # Create empty list of data frames
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
            
        # Raise error if transition under desired name was not yet saved
        except FileNotFoundError:
            raise FileNotFoundError(f'No transition yet saved under the name {transition}.')
            
    # Return the data frames of the chosen transitions as a list of data frames
    return list_of_transitions

###############################################################################
###############################################################################
# Function to get informative labels for model-shock combinations
def get_labels(comparison):
    """Obtain labels for chosen comparison.
    
    For a given dictionary with model-shock combinations which are to be
    compared, this function returns a list of informative labels.
    """
    # Correspondence of model-shock combinations to labels
    correspondence = {'baseline_limit_permanent': 'Baseline; Shock to \u03C6',
                      'baseline_limit_permanent_asymmetric': 'Credit Easing; Shock to \u03C6',
                      'baseline_wedge_permanent': 'Baseline; Shock to \u03BA',
                      'baseline_wedge_permanent_asymmetric': 'Credit Easing; Shock to \u03BA',
                      'end_L_limit_permanent': 'End. LS; Shock to \u03C6',
                      'end_L_limit_permanent_asymmetric': 'End. LS; Credit Easing; Shock to \u03C6',
                      'end_L_wedge_permanent': 'End. LS; Shock to \u03BA',
                      'end_L_wedge_permanent_asymmetric': 'End. LS; Credit Easing; Shock to \u03BA',
                      'slow_shock_limit_permanent': 'Slow Shock; Shock to \u03C6',
                      'slow_shock_limit_permanent_asymmetric': 'Slow Shock; Credit Easing; Shock to \u03C6',
                      'slow_shock_wedge_permanent': 'Slow Shock; Shock to \u03BA',
                      'slow_shock_wedge_permanent_asymmetric': 'Slow Shock; Credit Easing; Shock to \u03BA',
                      'fast_shock_limit_permanent': 'Fast Shock; Shock to \u03C6',
                      'fast_shock_limit_permanent_asymmetric': 'Fast Shock; Credit Easing; Shock to \u03C6',
                      'fast_shock_wedge_permanent': 'Fast Shock; Shock to \u03BA',
                      'fast_shock_wedge_permanent_asymmetric': 'Fast Shock; Credit Easing; Shock to \u03BA',
                      'low_B_limit_permanent': 'Low B; Shock to \u03C6',
                      'low_B_limit_permanent_asymmetric': 'Low B; Credit Easing; Shock to \u03C6',
                      'low_B_wedge_permanent': 'Low B; Shock to \u03BA',
                      'low_B_wedge_permanent_asymmetric': 'Low B; Credit Easing; Shock to \u03BA',
                      'no_ZLB_limit_permanent': 'No ZLB; Shock to \u03C6',
                      'no_ZLB_limit_permanent_asymmetric': 'No ZLB; Credit Easing; Shock to \u03C6',
                      'no_ZLB_wedge_permanent': 'No ZLB; Shock to \u03BA',
                      'no_ZLB_wedge_permanent_asymmetric': 'No ZLB; Credit Easing; Shock to \u03BA',
                      'very_slow_phi_limit_permanent': 'Very Slow \u03C6; Shock to \u03C6', 
                      'very_slow_phi_limit_permanent_asymmetric': 'Very Slow \u03C6; Credit Easing; Shock to \u03C6'}
    
    # Initialise list 
    list_of_labels = []
    
    # Get the values of the input dictionary
    comparison_list = list(comparison.values())
    
    # Loop through values to fill list of labels
    for component in comparison_list:
        label = correspondence[component]
        list_of_labels.append(label)
    
    # Return list of labels
    return list_of_labels

###############################################################################
###############################################################################
# Function to check for negative values in an array 
def check_for_negative_values(array_impl_obj):
    """Check for negative entries.
    
    This function raises an error whenever a negative value appears in the 
    provided data frame.
    """
    # Convert input into a numpy array
    array = jnp.asarray(array_impl_obj)

    # Check for negative entries
    negative_indices = jnp.argwhere(array < 0)
    
    # Raise error if negative entries were found
    if negative_indices.size > 0:
        raise ValueError('Warning: Negative values found in the consumption responses.')
    else:
        print('No negative values found in the consumption responses.')

###############################################################################
###############################################################################
# Function to obtain transition of aggregates and cross-section and check for 
# non-negativity of consumption
def get_agg_and_dist_transitions_and_check_c(terminal_model,
                                             initial_stst,
                                             initial_distr):
    """Obtain transition after shock.
    
    This function calculates for a terminal model and for initial conditions 
    (aggregate and cross-sectional characteristics of the initial steady 
     state) the transition of the model from the initial to the terminal state 
    and checks whether the individual consumption responses contain negative 
    values.
    """
    # Get transition of aggregate variables with Jacobian of terminal model
    agg_x, _ = terminal_model.find_path(init_state = initial_stst.values(), # pass initial steady state (aggregate block)
                                        init_dist = initial_distr) # pass initial distribution
    
    # Get transition of cross-sectional outcomes
    dist_x = terminal_model.get_distributions(trajectory = agg_x, # pass aggregate transition 
                                              init_dist = initial_distr) # pass initial distribution
    
    # Check cross-sectional dynamics of consumption for negative entries
    check_for_negative_values(dist_x['c'])
    
    # Return aggregate and cross-sectional transitions
    return agg_x, dist_x
    
###############################################################################
###############################################################################
# Function to truncate asset distribution at some threshold
def shorten_asset_dist(hank_model, 
                       x_threshold,
                       percent=100):
    """"Shorten asset distribution.
    
    This function returns a truncated asset distribution for a given model and
    threshold. To that end, it gets the full distribution, then collapses the 
    skills dimension and calculates the cumulative density from points at and
    greater than the threshold. This is then the last entry in the data frame.
    """
    # Get asset grid
    a_grid = hank_model['context']['a_grid']
    
    # Get distribution over assets as data frame
    distribution_skills_and_assets = hank_model['steady_state']['distributions'][0]
    distribution_assets = np.column_stack([a_grid, 
                                           percent*jnp.sum(distribution_skills_and_assets, 
                                                           axis = 0)])
    distribution_assets_df = pd.DataFrame(distribution_assets, 
                                          columns = ['grid', 'distribution'])

    # Filter the data frame based on the threshold
    filtered_df = distribution_assets_df[distribution_assets_df['grid'] < x_threshold]

    # Calculate the sum of shares for grid points above or equal to the threshold
    sum_share = distribution_assets_df[distribution_assets_df['grid'] >= x_threshold]['distribution'].sum()

    # Create a new row with the threshold and the cumulative density from that point onward
    threshold_row = pd.DataFrame({'grid': [x_threshold], 'distribution': [sum_share]})

    # Concatenate the filtered data frame with the threshold row
    short_asset_dist = pd.concat([filtered_df, threshold_row])

    # Reset the index of the new data frame
    short_asset_dist.reset_index(drop=True, inplace=True)
    
    # Return shortened distribution
    return short_asset_dist

###############################################################################
###############################################################################
# Function to convert certain dates to pandas datetime
def convert_quarter_to_datetime(quarter_str):
    """Convert dates.
    
    This function converts dates which are in string format year:quarter to a
    pandas datetime format.
    """
    # Split the input string into 'year' and 'quarter'
    year, quarter = quarter_str.split(':')
    
    # Get the number of the quarter
    quarter_number = int(quarter[1:])
    
    # Calculate the starting month of the quarter
    quarter_start_month = (quarter_number - 1) * 3 + 1
    
    # Construct a date string which is easy to convert to pandas datetime
    return pd.to_datetime(f"{year}-{quarter_start_month:02d}-01")
