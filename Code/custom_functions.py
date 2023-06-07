#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Andreas Koundouros, University of Bonn
Date: 15.05.2023

This file contains some custom functions used throughout the project.
"""

###############################################################################
###############################################################################
# Import packages
import os
import pandas as pd
import jax.numpy as jnp

###############################################################################
###############################################################################
# Custom functions

def find_closest_grid_point(ar_borrowing_limit, asset_grid):
    import jax.numpy as jnp
    array_distances = jnp.abs(asset_grid - ar_borrowing_limit)
    
    indx_min_distance = jnp.argmin(array_distances)
    
    closest_grid_point = asset_grid[indx_min_distance]
    
    return closest_grid_point, indx_min_distance

def find_stable_time(list_over_time):
    stable_time_index = 0
    stable_value = list_over_time[stable_time_index]
    
    for tt in range(len(list_over_time)):
        if list_over_time[tt+1] > stable_value:
            stable_value = list_over_time[tt+1]
        else:
            stable_time_index = tt
            break
        
    return stable_time_index

###############################################################################
# Function to compare initial and terminal steady state
def make_stst_comparison(hank_model_init, hank_model_term, 
                         save_tables, path, percent=100):
    round_func_4 = lambda x: round(float(x), 4) # Rounding functions
    round_func_2 = lambda x: round(float(x), 2)
    
    hank_stst_df = pd.DataFrame(hank_model_init['stst'].items(), 
                                columns = ['Variable', 'Initial Steady State'])
    hank_stst_df['Initial Steady State'] = hank_stst_df['Initial Steady State'].apply(round_func_4)
    
    hank_stst_df_terminal = pd.DataFrame(hank_model_term['stst'].items(),
                                         columns = ['Variable', 'Terminal Steady State'])
    hank_stst_df_terminal['Terminal Steady State'] = hank_stst_df_terminal['Terminal Steady State'].apply(round_func_4)
    
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
    
    # Add share of indebted households
    row_share_indebted = {'Variable': 'Share Indebted',
                          'Initial Steady State': jnp.sum(jnp.where(a_grid_init < 0, 
                                                                    distribution_assets_initial, 
                                                                    0)).round(2).item(),
                          'Terminal Steady State': jnp.sum(jnp.where(a_grid_init < 0, 
                                                                    distribution_assets_terminal, 
                                                                    0)).round(2).item()}
    row_share_indebted_df = pd.DataFrame([row_share_indebted])
    full_stst_analysis = pd.concat([full_stst_analysis, row_share_indebted_df], 
                                   ignore_index=True)
    
    # Add share of households at borrowing limit    
    row_share_limit = {'Variable': 'Share at Limit',
                       'Initial Steady State': distribution_assets_initial[0].round(2).item(),
                       'Terminal Steady State': distribution_assets_terminal[distribution_assets_terminal>0][0].round(2).item()}
    row_share_limit_df = pd.DataFrame([row_share_limit])
    full_stst_analysis = pd.concat([full_stst_analysis, row_share_limit_df], 
                                   ignore_index=True)
    
    # Add column which calculates changes in percent between the steady states
    full_stst_analysis['Percent Change'] = (percent*(full_stst_analysis['Terminal Steady State']-full_stst_analysis['Initial Steady State'])/full_stst_analysis['Initial Steady State']).apply(round_func_2)
    
    # Save table for LaTeX
    if save_tables == True and 'n' in hank_model_init['steady_state']['decisions'].keys():
        stst_table_path = os.path.join(path, 'stst_comparison_labour.tex')
        full_stst_analysis.to_latex(stst_table_path, 
                                    label = 'tab:stst_labour', index = False)
    elif save_tables == True and 'n' not in hank_model_init['steady_state']['decisions'].keys():
        stst_table_path = os.path.join(path, 'stst_comparison.tex')
        full_stst_analysis.to_latex(stst_table_path, 
                                    label = 'tab:stst', index = False)
    
    return full_stst_analysis