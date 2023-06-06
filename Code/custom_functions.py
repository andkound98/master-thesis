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
import pandas as pd

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

def make_stst_comparison(hank_model_init, hank_model_term, save_tables, percent=100):
    round_func_4 = lambda x: round(float(x), 4) # Rounding functions
    round_func_2 = lambda x: round(float(x), 2)
    
    hank_stst_df = pd.DataFrame(hank_model_init['stst'].items(), 
                                columns = ['Variable', 'Initial Steady State'])
    hank_stst_df['Initial Steady State'] = hank_stst_df['Initial Steady State'].apply(round_func_4)
    
    hank_stst_df_terminal = pd.DataFrame(hank_model_term['stst'].items(),
                                         columns = ['Variable', 'Terminal Steady State'])
    hank_stst_df_terminal['Terminal Steady State'] = hank_stst_df_terminal['Terminal Steady State'].apply(round_func_4)
    
    # Compare steady states
    full_stst_analysis = pd.merge(hank_stst_df, hank_stst_df_terminal, 
                                  on = 'Variable', how = 'left')
    full_stst_analysis['Percent Change'] = (percent*(hank_stst_df_terminal['Terminal Steady State']-hank_stst_df['Initial Steady State'])/hank_stst_df['Initial Steady State']).apply(round_func_2)
    
    # Save table for LaTeX
    if save_tables == True:
        stst_table_path = '/Users/andreaskoundouros/Documents/stst.tex'
        full_stst_analysis.style.hide(axis="index").to_latex(buf = stst_table_path, 
                                                             label = 'tab:stst', 
                                                             hrules=False)
    
    return full_stst_analysis