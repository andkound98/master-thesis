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