#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Andreas Koundouros
Date: 26.08.2023

This file contains functions necessary for the solution of the HANK models.

Based on example functions file from Gregor Boehl, see:
https://github.com/gboehl/econpizza/blob/master/econpizza/examples/hank_functions.py
"""

###############################################################################
###############################################################################
###############################################################################
# Packages
import numpy as np
import jax
import jax.numpy as jnp
from grgrjax import amax
from econpizza.utilities.interp import interpolate
from econpizza.utilities.grids import log_grid

from custom_functions import find_closest_grid_point # get grid point closest to given input

###############################################################################
###############################################################################
###############################################################################
# Functions for baseline HANK model (w/o endogenous labour supply)

###############################################################################
###############################################################################
# Function to initialise EGM
def egm_init(a_grid, skills_grid):
    """EGM initialisation.
    
    This function initialises the marginal utility of consumption as an array
    with some reasonably small number. With this, the EGM can start.
    """
    return jnp.ones((skills_grid.shape[0], a_grid.shape[0]))*1e-2

###############################################################################
###############################################################################
# Function for a single EGM step
def egm_step(Wa_p, a_grid, skills_grid, w, n, T, R, Rminus, beta, sigma_c, sigma_l, db, lower_bound_a):    
    """One Step of the Endogenous Gridpoint Method without endogeneous labour choice.
    
    This function takes a single backward step via EGM. It is iterated on 
    backwards in order to obtain optimal consumption and asset holding 
    policies.
    """

    # Marginal utility of consumption as implied by next period's value 
    # function
    ux_nextgrid = beta * Wa_p
    
    # Labour income
    labor_inc = skills_grid[:, None]*n*w 

    # Next period's consumption from MUC and marginal utility of labor
    c_nextgrid = ux_nextgrid**(-1/sigma_c) + labor_inc/(1 + sigma_l)
    
    # Full interest rate schedule
    Rfull = jnp.where(a_grid < 0,
                      Rminus,
                      R)
    
    # Consumption in grid space
    lhs = c_nextgrid - labor_inc + a_grid[None, :] - T[:, None]
    rhs = Rfull * a_grid
    c = interpolate(lhs, rhs, c_nextgrid)

    # Today's asset holdings
    a = rhs + labor_inc + T[:, None] - c
    
    # Find lower bound on-grid
    lower_bound_a, _ = find_closest_grid_point(lower_bound_a, a_grid) 
    
    # Fix consumption and assets for constrained households at the currently
    # valid borrowing limit
    c = jnp.where(a < lower_bound_a, 
                  labor_inc + rhs + T[:, None] - lower_bound_a, 
                  c)
    a = jnp.where(a < lower_bound_a, 
                  lower_bound_a, 
                  a)
    
    # Calculate the marginal propensity to consume (MPC) out of a small
    # increase (given by db) of liquid wealth
    mpc = (interpolate(a, (a + db), c) - c) / db
    
    # Restrict MPCs to at most 1
    mpc = jnp.where(mpc > 1.0, 1.0, mpc)
    
    # Calculate new marginal utility of consumtion for next EGM step
    Wa = Rfull * (c - labor_inc/(1 + sigma_l)) ** (-sigma_c)
    
    # Return new marginal utility of consumption, asset holdings, consumption 
    # and marginal propensity to consume
    return Wa, a, c, mpc

###############################################################################
###############################################################################
###############################################################################
# Functions for extended HANK model with endogenous labour supply

###############################################################################
###############################################################################
# Function to initialise EGM
def egm_init_labour(a_grid, we, R, Rminus, sigma_c, T):
    """EGM initialisation.
    
    This function initialises the marginal utility of consumption with a rough 
    guess on consumption, a fraction of cash-on-hand in particular.
    """
    
    # Get full interest rate schedule
    Rfull = jnp.where(a_grid < 0,
                      Rminus,
                      R)
    
    # Calculate cash-on-hand in order to derive marginal utility of consumption
    # from it
    coh = Rfull * a_grid[None, :] + we[:, None] + T[:, None]
    
    Wa = Rfull * (0.1 * coh) ** (-sigma_c)
    
    return Wa

###############################################################################
###############################################################################
# Function for a single EGM step
def egm_step_labour(Wa_p, a_grid, we, trans, R, Rminus, beta, sigma_c, sigma_l, vphi, db, lower_bound_a):
    """One Step of the Endogenous Gridpoint Method with endogenous labour choice. 
    
    This function takes a single backward step via EGM. It is iterated on 
    backwards in order to obtain optimal consumption, asset holding and labour
    policies.
    """

    # Marginal utility of consumption as implied by next period's value 
    # function
    uc_nextgrid = beta * Wa_p
    
    # Next period's consumption and labour supply from MUC
    c_nextgrid, n_nextgrid = cn(uc_nextgrid, we[:, None], sigma_c, sigma_l, vphi)
    
    # Full interest rate schedule
    Rfull = jnp.where(a_grid < 0,
                      Rminus,
                      R)

    # Consumption and labour supply in grid space
    lhs = c_nextgrid - we[:, None] * n_nextgrid + a_grid[None, :] - trans[:, None]
    rhs = Rfull * a_grid
    c = interpolate(lhs, rhs, c_nextgrid)
    n = interpolate(lhs, rhs, n_nextgrid)

    # Today's asset holdings
    a = rhs + we[:, None] * n + trans[:, None] - c
    
    # Find lower bound on-grid
    lower_bound_a, _ = find_closest_grid_point(lower_bound_a, a_grid) 
    
    # Fix consumption and labour for constrained households at the currently 
    # valid borrowing limit
    c, n = jnp.where(a < lower_bound_a, 
                     solve_cn(we[:, None], 
                              rhs + trans[:, None] - lower_bound_a, 
                              sigma_c, sigma_l, vphi, Wa_p), 
                     jnp.array((c, n)))
    
    # Fix assets for constrained households at the currently valid borrowing 
    # limit
    a = jnp.where(a > lower_bound_a, 
                  a, 
                  lower_bound_a)
    
    # Calculate the marginal propensity to consume (MPC) out of a small
    # increase (given by db) of liquid wealth
    mpc = (interpolate(a, (a + db), c) - c) / db
    
    # Restrict MPCs to at most 1
    mpc = jnp.where(mpc > 1.0, 1.0, mpc)

    # calculate new MUC for next EGM step
    Wa = Rfull * c ** (-sigma_c)
    
    # return new MUC, asset holdings, consumption, labour supply and MPCs
    return Wa, a, c, n, mpc

###############################################################################
###############################################################################
# Function for determining the optimal consumption and labour supply choices
def cn(uc, w, sigma_c, sigma_l, vphi):
    """Optimal Consumption and Labour Supply.

    Given a guess/value for the marginal utility of consumption, u'(c), the 
    effective wage w and household parameters, this function returns the 
    optimal choices for consumption c and labour supply n. To do so, it makes 
    use of the FOCs for the household problem:
        
        I)  c = (beta*(R/pi{+1})c{+1}^{-sigma_c})^{-1/sigma_c}
        II) n = (c^{-sigma_c}*w*e/phi)^{1/sigma_l}
    """
    return jnp.array((uc ** (-1/sigma_c), (w * uc / vphi) ** (1/sigma_l)))

###############################################################################
###############################################################################
#  
def solve_cn(w, trans, sigma_c, sigma_l, vphi, uc_seed):
    uc = solve_uc(w, trans, sigma_c, sigma_l, vphi, uc_seed)
    return cn(uc, w, sigma_c, sigma_l, vphi)

###############################################################################
###############################################################################
#
def solve_uc_cond(carry):
    """Check Net Expenditure Condition.
    
    This function checks whether household net expenditures are significantly
    above zero. For constrained households this should be zero.
    
    Parameters:
    ----------
    carry           : contains net expenditures 

    Returns:
    ----------
    amax(ne) > 1e-8 : true if net expenditures are significantly above zero
    """
    ne, _, _ = carry
    return amax(ne) > 1e-8

###############################################################################
###############################################################################
# 
def solve_uc_body(carry):
    ne, log_uc, pars = carry
    ne, ne_p = netexp(log_uc, *pars)
    log_uc -= ne / ne_p
    return ne, log_uc, pars

###############################################################################
# 
def solve_uc(w, trans, sigma_c, sigma_l, vphi, uc_seed):
    """Solve for optimal uc given in log uc space.
    max_{c, n} c**(1-sigma_c) + vphi*n**(1+sigma_l) s.t. c = w*n + T
    """
    log_uc = jnp.log(uc_seed)
    pars = w, trans, sigma_c, sigma_l, vphi
    _, log_uc, _ = jax.lax.while_loop(solve_uc_cond, 
                                      solve_uc_body, 
                                      (uc_seed, log_uc, pars))
    return jnp.exp(log_uc)

###############################################################################
###############################################################################
# Function for calculating net expenditures 
def netexp(log_uc, w, trans, sigma_c, sigma_l, vphi):
    """Return net expenditure as a function of log uc and its derivative
    """
    c, n = cn(jnp.exp(log_uc), w, sigma_c, sigma_l, vphi)
    ne = c - w * n - trans

    # c and n have elasticities of -1/sigma_c and 1/sigma_l wrt log u'(c)
    c_loguc = -1/sigma_c * c
    n_loguc = 1/sigma_l * n
    netexp_loguc = c_loguc - w * n_loguc

    return ne, netexp_loguc

###############################################################################
###############################################################################
# Function for calculating the effective wage
def wages(w, e_grid):
    """Effective wage calculation.
    
    This function calculates the effective wage rate for each level of 
    household productivity for a given overall wage.
    
    Parameters:
    ----------
    w       : wage per effective labour input 
    e_grid  : vector with the levels of productivity 

    Returns:
    ----------
    we      : vector with the item-wise product of w and e_grid
    """
    we = w * e_grid
    return we

###############################################################################
###############################################################################
# Function for calculating the effective labour supply
def labor_supply(n, e_grid):
    """Effective labour supply calculation.
    
    This function calculates the effective labour supply for each level of 
    labour supply across all combinations of asset holdings and productivity 
    levels.
    
    Parameters:
    ----------
    n       : labour supply levels across the distribution 
    e_grid  : vector with the levels of productivity 

    Returns:
    ----------
    ne      : effective labour supplies
    """
    ne = e_grid[:, None] * n
    return ne


###############################################################################
###############################################################################
###############################################################################
# Functions common across HANK models

###############################################################################
###############################################################################
# Function for creating the asset grid###############################################################################
def create_grid(amax, n, amin, rho_a, amin_terminal, T=200):
    """Asset grid.
    
    This function calculates the dividends net of taxes/transfers the household
    receives. Dividends accrue due to monopoly profits in the model and taxes/
    transfers are due to the fiscal authority running a balanced budget.
    
    Parameters:
    ----------
    amax            : maximum asset holdings
    n               : number of grid points for the log grid
    amin            : initial borrowing limit
    rho_a           : persistence in borrowing limit shock
    amin_terminal   : terminal borrowing limit
    T               :

    Returns:
    ----------
    full_grid       : final asset grid
    len(full_grid)  : length of final asset grid
    """
    
    # Initialise a typical log grid
    initialise_log_grid = log_grid(amax, n, amin)
    
    # Create the full path of the borrowing limit from the initial borrowing 
    # limit to the terminal borrowing limit, using the given persistence of
    # the transition process
    path_borrowing_limit = [np.nan]*T # initialise empty container
    path_borrowing_limit[0] = amin # initialise first entry of the container
    
    if amin_terminal == 0:
        amin_terminal = -0.01
    
    for tt in range(T-1): # iterate forward by using the transition process of 
    # the borrowing limit
        path_borrowing_limit[tt+1] = round(amin_terminal*(path_borrowing_limit[tt]/amin_terminal)**rho_a, 6)
    
    path_borrowing_limit = [num for num in path_borrowing_limit if num < amin_terminal]
    path_borrowing_limit.append(amin_terminal) # ensure that terminal borrowing limit is included 
    path_borrowing_limit.append(0) # ensure that 0 is included
    path_borrowing_limit.pop(0) # delete double initial borrowing limit
    
    path_borrowing_limit_rev = [np.nan]*T 
    path_borrowing_limit_rev[0] = amin_terminal 
    
    for tt in range(T-1):
        path_borrowing_limit_rev[tt+1] = round(amin*(path_borrowing_limit_rev[tt]/amin)**rho_a, 6)
    
    path_borrowing_limit_rev = [num for num in path_borrowing_limit_rev if num > amin]
    path_borrowing_limit_rev.pop(0)
    
    # Combine initial log grid and grid values of borrowing limit path and 
    # sort the result
    full_grid = jnp.append(initialise_log_grid, 
                           jnp.array(path_borrowing_limit)).sort()
    
    full_grid = jnp.append(full_grid,
                            jnp.array(path_borrowing_limit_rev)).sort()
    
    full_grid = jnp.unique(full_grid)
    
    # Return the final grid and its length
    return full_grid, len(full_grid)

###############################################################################
###############################################################################
# Function for calculating transfers to household
def transfers(skills_stationary, Div, Tax, skills_grid):
    """Transfer calculation.
    
    This function calculates the dividends net of taxes/transfers the household
    receives according to its skill levels.
    """
    # Hardwired incidence rules are proportional to skill
    rule = skills_grid
    div = Div / jnp.sum(skills_stationary * rule) * rule # dividends
    tax = Tax / jnp.sum(skills_stationary * rule) * rule # taxes
    T = div - tax # dividends less taxes
    
    # Retrun net transfers
    return T
