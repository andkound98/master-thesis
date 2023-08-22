#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Andreas Koundouros, University of Bonn
Date: 26.04.2023

This file contains the functions for the HANK models with and without
endogenous labour supply.
"""

###############################################################################
###############################################################################
# Import packages
import jax
from jax import device_put
import jax.numpy as jnp
import numpy as np
from grgrjax import jax_print, amax
from econpizza.utilities.interp import interpolate
from econpizza.utilities.grids import log_grid

from custom_functions import find_closest_grid_point

###############################################################################
###############################################################################
###############################################################################
# Functions for HANK without Endogenous Labour Supply

###############################################################################
# Function to initialise EGM
def egm_init(a_grid, skills_grid):
    """EGM initialisation.
    
    This function initialises the marginal utility of consumption as an array
    with some reasonably small number. The dimensions of the array are: number
    of skills times number of asset grid points. 
    With this, the EGM can start iterating backwards to calculate the asset and
    consumption policy functions.
    
    Parameters:
    ----------
    a_grid       : asset grid
    skills_grid  : skills grid

    Returns:
    ----------
    array of 1e-2: initialised marginal utility of consumption
    """
    return jnp.ones((skills_grid.shape[0], a_grid.shape[0]))*1e-2

###############################################################################
# Function for a single EGM step
def egm_step(Wa_p, a_grid, skills_grid, w, n, T, R, Rminus, beta, sigma_c, sigma_l, da, lower_bound_a):    
    """One Step of the Endogenous Gridpoint Method (EGM).
    
    This function takes a single backward step via EGM. It is iterated on 
    backwards in order to obtain optimal consumption and asset holding 
    policies.
    
    Parameters:
    ----------
    Wa_p        : next period's marginal continuation value
    a_grid      : asset grid
    skills_grid : skills grid
    w           : wage 
    n           : labour hours
    T           : dividends net of taxes
    R           : real interest rate
    beta        : discount factor
    sigma_c     : risk aversion
    sigma_l     : inverse Frisch elasticity of labour supply
    db          : step size in MPC calculation
    lower_bound_a:borrowing limit 

    Returns:
    ----------
    Wa          : this period's marginal continuation value
    a           : asset policy
    c           : consumption policy
    mpc         : marginal propensity to consume
    """

    # MUC as implied by next periods value function
    ux_nextgrid = beta * Wa_p
    
    # calculate labor income
    labor_inc = skills_grid[:, None]*n*w 

    # next period's consumption from MUC and MU of labor
    c_nextgrid = ux_nextgrid**(-1/sigma_c) + labor_inc/(1 + sigma_l)
    
    # Get full interest rate schedule
    Rfull = jnp.where(a_grid < 0,
                      Rminus,
                      R)
    
    # get consumption in grid space
    lhs = c_nextgrid - labor_inc + a_grid[None, :] - T[:, None]
    rhs = Rfull * a_grid
    c = interpolate(lhs, rhs, c_nextgrid)

    # get todays distribution of assets
    a = rhs + labor_inc + T[:, None] - c
    
    # find lower bound on-grid
    lower_bound_a, _ = find_closest_grid_point(lower_bound_a, a_grid) 
    
    # fix consumption and labor for constrained households at the current 
    # borrowing limit
    c = jnp.where(a < lower_bound_a, 
                  labor_inc + rhs + T[:, None] - lower_bound_a, 
                  c)
    a = jnp.where(a < lower_bound_a, 
                  lower_bound_a, 
                  a)
    
    # Calculate the marginal propensity to consume (MPC) out of a small
    # increase (given by db) of liquid wealth
    mpc = (interpolate(a, (a + da), c) - c) / da
    
    # Ensure that MPC is at most 1
    mpc = jnp.where(mpc > 1.0, 
                    1.0, 
                    mpc)
    
    # calculate new MUC for next EGM step
    Wa = Rfull * (c - labor_inc/(1 + sigma_l)) ** (-sigma_c)
    
    # return new MUC, asset holdings, consumption and MPCs
    return Wa, a, c, mpc

###############################################################################
###############################################################################
###############################################################################
# Functions for HANK with Endogenous Labour Supply

###############################################################################
# Function to initialise EGM

def egm_init_labour(a_grid, we, R, Rminus, sigma_c, T):
    """EGM initialisation.
    
    This function initialises the marginal utility of consumption as an array
    with some reasonably small number. The dimensions of the array are: number
    of skills times number of asset grid points. 
    With this, the EGM can start iterating backwards to calculate the asset,
    consumption and labour policy functions.
    
    Parameters:
    ----------
    a_grid          : asset grid
    we              : effective wage 
    R               : real interest rate
    sigma_c         : risk aversion
    T               : dividends net of taxes/transfers

    Returns:
    ----------
    Wa              : initialised marginal continuation value
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
# Function for a single EGM step

def egm_step_labour(Wa_p, a_grid, we, trans, R, Rminus, beta, sigma_c, sigma_l, vphi, db, lower_bound_a):
    """A single backward step via EGM
    """

    # MUC as implied by next periods value function
    uc_nextgrid = beta * Wa_p
    
    # back out consumption and labor supply from MUC
    c_nextgrid, n_nextgrid = cn(uc_nextgrid, we[:, None], sigma_c, sigma_l, vphi)
    
    # Get full interest rate schedule
    Rfull = jnp.where(a_grid < 0,
                      Rminus,
                      R)

    # get consumption and labor supply in grid space
    lhs = c_nextgrid - we[:, None] * n_nextgrid + a_grid[None, :] - trans[:, None]
    rhs = Rfull * a_grid
    c = interpolate(lhs, rhs, c_nextgrid)
    n = interpolate(lhs, rhs, n_nextgrid)

    # get todays distribution of assets
    a = rhs + we[:, None] * n + trans[:, None] - c
    
    # find lower bound on-grid
    lower_bound_a, _ = find_closest_grid_point(lower_bound_a, a_grid) 
    
    # fix consumption and labour for constrained households
    c, n = jnp.where(a < lower_bound_a, 
                     solve_cn(we[:, None], 
                              rhs + trans[:, None] - lower_bound_a, 
                              sigma_c, sigma_l, vphi, Wa_p), 
                     jnp.array((c, n)))
    
    # fix asset holdings for constrained households
    a = jnp.where(a > lower_bound_a, 
                  a, 
                  lower_bound_a)
    
    # Calculate mpc
    mpc = (interpolate(a, (a + db), c) - c) / db
    
    # Ensure that MPC is at most 1
    mpc = jnp.where(mpc > 1.0,
                    1.0, 
                    mpc)

    # calculate new MUC for next EGM step
    Wa = Rfull * c ** (-sigma_c)
    
    # return new MUC, asset holdings, consumption, labour supply and MPCs
    return Wa, a, c, n, mpc

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

    Parameters:
    ----------
    uc          : guess/value for the marginal utility of consumption
    w           : effective wage (productivity times aggregate wage rate)
    sigma_c     : coefficient of risk aversion
    sigma_l     : inverse Frisch elasticity
    vphi        : coefficient on disutility of labour

    Returns:
    ----------
    uc ** (-1/sigma_c)              : optimal consumption given next period's 
                                      marginal utility of consumption
    w * uc / vphi) ** (1/sigma_l)   : optimal labour supply given next period's 
                                      marginal utility of consumption and the 
                                      effective wage
    """
    return jnp.array((uc ** (-1/sigma_c), (w * uc / vphi) ** (1/sigma_l)))

###############################################################################
#  

def solve_cn(w, trans, sigma_c, sigma_l, vphi, uc_seed):
    uc = solve_uc(w, trans, sigma_c, sigma_l, vphi, uc_seed)
    return cn(uc, w, sigma_c, sigma_l, vphi)

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
# Common Functions

###############################################################################
# Function for creating the asset grid

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
# Function for calculating transfers to household

def transfers(skills_stationary, Div, Tax, skills_grid):
    """Transfer calculation.
    
    This function calculates the dividends net of taxes/transfers the household
    receives. Dividends accrue due to monopoly profits in the model and taxes/
    transfers are due to the fiscal authority running a balanced budget.
    
    Parameters:
    ----------
    skills_stationary       : 
    Div                     :
    Tax                     :
    skills_grid             : 

    Returns:
    ----------
    T                       : 
    """
    
    # hardwired incidence rules are proportional to skill; scale does not matter
    rule = skills_grid
    div = Div / jnp.sum(skills_stationary * rule) * rule
    tax = Tax / jnp.sum(skills_stationary * rule) * rule
    T = div - tax
    return T
