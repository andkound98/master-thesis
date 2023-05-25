#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Andreas Koundouros, University of Bonn
Date: 26.04.2023

This file contains the functions for the HANK model with endogenous labour supply.
"""

import jax
from jax import device_put
import jax.numpy as jnp
import numpy as np
from grgrjax import jax_print, amax
from econpizza.utilities.interp import interpolate
from econpizza.utilities.grids import log_grid

#from custom_functions import find_closest_grid_point

def hh_init(a_grid, we, R, sigma_c, T):
    """The initialization for the value function
    """
    # Calculate cash-on-hand in order to derive marginal utility of consumption
    # from it, so to initialise the algorithm
    coh = R * a_grid[None, :] + we[:, None] + T[:, None]
    
    # Marginal utility of consumption
    Va = R * (0.1 * coh) ** (-sigma_c)
    
    return Va

def hh_borrowing(Va_p, a_grid, we, trans, R, beta, sigma_c, sigma_l, vphi, lower_bound_a, db):
    """A single backward step via EGM
    """

    # MUC as implied by next periods value function
    uc_nextgrid = beta * Va_p
    # back out consumption and labor supply from MUC
    c_nextgrid, n_nextgrid = cn(uc_nextgrid, we[:, None], sigma_c, sigma_l, vphi)

    # get consumption and labor supply in grid space
    lhs = c_nextgrid + a_grid[None, :] - we[:, None] * n_nextgrid - trans[:, None]
    rhs = R * a_grid

    c = interpolate(lhs, rhs, c_nextgrid)
    n = interpolate(lhs, rhs, n_nextgrid)

    # get todays distribution of assets
    a = rhs + we[:, None] * n + trans[:, None] - c
    
    #lower_a, _ = find_closest_grid_point(lower_bound_a, a_grid)
    
    lower_a = lower_bound_a
    
    # fix consumption and labor for constrained households
    c, n = jnp.where(a < lower_a, 
                     solve_cn(we[:, None], rhs + trans[:, None] - lower_a, sigma_c, sigma_l, vphi, Va_p), 
                     jnp.array((c, n)))
    
    # Fix assets where they would be below the borrowing constraint to the 
    # borrowing constraint, i.e. ensure that the borrowing constraint holds
    a = jnp.where(a < lower_a, 
                  lower_a, 
                  a)

    # Calculate the new marginal utility of consumption, to be used in the next
    # EGM step
    Va = R * c ** (-sigma_c)
    
    # Calculate mpc
    mpc = (interpolate(a, (a + db), c) - c) / db
    
    # Ensure that MPC is at most 1
    mpc = jnp.where(mpc > 1.0, 
                    1.0, 
                    mpc)

    return Va, a, c, n, mpc

def hh_borrowing_rbar(Va_p, a_grid, we, trans, R, Rcosts, beta, sigma_c, sigma_l, vphi, lower_bound_a, db):
    """A single backward step via EGM
    """

    # MUC as implied by next periods value function
    uc_nextgrid = beta * Va_p
    # back out consumption and labor supply from MUC
    c_nextgrid, n_nextgrid = cn(uc_nextgrid, we[:, None], sigma_c, sigma_l, vphi)

    # get consumption and labor supply in grid space
    lhs = c_nextgrid + a_grid[None, :] - we[:, None] * n_nextgrid - trans[:, None]
    
    Rbar = R + Rcosts
    
    rhs = a_grid
    rhs = jnp.where(a_grid < 0,
                    Rbar * a_grid, 
                    R * a_grid)

    c = interpolate(lhs, rhs, c_nextgrid)
    n = interpolate(lhs, rhs, n_nextgrid)

    # get todays distribution of assets
    a = rhs + we[:, None] * n + trans[:, None] - c
    
    #lower_a, _ = find_closest_grid_point(lower_bound_a, a_grid)
    
    lower_a = lower_bound_a
    
    # fix consumption and labor for constrained households
    c, n = jnp.where(a < lower_a, 
                     solve_cn(we[:, None], rhs + trans[:, None] - lower_a, sigma_c, sigma_l, vphi, Va_p), 
                     jnp.array((c, n)))
    
    # Fix assets where they would be below the borrowing constraint to the 
    # borrowing constraint, i.e. ensure that the borrowing constraint holds
    a = jnp.where(a < lower_a, 
                  lower_a, 
                  a)

    # Calculate the new marginal utility of consumption, to be used in the next
    # EGM step
    Va = c ** (-sigma_c)
    
    Va = jnp.where(a < lower_a, 
                   Rbar * Va, 
                   R * Va)
    
    # Calculate mpc
    mpc = (interpolate(a, (a + db), c) - c) / db
    
    # Ensure that MPC is at most 1
    mpc = jnp.where(mpc > 1.0, 
                    1.0, 
                    mpc)

    return Va, a, c, n, mpc


def cn(uc, w, sigma_c, sigma_l, vphi):
    """This function returns the optimal choices for consumption c and labour
    supply n as a function of the marginal utility of consumption u'(c), where 
    use is made of the FOCs of the households' optimisation problem. Note that 
    the input w is already the effective wage, i.e. the wage adjusted for 
    the household's productivity level. sigma_c, sigma_l and vphi are the 
    parameters of the utility function of the households.
    """
    return jnp.array((uc ** (-1/sigma_c), (w * uc / vphi) ** (1/sigma_l)))


def solve_cn(w, trans, sigma_c, sigma_l, vphi, uc_seed):
    uc = solve_uc(w, trans, sigma_c, sigma_l, vphi, uc_seed)
    return cn(uc, w, sigma_c, sigma_l, vphi)


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


def solve_uc_body(carry):
    ne, log_uc, pars = carry
    ne, ne_p = netexp(log_uc, *pars)
    log_uc -= ne / ne_p
    return ne, log_uc, pars


def solve_uc(w, trans, sigma_c, sigma_l, vphi, uc_seed):
    """Solve for optimal uc given in log uc space.
    max_{c, n} c**(1-sigma_c) + vphi*n**(1+sigma_l) s.t. c = w*n + T
    """
    log_uc = jnp.log(uc_seed)
    pars = w, trans, sigma_c, sigma_l, vphi
    _, log_uc, _ = jax.lax.while_loop(solve_uc_cond, solve_uc_body, (uc_seed, log_uc, pars))
    return jnp.exp(log_uc)


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


def transfers(pi_e, Div, Tax, e_grid):
    # hardwired incidence rules are proportional to skill; scale does not matter
    tax_rule, div_rule = e_grid, e_grid
    div = Div / jnp.sum(pi_e * div_rule) * div_rule
    tax = Tax / jnp.sum(pi_e * tax_rule) * tax_rule
    T = div - tax
    return T


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

##################################
##################################
##################################
##################################
def special_grid(amax, n, amin, rho_a, amin_terminal, T=200):
    initialise_log_grid = log_grid(amax, n, amin)
    
    path_borrowing_limit = [np.nan]*T
    path_borrowing_limit[0] = amin
    for tt in range(T-1):
        path_borrowing_limit[tt+1] = round(amin_terminal*(path_borrowing_limit[tt]/amin_terminal)**rho_a, 8)
    
    path_borrowing_limit = [num for num in path_borrowing_limit if num < amin_terminal]
    path_borrowing_limit.append(amin_terminal)
    path_borrowing_limit.append(0)
    path_borrowing_limit.pop(0)
    
    full_grid = jnp.append(initialise_log_grid, jnp.array(path_borrowing_limit)).sort()
    
    return full_grid, len(full_grid)