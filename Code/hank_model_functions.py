# -*- coding: utf-8 -*-

"""functions for the one-asset HANK model with labor choice. Heavily inspired by https://github.com/shade-econ/sequence-jacobian/#sequence-space-jacobian
"""

import jax
from jax import device_put
import jax.numpy as jnp
import numpy as np
from grgrjax import jax_print, amax
from econpizza.utilities.interp import interpolate
from econpizza.utilities.grids import log_grid

from custom_functions import find_closest_grid_point


def hh_init(a_grid, we, R, sigma_c, T):
    """The initialization for the value function
    """
    
    # Calculate cash-on-hand in order to derive marginal utility of consumption
    # from it, so to initialise the algorithm
    coh = R * a_grid[None, :] + we[:, None] + T[:, None]
    
    # Marginal utility of consumption
    Va = R * (0.1 * coh) ** (-sigma_c)
    
    return Va


# def hh(Va_p, a_grid, we, trans, R, beta, sigma_c, sigma_l, vphi):
#     """A single backward step via EGM
#     """

#     # MUC as implied by next periods value function
#     uc_nextgrid = beta * Va_p
#     # back out consumption and labor supply from MUC
#     c_nextgrid, n_nextgrid = cn(uc_nextgrid, we[:, None], sigma_c, sigma_l, vphi)

#     # get consumption and labor supply in grid space
#     lhs = c_nextgrid - we[:, None] * n_nextgrid + \
#         a_grid[None, :] - trans[:, None]
#     rhs = R * a_grid

#     c = interpolate(lhs, rhs, c_nextgrid)
#     n = interpolate(lhs, rhs, n_nextgrid)

#     # get todays distribution of assets
#     a = rhs + we[:, None] * n + trans[:, None] - c
    
#     # 
#     # fix consumption and labor for constrained households
#     c, n = jnp.where(a < a_grid[0], 
#                      solve_cn(we[:, None], rhs + trans[:, None] - a_grid[0], sigma_c, sigma_l, vphi, Va_p), 
#                      jnp.array((c, n)))
    
#     # Fix assets where they would be below the borrowing constraint to the 
#     # borrowing constraint, i.e. ensure that the borrowing constraint holds
#     a = jnp.where(a > a_grid[0], a, a_grid[0])

#     # Calculate the new marginal utility of consumption, to be used in the next
#     # EGM step
#     Va = R * c ** (-sigma_c)

#     return Va, a, c, n

def hh_borrowing(Va_p, a_grid, we, trans, R, beta, sigma_c, sigma_l, vphi, lower_bound_a, db):
    """A single backward step via EGM
    """

    # MUC as implied by next periods value function
    uc_nextgrid = beta * Va_p
    # back out consumption and labor supply from MUC
    c_nextgrid, n_nextgrid = cn(uc_nextgrid, we[:, None], sigma_c, sigma_l, vphi)

    # get consumption and labor supply in grid space
    lhs = c_nextgrid - we[:, None] * n_nextgrid + \
        a_grid[None, :] - trans[:, None]
    rhs = R * a_grid

    c = interpolate(lhs, rhs, c_nextgrid)
    n = interpolate(lhs, rhs, n_nextgrid)

    # get todays distribution of assets
    a = rhs + we[:, None] * n + trans[:, None] - c
    
    lower_a, _ = find_closest_grid_point(lower_bound_a, a_grid)
    
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

def hh_borrowing_new(Va_p, a_grid, we, trans, R, Rbar, beta, sigma_c, sigma_l, vphi, lower_bound_a, db):
    """A single backward step via EGM
    """

    # MUC as implied by next periods value function
    uc_nextgrid = beta * Va_p
    # back out consumption and labor supply from MUC
    c_nextgrid, n_nextgrid = cn(uc_nextgrid, we[:, None], sigma_c, sigma_l, vphi)

    # get consumption and labor supply in grid space
    lhs = c_nextgrid - we[:, None] * n_nextgrid + \
        a_grid[None, :] - trans[:, None]
    
    rhs = a_grid
    rhs = jnp.where(a_grid < 0,
                    Rbar * a_grid, 
                    R * a_grid)

    c = interpolate(lhs, rhs, c_nextgrid)
    n = interpolate(lhs, rhs, n_nextgrid)

    # get todays distribution of assets
    a = rhs + we[:, None] * n + trans[:, None] - c
    
    lower_a, _ = find_closest_grid_point(lower_bound_a, a_grid)
    
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

##### NEW
def new_special_grid(amax, n, amin, cutoff=False, cutoff_value=0):
    if cutoff == True:
        grid = log_grid(amax, n, amin)
        grid = grid[cutoff_value:]
    else:
        grid = log_grid(amax, n, amin)

    return grid


def ex_grid(a_min_1, a_min_2, a_max, n_1, n_2):
    grid_1 = jnp.array(np.linspace(a_min_1, a_min_2, n_1))
    grid_1 = jnp.delete(grid_1, -1)
    
    grid_2 = log_grid(a_max, n_2 + 1, a_min_2)
    
    final_grid = jnp.append(grid_1, grid_2)
    
    return final_grid

def import_gl_grid():
    import scipy.io
    import numpy as np
    
    path_to_file = '/Users/andreaskoundouros/Documents/Uni-Masterarbeit/Literature/2017 Guerrieri Lorenzoni/Replication Codes/gl_ex_grid.mat'

    # Load the .mat file into a dictionary
    mat_data = scipy.io.loadmat(path_to_file)

    # Extract the vectors and matrix from the dictionary
    gl_y = mat_data['theta']
    gl_pi = mat_data['pr']
    gl_Pi = mat_data['Pr']
    
    gl_y_new = jax.device_put(np.squeeze(gl_y))
    gl_pi_new = jax.device_put(np.squeeze(gl_pi))
    gl_Pi_new = jax.device_put(np.squeeze(gl_Pi))
    
    return gl_y_new, gl_pi_new, gl_Pi_new