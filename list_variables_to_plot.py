#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Andreas Koundouros
Date: 25.08.2023

This file allows to specify a dictionary with lists of variables from the 
employed models to be plotted.
"""

# Selection of aggregate variables
aggregate_variables = [['phi', 'Borrowing Limit', 'Model Units'],
                       ['kappa', 'Interest Rate Wedge', 'Annualised Rate'],
                       ['C', 'Consumption', 'Percent Deviation'], 
                       ['N', 'Labour Hours', 'Percent Deviation'],
                       ['w', 'Wage', 'Percent Deviation'], 
                       ['mc', 'Marginal Costs', 'Percent Deviation'],
                       ['pi', 'Inflation', 'Annualised Net Rate'],
                       ['Rr', 'Ex-Post Real Interest Rate', 'Annualised Net Rate'],
                       ['Rrminus', 'Interest Rate on Neg. Assets', 'Annualised Net Rate'],
                       ['R', 'Nominal Interest Rate',
                        'Rn', 'Notional Interest Rate', 'Annualised Net Rate'],  
                       ['div', 'Dividends', 'Percent Deviation'],
                       ['tau', 'Taxes', 'Percent Deviation'],
                       ['D', 'Household Debt', 'Model Units'], 
                       ['DY', 'Household Debt-to-GDP', 'Percent of Output']]

# Selection of debt variables (in case these should be plotted additionally)
debt_variables = [['D', 'Household Debt', 'Model Units'], 
                  ['DY', 'Household Debt-to-GDP', 'Percent of Output']]

# Selection of cross-section variables
distributional_variables = [['Bot25A', 'Bottom-25% Assets Share', 'Percent'],
                            ['Bot25C', 'Bottom-25% Consumption Share', 'Percent'],
                            ['Bot50A', 'Bottom-50% Assets Share', 'Percent'],
                            ['Bot50C', 'Bottom-50% Consumption Share', 'Percent'],
                            ['Top25A', 'Top-25% Assets Share', 'Percent'],
                            ['Top25C', 'Top-25% Consumption Share', 'Percent'],
                            ['Top10A', 'Top-10% Assets Share', 'Percent'],
                            ['Top10C', 'Top-10% Consumption Share', 'Percent']]

# Create dictionary of the two lists
dict_of_variables_to_plot = {'aggregate': aggregate_variables,
                             'debt': debt_variables,
                             'cross_sec': distributional_variables}
