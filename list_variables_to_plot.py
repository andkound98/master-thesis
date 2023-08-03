#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Andreas Koundouros
Date: 21.06.2023

This file creates lists of variables to be plotted.
"""

# List of aggregate variables to be plotted
aggregate_variables = [['phi', 'Borrowing Limit', 'Model Units'],
                       ['Rbar', 'Interest Rate Wedge', 'Annualised Rate'],
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

# List of distributional variables to be plotted
distributional_variables = [['Top10C', 'Top10% Consumption Share', 'Percent'],
                            ['Bot25C', 'Bottom25% Consumption Share', 'Percent'],
                            ['Top10A', 'Top10% Assets Share', 'Percent'],
                            ['Bot25A', 'Bottom25% Assets Share', 'Percent']]

# Create dictionary of the two lists
dict_of_variables_to_plot = {'aggregate': aggregate_variables,
                             'cross_sec': distributional_variables}
