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
                       ['beta', 'Discount Factor', 'Discount Factor'],
                       ['C', 'Consumption', '% Deviation'], 
                       ['N', 'Labour Hours', '% Deviation'],
                       ['w', 'Wage', '% Deviation'], 
                       ['mc', 'Marginal Costs', '% Deviation'],
                       ['pi', 'Inflation', 'Annualised Net Rate'],
                       ['Rr', 'Ex-Post Real Interest Rate', 'Annualised Net Rate'],
                       ['Rrminus', 'Interest Rate on Neg. Assets', 'Annualised Net Rate'],
                       ['R', 'Nominal Interest Rate',
                        'Rn', 'Notional Interest Rate', 'Annualised Net Rate'],  
                       ['div', 'Dividends', '% Deviation'],
                       ['tau', 'Taxes', '% Deviation'],
                       ['D', 'Household Debt', 'Model Units'], 
                       ['DY', 'Household Debt-to-GDP', '% of Output']]

# List of distributional variables to be plotted
distributional_variables = [['Top10C', 'Top10% Consumption Share', '%'],
                            ['Bot25C', 'Bottom25% Consumption Share', '%'],
                            ['Top10A', 'Top10% Assets Share', '%'],
                            ['Bot25A', 'Bottom25% Assets Share', '%']]

# Create dictionary of the two lists
dict_of_variables = {'aggregate': aggregate_variables,
                     'cross_sec': distributional_variables}
