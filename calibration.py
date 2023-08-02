#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Andreas Koundouros
Date: 13.07.2023

This file calculates targets for the calibration of the employed HANK model. In
particular, the file downloads FRED data on government bonds, household debt, 
GDP and the GDP deflator in the US as well as Flow of Funds (FoF) data on 
household holdings of liquid assets in the US.

After obtaining the data, the code calculates the market value of government 
bonds in model units, household debt over output and household liquid assets in 
model units.

Note that a FRED API key is required to obtain the FRED data. For the FoF data,
the respective CSV file has to be downloaded 
(https://www.federalreserve.gov/releases/z1/). 
"""

###############################################################################
###############################################################################
# Import packages
import os # path creation
import pandas as pd # data wrangling
import numpy as np # data wrangling
import plotly.express as px # plotting
import plotly.io as pio # plotting
from full_fred.fred import Fred # for easy access to the FRED data base 

###############################################################################
###############################################################################
# Preliminaries
save_results = False # If true, results (tables and plots) are stored

pio.renderers.default = 'svg' # For plotting in the Spyder window

###############################################################################
###############################################################################
# FRED Data

###############################################################################
# Get path to API and initialise FRED package

# Get path
full_path_api = os.path.join('/', 
                             'Users', 
                             'andreaskoundouros', 
                             'Documents', 
                             'RA-Saidi', 
                             'Projekt-Sanktionen', 
                             'api_key_fred.txt')

# Initialise FRED package
fred = Fred(full_path_api)
fred.set_api_key_file(full_path_api)
        
# Get the FRED API
with open(full_path_api, 'r') as api_txt:
    my_FRED_api = api_txt.read() # Read the API from the text file 

###############################################################################
# Set required variables and respective series IDs
fred_variables = {'Government Bonds': 'MVGFD027MNFRBDAL', # Market Value of Gross Federal Debt
                  'Household Debt': 'HCCSDODNS', # Consumer Credit (Households and Nonprofit Organizations)
                  'GDP': 'GDP', # GDP (nominal, annualised)
                  'GDP Deflator': 'GDPDEF'} # GDP Deflator

###############################################################################
# Preliminaries 

# Empty data frame
FRED_data = pd.DataFrame()

# Dates for column of dates
dates = pd.date_range(start = "1940-01-01", # Some early starting value
                      end = "2023-06-01", # Some late ending value
                      freq = "MS") # Monthly frequency

###############################################################################
# Create full data set

# Obtain and merge observations into data frame
for key in fred_variables.keys():
    fred_dt = fred.get_series_df(f'{fred_variables[key]}')[['date', 'value']] # Get observations for given series ID
    fred_dt.columns = ['Date', f'{key}'] # Rename columns 
    fred_dt.loc[:, 'Date'] = pd.to_datetime(fred_dt['Date']) # Convert date column
    fred_dt.replace('.', np.nan, inplace=True) # Replace potential '.' in the data
    fred_dt[f'{key}'] = fred_dt[f'{key}'].astype(float) # Convert object into float
    
    # Merge observations into data frame
    if FRED_data.empty == True: 
        FRED_data = fred_dt
    else:
        FRED_data = pd.merge(FRED_data, 
                             fred_dt,
                             on='Date', 
                             how='left')
    
# Create quarterly observation for monthly Government Bonds
quarterly_bonds = []
for i in range(len(FRED_data)):
    if pd.isnull(FRED_data.at[i, 'Household Debt']):
        quarterly_bonds.append(None)
    else:
        three_month_avg = FRED_data.iloc[i:i+3, 1].astype(float).mean() # Take three-month average to obtain quarterly data
        quarterly_bonds.append(three_month_avg)
FRED_data["Government Bonds"] = quarterly_bonds 

# Drop entries before specified date
start_date = '1951-10-01' # Full observations start in 1951:Q4
#start_date = '2013-01-01' # Include only more recent observation
mask = FRED_data['Date'] >=  pd.to_datetime(start_date)
FRED_data = FRED_data[mask]

# Drop NAs
FRED_data = FRED_data.dropna()

###############################################################################
###############################################################################
# FoF Data

###############################################################################
# Set required variables and respective IDs
fof_variables = {'Date': 'date',
                 'Foreign deposits': 'LM153091003.Q',
                 'Checkable deposits and currency': 'FL153020005.Q',
                 'Time and savings deposits': 'FL153030005.Q',
                 'Money market fund shares': 'FL153034005.Q',
                 'Debt securities': 'LM154022005.Q',
                 'Corporate equities': 'LM153064105.Q',
                 'Mutual fund shares': 'LM153064205.Q'}

# Read the CSV file with the FoF Table B.101 data into a pandas DataFrame
path_to_fof = os.path.join(os.getcwd(),
                           'Data',
                           'FoF_B101.csv')
fof_data = pd.read_csv(path_to_fof)
    
# Get the list of columns to keep from the specified dictionary
keep_col = list(fof_variables.values())
    
# Keep only the specified columns
fof_data = fof_data[keep_col]

# Rename the date column
fof_data.rename(columns={'date': 'Date'}, inplace=True)

def convert_quarter_to_datetime(quarter_str):
    year, quarter = quarter_str.split(':')
    quarter_number = int(quarter[1:])
    quarter_start_month = (quarter_number - 1) * 3 + 1
    return pd.to_datetime(f"{year}-{quarter_start_month:02d}-01")

fof_data['Date'] = fof_data['Date'].apply(convert_quarter_to_datetime)

# Drop entries before specified date
mask = fof_data['Date'] >=  pd.to_datetime(start_date)
fof_data = fof_data[mask]

fof_data[fof_data.columns.drop('Date')] = fof_data[fof_data.columns.drop('Date')].apply(pd.to_numeric)
fof_data[fof_data.columns.drop('Date')] = fof_data[fof_data.columns.drop('Date')] / 1000

# Create variable of total liquid assets as simple sum of all the kept 
# variables
fof_data['Total Liquid Assets'] = fof_data.iloc[:, 1:].sum(axis=1)

###############################################################################
###############################################################################
# Obtain one single data set 
data = pd.merge(FRED_data, 
                fof_data[['Date', 'Total Liquid Assets']],
                on='Date', 
                how='left')

###############################################################################
###############################################################################
# Final calcualtions

# Define steady state output of the HANK model (needed for conversion into 
# units of the model)
ySS = 0.9129 # 0.9410
#ySS = 0.5310

# Deflate Government Bonds and divide by deflated ANNUALISED GDP, then convert
# to quarterly; finally, obtain it in terms of the steady-state model output
data['B'] = 4*ySS*((data["Government Bonds"]/data["GDP Deflator"])/(data["GDP"]/data["GDP Deflator"]))

# Deflate Household Debt and divide by deflated ANNUALISED GDP, then convert to
# quarterly; finally, obtain it in terms of the steady-state model output
data['D_y'] = 4*((data["Household Debt"]/data["GDP Deflator"])/(data["GDP"]/data["GDP Deflator"]))

# 
data['L'] = 4*ySS*((data["Total Liquid Assets"]/data["GDP Deflator"])/(data["GDP"]/data["GDP Deflator"]))

###############################################################################
###############################################################################
# Calibration parameters/targets

# Obtain the means of Government Bonds and Household Debt over the full sample
mean_B = data['B'].mean()
mean_D_y = data['D_y'].mean()
mean_L = data['L'].mean()

# Print the results
print(round(mean_B,2)) # Government bond supply, B
print(round(mean_D_y,2)) # Steady-state ratio of household-debt to output, D_{ss}/y_{ss}
print(round(mean_L,2)) # Total liquid assets, B

###############################################################################
###############################################################################
# Plot time series of total liquid assets, market value of government debt and
# consumer credit

# Total Liquid Assets
fig_l = px.line(data,
                x = 'Date',
                y = 'L',
                color_discrete_sequence=[px.colors.qualitative.D3[0]])
fig_l.update_layout(title='', # empty title
                   xaxis_title=None, # x-axis labeling
                   yaxis_title='Quarterly Model Units', # y-axis labeling
                   legend=dict(orientation="h", # horizontal legend
                               yanchor="bottom", y=1.02, 
                               xanchor="right", x=1), 
                   legend_title=None, 
                   plot_bgcolor='whitesmoke', 
                   margin=dict(l=15, r=15, t=5, b=5),
                   font=dict(family="Times New Roman", # adjust font
                             size=20,
                             color="black"))
fig_l.add_hline(y=mean_L, line_dash="dot",
                line_color=px.colors.qualitative.D3[1],
                annotation_text=f'{round(mean_L,2)}', 
                annotation_position='bottom right',
                annotation_font_size=20,
                annotation_font_color='black')
fig_l.update_traces(line=dict(width=3))
fig_l.show()

# Save figure
path_plot_l = os.path.join(os.getcwd(),
                           'Results',
                           'FRED_l.svg')
fig_l.write_image(path_plot_l)

# Government Bonds
fig_b = px.line(data,
                x = 'Date',
                y = 'B',
                color_discrete_sequence=[px.colors.qualitative.D3[0]])
fig_b.update_layout(title='', # empty title
                   xaxis_title=None, # x-axis labeling
                   yaxis_title='Quarterly Model Units', # y-axis labeling
                   legend=dict(orientation="h", # horizontal legend
                               yanchor="bottom", y=1.02, 
                               xanchor="right", x=1), 
                   legend_title=None, 
                   plot_bgcolor='whitesmoke', 
                   margin=dict(l=15, r=15, t=5, b=5),
                   font=dict(family="Times New Roman", # adjust font
                             size=20,
                             color="black"))
fig_b.add_hline(y=mean_B, line_dash="dot",
                line_color=px.colors.qualitative.D3[1],
                annotation_text=f'{round(mean_B,2)}', 
                annotation_position='bottom right',
                annotation_font_size=20,
                annotation_font_color='black')
fig_b.update_traces(line=dict(width=3))
fig_b.show()

# Save figure
path_plot_b = os.path.join(os.getcwd(),
                           'Results',
                           'FRED_b.svg')
fig_b.write_image(path_plot_b)

# Household Debt
fig_d = px.line(data,
                x = 'Date',
                y = 'D_y',
                color_discrete_sequence=[px.colors.qualitative.D3[0]])
fig_d.update_layout(title='', # empty title
                   xaxis_title=None, # x-axis labeling
                   yaxis_title='Fraction of Quarterly GDP', # y-axis labeling
                   legend=dict(orientation="h", # horizontal legend
                               yanchor="bottom", y=1.02, 
                               xanchor="right", x=1), 
                   legend_title=None, 
                   plot_bgcolor='whitesmoke', 
                   margin=dict(l=15, r=15, t=5, b=5),
                   font=dict(family="Times New Roman", # adjust font
                             size=20,
                             color="black"))
fig_d.add_hline(y=mean_D_y, line_dash="dot",
                line_color=px.colors.qualitative.D3[1],
                annotation_text=f'{round(mean_D_y,2)}', 
                annotation_position='bottom right',
                annotation_font_size=20,
                annotation_font_color='black')
fig_d.update_traces(line=dict(width=3))
fig_d.show()

# Save figure
path_plot_d = os.path.join(os.getcwd(),
                           'Results', 
                           'FRED_d.svg')
fig_d.write_image(path_plot_d)
