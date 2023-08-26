#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Andreas Koundouros
Date: 26.08.2023

This file calculates targets for the calibration of the models in the thesis 
and plots some of the time series. 

Some of the data are downloaded from FRED. To obtain the FRED data, a FRED API 
key is required (see https://fred.stlouisfed.org/docs/api/api_key.html). The 
user must set the path 'full_path_api' to where he/she has stored the text file
with the key. The rest of the data are from the Flow of Funds (FoF), which were 
dowloaded from https://www.federalreserve.gov/releases/z1/ and are stored as a 
CSV file in the 'Data' folder of this project.
"""

###############################################################################
###############################################################################
###############################################################################
# Packages
import os # path management
import plotly.io as pio # plot settings
import pandas as pd # data wrangling
import numpy as np # data wrangling
import plotly.express as px # plotting
from full_fred.fred import Fred # convenient package for FRED access

###############################################################################
###############################################################################
###############################################################################
# Imports

from custom_functions import convert_quarter_to_datetime # function to convert to datetime

###############################################################################
# Preliminaries
save_results = False # True: save results 

pio.renderers.default = 'svg' # For plotting in the Spyder window

###############################################################################
###############################################################################
###############################################################################
# FRED Data

###############################################################################
# Get path to API and initialise FRED package

# Get path
full_path_api = os.path.join('api_key_fred.txt') # SET THE PATH TO YOUR FRED API HERE

# Initialise FRED package
fred = Fred(full_path_api)
fred.set_api_key_file(full_path_api)
        
# Get the FRED API
with open(full_path_api, 'r') as api_txt:
    my_FRED_api = api_txt.read() # Read the API from the text file 

###############################################################################
# Set required variables and respective series IDs
fred_variables = {'Government Bonds': 'MVGFD027MNFRBDAL', # Market Value of Gross Federal Debt
                  'Household Debt': 'HCCSDODNS', # Households and Nonprofit Organizations; Consumer Credit; Liability, Level
                  'GDP': 'GDP', # GDP (nominal, annualised)
                  'GDP Deflator': 'GDPDEF'} # GDP Deflator

###############################################################################
# Empty data frame
FRED_data = pd.DataFrame()

# Initialise column of dates
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
                             on='Date',how='left')
    
# Create quarterly observation for monthly Government Bonds
quarterly_bonds = []
for i in range(len(FRED_data)):
    if pd.isnull(FRED_data.at[i, 'Household Debt']):
        quarterly_bonds.append(None)
    else:
        three_month_avg = FRED_data.iloc[i:i+3, 1].astype(float).mean() # Take three-month average to obtain quarterly data
        quarterly_bonds.append(three_month_avg)
FRED_data["Government Bonds"] = quarterly_bonds 

# Drop entries before 1951:Q4 as all series are available from then
start_date = '1951-10-01'
mask = FRED_data['Date'] >=  pd.to_datetime(start_date)
FRED_data = FRED_data[mask]

# Drop NAs
FRED_data = FRED_data.dropna()

###############################################################################
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
path_to_fof = os.path.join(os.getcwd(),'Data','FoF_B101.csv')
fof_data = pd.read_csv(path_to_fof)
    
# # Keep only the specified columns
keep_col = list(fof_variables.values())
fof_data = fof_data[keep_col]

# Rename the date column and convert to datetime object
fof_data.rename(columns={'date': 'Date'}, inplace=True)
fof_data['Date'] = fof_data['Date'].apply(convert_quarter_to_datetime)

# Drop entries before specified date
mask = fof_data['Date'] >=  pd.to_datetime(start_date)
fof_data = fof_data[mask]

# Make all columns (except Date) numeric and divide by 1000
fof_data[fof_data.columns.drop('Date')] = fof_data[fof_data.columns.drop('Date')].apply(pd.to_numeric)
fof_data[fof_data.columns.drop('Date')] = fof_data[fof_data.columns.drop('Date')] / 1000

# Create variable of total liquid assets as sum of all the kept variables
fof_data['Total Liquid Assets'] = fof_data.iloc[:, 1:].sum(axis=1)

###############################################################################
###############################################################################
###############################################################################
# Combine data

# Obtain one single data set 
data = pd.merge(FRED_data,                                  # FRED data
                fof_data[['Date', 'Total Liquid Assets']],  # FoF data
                on='Date', how='left')

###############################################################################
###############################################################################
###############################################################################
# Calculations and visualisation

# Define steady-state output of the HANK model (needed for conversion into 
# units of the model)
ySS = 0.9129 # baseline model (hank_baseline)
#ySS = 0.944 # extended model with CRRA preferences and endogenous labour supply (hank_end_L)

# Deflate Total Liquid Assets and divide by deflated annualised GDP, then 
# convert to quarterly; finally, obtain it in terms of the steady-state model 
#output
data['L'] = 4*ySS*((data["Total Liquid Assets"]/data["GDP Deflator"])/(data["GDP"]/data["GDP Deflator"]))

# Deflate Household Debt and divide by deflated annualised GDP, then convert to
# quarterly
data['D_y'] = 4*((data["Household Debt"]/data["GDP Deflator"])/(data["GDP"]/data["GDP Deflator"]))

# Deflate Government Bonds and divide by deflated annualised GDP, then convert
# to quarterly; finally, obtain it in terms of the steady-state model output
data['B'] = 4*ySS*((data["Government Bonds"]/data["GDP Deflator"])/(data["GDP"]/data["GDP Deflator"]))

###############################################################################
# Obtain the means over the full sample
mean_B = data['B'].mean()
mean_D_y = data['D_y'].mean()
mean_L = data['L'].mean()

# Print the (rounded) results
print(round(mean_L,2)) # Total Liquid Assets, for B
print(round(mean_D_y,2)) # Household Debt, for D_{ss}/y_{ss}
print(round(mean_B,2)) # Government Bonds, for B

###############################################################################
###############################################################################
# Plot time series of data

# Total Liquid Assets
fig_l = px.line(data,x = 'Date',y = 'L',
                color_discrete_sequence=[px.colors.qualitative.D3[0]])
fig_l.update_layout(title='',
                   xaxis_title=None,
                   yaxis_title='Quarterly Model Units',
                   legend=dict(orientation="h",
                               yanchor="bottom", y=1.02, 
                               xanchor="right", x=1), 
                   legend_title=None, plot_bgcolor='whitesmoke', 
                   margin=dict(l=15, r=15, t=5, b=5),
                   font=dict(family="Times New Roman",size=20,color="black"))
fig_l.add_hline(y=mean_L, line_dash="dot", # add mean of series
                line_color=px.colors.qualitative.D3[1],
                annotation_text=f'{round(mean_L,2)}', 
                annotation_position='bottom right',
                annotation_font_size=20,
                annotation_font_color='black')
fig_l.update_traces(line=dict(width=3))
fig_l.show()

# Save figure
path_plot_l = os.path.join(os.getcwd(),'Results','calibration_liquid.svg')
fig_l.write_image(path_plot_l)

# Government Bonds
fig_b = px.line(data,x = 'Date',y = 'B',
                color_discrete_sequence=[px.colors.qualitative.D3[0]])
fig_b.update_layout(title='',
                   xaxis_title=None,
                   yaxis_title='Quarterly Model Units',
                   legend=dict(orientation="h",
                               yanchor="bottom", y=1.02, 
                               xanchor="right", x=1), 
                   legend_title=None, plot_bgcolor='whitesmoke', 
                   margin=dict(l=15, r=15, t=5, b=5),
                   font=dict(family="Times New Roman",size=20,color="black"))
fig_b.add_hline(y=mean_B, line_dash="dot", # add mean of series
                line_color=px.colors.qualitative.D3[1],
                annotation_text=f'{round(mean_B,2)}', 
                annotation_position='bottom right',
                annotation_font_size=20,
                annotation_font_color='black')
fig_b.update_traces(line=dict(width=3))
fig_b.show()

# Save figure
path_plot_b = os.path.join(os.getcwd(),'Results','calibration_b.svg')
fig_b.write_image(path_plot_b)

# Household Debt
fig_d = px.line(data,x = 'Date',y = 'D_y',
                color_discrete_sequence=[px.colors.qualitative.D3[0]])
fig_d.update_layout(title='',
                   xaxis_title=None,
                   yaxis_title='Fraction of Quarterly GDP',
                   legend=dict(orientation="h",
                               yanchor="bottom", y=1.02, 
                               xanchor="right", x=1), 
                   legend_title=None, plot_bgcolor='whitesmoke', 
                   margin=dict(l=15, r=15, t=5, b=5),
                   font=dict(family="Times New Roman",size=20,color="black"))
fig_d.add_hline(y=mean_D_y, line_dash="dot", # add mean of series
                line_color=px.colors.qualitative.D3[1],
                annotation_text=f'{round(mean_D_y,2)}', 
                annotation_position='bottom right',
                annotation_font_size=20,
                annotation_font_color='black')
fig_d.update_traces(line=dict(width=3))
fig_d.show()

# Save figure
path_plot_d = os.path.join(os.getcwd(),'Results', 'calibration_d.svg')
fig_d.write_image(path_plot_d)
