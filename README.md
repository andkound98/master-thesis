# Master Thesis: Financial Constraints and Household Heterogeneity in the Macroeconomy

This repository accompanies my master thesis, titled *Financial Constraints and Household Heterogeneity in the Macroeconomy* and supervised by Prof. Dr. Keith Kuester and Dr. Gregor Böhl. This thesis was presented to the Department of Economics at the Rheinische Friedrich-Wilhelms-Universität Bonn in partial fulfillment of the requirements for the degree of Master of Science (M.Sc.) in Economics. In particular, you can find here the thesis itself and the codes to generate the results as well as the figures found therein.

## `main.py`

This is the main file of the project. The code herein reproduces the results found in the thesis. It allows the user to select one or more of the possible combinations of models and shocks. Having set the respective choices, the code loads, adjusts and solves the initial and terminal models, computes the fully non-linear perfect-foresight transition paths and plots various informative plots about the steady states and transitions. If desired, the results are stored in the folder 'Results'.

## `compare_transitions.py`

This file creates plots that compare the transitions produced by different models and shocks. To compare the transitions, they must have been implemented first through the main file and stored in a folder in 'Results'.

## `custom_functions.py`

This file contains custom functions used throughout the project, inter alia in the `main.py` file and the `plot_functions.py` file.

## `plot_functions.py`

This file contains custom functions for plotting various results from the HANK models, e.g. policies, distributions and transitions.

## `calibration.py`

This file calculates targets for the calibration of the employed HANK model.

## Models 

The folder [**Models**](https://github.com/andkound98/master-thesis/tree/main/Models) contains the models employed by the various code files.

## Results

The folder [**Results**](https://github.com/andkound98/master-thesis/tree/main/Results) contains the results produced by the `main.py` file and the `plot_functions.py` file.

## Thesis 

The folder [**Thesis**](https://github.com/andkound98/master-thesis/tree/main/Thesis) contains the TeX code of the thesis as well as its `.bib` file. The PDF version of the thesis is also in this folder.

---
All these codes were run using the Spyder IDE 5.4.3 with Python 3.9.12 and [`Econpizza`](https://github.com/gboehl/econpizza/tree/master) 0.6.1 on macOS 12.6.5.
