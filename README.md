# Master Thesis: Financial Constraints and Household Heterogeneity in the Macroeconomy

This repository accompanies my master thesis, titled *Financial Constraints and Household Heterogeneity in the Macroeconomy* and supervised by Prof. Dr. Keith Kuester and Dr. Gregor Böhl. This thesis was presented to the Department of Economics at the Rheinische Friedrich-Wilhelms-Universität Bonn in partial fulfillment of the requirements for the degree of Master of Science (M.Sc.) in Economics. In particular, you can find here the thesis itself and the codes to generate the results as well as the figures found therein.

## `main.py`

This is the main file of the project. The code herein produces the core results of my thesis. Some of the results are used by other code files, such as `compare_transitions.py`. The code here allows the user to select one or more of the possible combinations of models and shocks. Having set the respective choices, the code loads, adjusts and solves the initial and terminal models, computes the fully non-linear perfect-foresight transition paths and prints various informative plots and tables about the steady states and the transitions. If desired, these results are stored in the folder [**Results**](https://github.com/andkound98/master-thesis/tree/main/Results). The transitions can also be stored as pickle files in the end, for easy re-use by `compare_transitions.py`.

## `compare_transitions.py`

This file contains code which visually compares the transitions produced by different models and shocks. In particular, the user selects the combination of instances to be compared. To compare transitions, these must have been implemented and stored first via the `main.py file`.

## `beta_shock.py`

This file implements the permanent shock to the household discount factor in the baseline model, as described in appendix E.3 of the thesis.

## `calibration.py`

This file calculates targets for the calibration of the models in the thesis and plots some of the time series. 

## `custom_functions.py`

This file contains custom functions used throughout the project.

## `plot_functions.py`

This file contains custom functions for plotting various results throughout the project.

## Models 

The folder [**Models**](https://github.com/andkound98/master-thesis/tree/main/Models) contains the models employed by the various code files:
1. `hank_baseline.yml`: baseline model
2. `hank_slow_shock.yml`: model with high persistence in shock processes
3. `hank_fast_shock.yml`: model with low persistence in shock processes
4. `hank_end_L.yml`: extended model with CRRA preferences and endogenous labour supply
5. `hank_very_slow_phi.yml`: model with very high persistence in borrowing limit
6. `hank_no_ZLB.yml`: model without zero-lower bound
7. `hank_low_B.yml`: model with low calibration of liquid assets
8. `hank_baseline_beta.yml`: baseline model with shock to household discount factor

All of these are to the largest part based on the models from the [`Econpizza`](https://github.com/gboehl/econpizza/tree/master) package, which is also used for the model solution. Finally, the [**Models**](https://github.com/andkound98/master-thesis/tree/main/Models) folder also contains `hank_functions.py`, which contains functions necessary for the solution of the models and similarly derives almost entirely from an example file in [`Econpizza`](https://github.com/gboehl/econpizza/tree/master).

## Results

The folder [**Results**](https://github.com/andkound98/master-thesis/tree/main/Results) contains the results produced by the `main.py` file and the `custom_functions.py` file.

## Data

The folder [**Data**](https://github.com/andkound98/master-thesis/tree/main/Data) contains a CSV file with data on US household balance sheets, obtained from the Federal Reserve website on the [Flow of Funds](https://www.federalreserve.gov/releases/z1/) and used by `calibration.py`.

## Thesis 

The folder [**Thesis**](https://github.com/andkound98/master-thesis/tree/main/Thesis) contains the TeX code of the thesis, its `.bib` file and its PDF version.

---

All these codes were run using the Spyder IDE 5.4.3 with Python 3.9.12 and [`Econpizza`](https://github.com/gboehl/econpizza/tree/master) 0.6.1 on macOS 12.6.5.
