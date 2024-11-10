This repo contains code for the paper [Data-driven Piecewise Affine Decision Rules for Stochastic Programming with Covariate Information](https://arxiv.org/abs/2304.13646)

# Code structure
Simulated data is generated using the script `generate_data.py`.
All scripts for different experiments are `run_*.py`.
General utility functions are in `./utils/tools.py`; tool functions for newsvendor and product placement problems are in `./utils/tools_nv.py` and `./utils/tools_pp.py`, respectively. Experiment functions are in `./utils/experiments.py`. The data generation function and true demand models are in `./utils/data_generator.py`.
All methods are in `./utils/methods`. The code for "stochastic optimization forests" ("sof" method in the scripts and the paper) is in `./utils/methods/StochOptForest_nv_tree_utilities.py` and `./utils/methods/StochOptForest_tree.py` which are from https://github.com/CausalML/StochOptForest.

# How to run
## data generation
To generate simulated data, use the following command:
```bash
python generate_data.py
```
Generated data will be saved in the address `./data/{problem_name}/{demand_model}/{problem_dim}/{feature_dim}/{data_uid}/data.npz`.
For confidentiality purposes, the real-world data is not provided.

## scripts
To run the experiments, use the following command:
```bash
python run_*.py
```
| Script | Problem |
| --- | --- |
| `run_nv_basic.py` | Newsvendor Problem: Basic Results |
| `run_nv_varyingsettings.py` | Newsvendor Problem with Varying Feature Dimensions and Varying Nonlinearity in Demand Model |
| `run_ncvx_nv.py` | Nonconvex Newsvendor Problem |
| `run_pp_varyingsettings.py` | Product Placement Problem with Varying Feature Dimensions and Varying Nonlinearity in Demand Model |
| `run_nv2prod_basic.py` | Constrained Newsvendor Problem: Basic Results |
| `run_nv2prod_varyingsettings.py` | Constrained Newsvendor Problem with Varying Feature Dimensions and Varying Nonlinearity in Demand Model |
| `run_ncvxconstr_nv2prod_basic.py` | Nonconvex Constrained Problem: Basic Results |
| `run_ncvxconstr_nv2prod_varyingsettings.py` | Nonconvex Constrained Problem with Varying Feature Dimensions in Demand Model |

More instructions on setting hyperparameters for each method can be seen in `run_nv_basic.py`.

## results
The results will be saved in the dir `./output/{problem_name}/{method_name}/{demand_model}/{problem_dim}/{x_dim}/{data_name}/exp_{time_stamp}.log`.

# Dependencies
## python 3.9.7
- gurobipy 9.5.0
- numpy 1.22.4
- scikit-learn 0.24.2