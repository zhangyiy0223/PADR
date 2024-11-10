'''
script to run experiments on 2-product constrained newsvendor problem with basic settings
'''
from utils.experiments import *


DATA_FOLDER_LIST = \
    [
        './data/nv2prod/lineardense/ydim2/xdim2',
    ]

PROBLEM_DICT = {
    'training_sample_size': [50, 100, 200, 400, 600, 800, 1000],
    'c_b_1': 8,
    'c_b_2': 2,
    'c_h_1': 2,
    'c_h_2': 8,
    'fixed_capacity': 60, # C_0
    'x_dim': None, # to be loaded from data
}

EXP_NAME = 'basic'
METHODS = ['PADR', 'RKHS-DR', 'wSAA/kNN', 'LDR', 'wSAA/kernel', 'wSAA/cart', 'wSAA/rf', 'wSAA/sof', 'simopt',]

PARAM_DICT = {}

# simopt
PARAM_DICT['simopt'] = {'simulated_num': 1000}

# RKHS
PARAM_DICT['RKHS-DR'] = {
    'gamma': [1, 3, 10, 30, 100, 300],
    'lbd': [1e-8, 1e-4, 1e-2, 1e-1],
    'min_eigen': 1e-3,
    'penalty-lbd': [100, 300],
    'penalty-gamma': [0, ],
}

# PADR
PARAM_DICT['PADR'] = {
    'K1K2': [(2, 2)],
    'epsilon': [0,],
    'shrinking_epsilon_ratio': [(i, j) for i in [0, 10] for j in [0.3]],
    'round': [10,],
    'iteration_alpha_beta_N0': [(10, 40, 1, 0)],
    'mu': [50],
    'eta': [0,],
    'sampling': True,
    # params for exact penalization
    'penalty-lbd': [100, 300],
    'penalty-gamma': [0, 0.05, 0.1],
}

# LDR
PARAM_DICT['LDR'] = {
    'penalty-lbd': [100, 300],
    'penalty-gamma': [0, 0.05, 0.1],
}

# wSAA
PARAM_DICT['wSAA/default'] = {'default': None}
PARAM_DICT['wSAA/kNN'] = {'k': [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]}
PARAM_DICT['wSAA/kernel'] = {'gamma': [0.1, 1, 10, 100], }
PARAM_DICT['wSAA/cart'] = {
    'max_depth': [30],
    'min_samples_split': [2, 10], 
    'min_samples_leaf': [1, 10],}
PARAM_DICT['wSAA/rf'] = {'n_estimators': [300]}
PARAM_DICT['wSAA/rf'].update(PARAM_DICT['wSAA/cart'])
PARAM_DICT['wSAA/sof'] = {
    'max_depth': [30], 
    'min_samples_leaf': [1, 10], 
    'n_estimators': [300], 
    'method': ['apx-soln', 'apx-risk']}



if __name__ == '__main__':
    
    # for each folder, list all subfolders
    DATA_ADDR_LIST = []
    for data_folder in DATA_FOLDER_LIST:
        folder_list = os.listdir(data_folder)
        folder_list.sort()
        DATA_ADDR_LIST += [check_folder_syntax(data_folder + '/' + data_name) for data_name in folder_list]
    
    for idx_data_addr, data_addr in enumerate(DATA_ADDR_LIST):
        for method in METHODS:
            _ = experiment_nv2prod(data_addr, method, PROBLEM_DICT, PARAM_DICT[method], EXP_NAME, 
                            curr_exp_idx=idx_data_addr+1, total_exp_num=len(DATA_ADDR_LIST))