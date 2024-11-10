'''
script to run experiments on newsvendor problem with varying feature dimensions and nonlinearity levels of the demand model;
the script is similar to run_nv_basic.py, but with different data folders;
'''

from utils.experiments import *


DATA_FOLDER_LIST = \
    [
        # dense model
        './data/nv/MA3k5dense/ydim1/xdim2',
        './data/nv/MA3k5dense/ydim1/xdim4',
        './data/nv/MA3k5dense/ydim1/xdim8',
        './data/nv/MA3k5dense/ydim1/xdim12',
        './data/nv/MA3k5dense/ydim1/xdim20',
        './data/nv/MA3k5dense/ydim1/xdim30',
        './data/nv/MA3k5dense/ydim1/xdim50',

        # sparse model
        './data/nv/MA3k5sparse/ydim1/xdim4',
        './data/nv/MA3k5sparse/ydim1/xdim8',
        './data/nv/MA3k5sparse/ydim1/xdim12',
        './data/nv/MA3k5sparse/ydim1/xdim20',
        './data/nv/MA3k5sparse/ydim1/xdim30',
        './data/nv/MA3k5sparse/ydim1/xdim50',

        # varying nonlinearity (dense model, p=2)
        './data/nv/MA3k1dense/ydim1/xdim2',
        './data/nv/MA3k3dense/ydim1/xdim2',
        './data/nv/MA3k7dense/ydim1/xdim2',
        './data/nv/MA3k10dense/ydim1/xdim2',
        './data/nv/MA3k15dense/ydim1/xdim2',

        # varying nonlinearity (dense model, p=20)
        './data/nv/MA3k1dense/ydim1/xdim20',
        './data/nv/MA3k3dense/ydim1/xdim20',
        './data/nv/MA3k7dense/ydim1/xdim20',
        './data/nv/MA3k10dense/ydim1/xdim20',
        './data/nv/MA3k15dense/ydim1/xdim20',
    ]

PROBLEM_DICT = {
    'training_sample_size': [1000],
    'c_b': 8,
    'c_h': 2,
    'x_dim': None, # to be loaded from data
    'ncvx_capacity': None
}

EXP_NAME = 'varying_settings'
METHODS = ['RKHS-DR', 'PADR', 'simopt', 'LDR', 'wSAA/kNN', 'wSAA/kernel', 'wSAA/cart', 'wSAA/rf', 'wSAA/sof', 'PO-PA', 'PO-L']


PARAM_DICT = {}

# SIMOPT
PARAM_DICT['simopt'] = {'simulated_num': 1000}

# RKHS-DR
PARAM_DICT['RKHS-DR'] = {
    'gamma': [1, 3, 10, 30, 100, 300],
    'lbd': [1e-8, 1e-4, 1e-2, 1e-1],
    'min_eigen': 1e-3
}

# PADR
PARAM_DICT['PADR'] = {
    'K1K2': [(2,2), (3,0), (3,3)],
    'epsilon': [0,],
    'shrinking_epsilon_ratio': [(i, j) for i in [0, 10, 100, 300, 1000, 3000] for j in [0.3]],
    'round': [10,],
    'iteration_alpha_beta_N0': [(10, 40, 1, 0)],
    'mu': [50],
    'eta': [0,],
    'sampling': True
}

# LDR
PARAM_DICT['LDR'] = {}

# wSAA
PARAM_DICT['wSAA/default'] = {'default': None}
PARAM_DICT['wSAA/kNN'] = {'k': [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]}
PARAM_DICT['wSAA/kernel'] = {'gamma': [1, 3, 10, 30, 100, 300],}
PARAM_DICT['wSAA/cart'] = {'max_depth': [10, 30], 'min_samples_split': [2, 10], 'min_samples_leaf': [1],}
PARAM_DICT['wSAA/rf'] = {'n_estimators': [300]}
PARAM_DICT['wSAA/rf'].update(PARAM_DICT['wSAA/cart'])
PARAM_DICT['wSAA/sof'] = {
    'max_depth': [10, 30],
    'min_samples_leaf': [1],
    'n_estimators': [300], 
    'method': ['apx-soln', 'apx-risk',]} # 'oracle' is time-consuming and not recommended

# PO-PA
PARAM_DICT['PO-PA'] = {
    'K1K2': [(2,2), (3,0), (3,3)],
    'epsilon': [0,],
    'shrinking_epsilon_ratio': [(i, j) for i in [0, 10, 100, 300, 1000, 3000] for j in [0.3]],
    'round': [10,],
    'iteration_alpha_beta_N0': [(10, 40, 1, 0)],
    'mu': [50],
    'eta': [0,],
    'sampling': True
}

# PO-L
PARAM_DICT['PO-L'] = {}



if __name__ == '__main__':
    
    # for each folder, list all subfolders
    DATA_ADDR_LIST = []
    for data_folder in DATA_FOLDER_LIST:
        folder_list = os.listdir(data_folder)
        folder_list.sort()
        DATA_ADDR_LIST += [check_folder_syntax(data_folder + '/' + data_name) for data_name in folder_list]
    
    for idx_data_addr, data_addr in enumerate(DATA_ADDR_LIST):
        for method in METHODS:
            _ = experiment_nv(data_addr, method, PROBLEM_DICT, PARAM_DICT[method], EXP_NAME, 
                            curr_exp_idx=idx_data_addr+1, total_exp_num=len(DATA_ADDR_LIST))