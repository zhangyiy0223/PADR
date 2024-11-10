'''
script to run experiments on newsvendor problem with a nonconvex capacity cost
'''

from utils.experiments import *

DATA_FOLDER_LIST = \
    [
        # add newsvendor dataset here
        './data/nv/MA3k5dense/ydim1/xdim2',
    ]

PROBLEM_DICT = {
    'training_sample_size': [50, 100, 200, 400, 600, 800, 1000],
    'c_b': 8,
    'c_h': 2,
    'x_dim': None, # to be loaded from data
    'ncvx_capacity': np.array([[-1, 0], [-0.6, -8], [-0.4, -15.6]]) # array representation for the concave capacity cost = min(z, 0.6z + 0.8, 0.4z + 15.6) = - np.maximum(ncvx_capacity @ [z, 1].T)
}

EXP_NAME = 'ncvx_basic'
METHODS = ['RKHS-DR', 'PADR', 'simopt', 'LDR', 'wSAA/kNN', 'wSAA/kernel', 'wSAA/cart', 'wSAA/rf',  'wSAA/default', 'PO-PA', 'PO-L']

PARAM_DICT = {}

# SIMOPT
PARAM_DICT['simopt'] = {'simulated_num': 1000}

# RKHS-DR
PARAM_DICT['RKHS-DR'] = {
    'gamma': [0.3, 1, 3, 10, 30, 100, 300],
    'lbd': [1e-8, 1e-4],
    'min_eigen': 1e-3
}

# PADR
PARAM_DICT['PADR'] = {
    'K1K2': [(2,2)], 
    'epsilon': [0,],
    'shrinking_epsilon_ratio': [(i, j) for i in [0, 100] for j in [0.3]],
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
PARAM_DICT['wSAA/kNN'] = {'k': [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500]}
PARAM_DICT['wSAA/kernel'] = {'gamma': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]}
PARAM_DICT['wSAA/cart'] = {'max_depth': [10, 30], 'min_samples_split': [2, 10], 'min_samples_leaf': [1, 10],}
PARAM_DICT['wSAA/rf'] = {'n_estimators': [300]}
PARAM_DICT['wSAA/rf'].update(PARAM_DICT['wSAA/cart'])

# PO-PA
PARAM_DICT['PO-PA'] = {
    'K1K2': [(2, 2), (4, 2), (4, 4)],
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
    
    DATA_ADDR_LIST = []
    for data_folder in DATA_FOLDER_LIST:
        folder_list = os.listdir(data_folder)
        folder_list.sort()
        DATA_ADDR_LIST += [check_folder_syntax(data_folder + '/' + data_name) for data_name in folder_list]

    for idx_data_addr, data_addr in enumerate(DATA_ADDR_LIST):
        for method in METHODS:
            _ = experiment_nv(data_addr, method, PROBLEM_DICT, PARAM_DICT[method], EXP_NAME, 
                            curr_exp_idx=idx_data_addr+1, total_exp_num=len(DATA_ADDR_LIST))