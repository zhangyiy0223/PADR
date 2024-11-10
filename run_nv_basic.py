'''
script to run experiments on newsvendor problem with basic settings
'''

from utils.experiments import *

# add data folders here
# format: ./data/{problem_name}/{data_name}/{problem_dim}/{feature_dim}
# the methods will load ALL data folders (format: 'data_{timestamp}_{uid}') under the directory, which can be seen after running `python generate_data.py`
DATA_FOLDER_LIST = \
    [
        './data/nv/MA3k5dense/ydim1/xdim2',
    ]

# problem parameters (already set)
PROBLEM_DICT = {
    'training_sample_size': [50, 100, 200, 400, 600, 800, 1000], # the methods will run experiments on the dataset with different training sample sizes
    'c_b': 8, # nv cost params
    'c_h': 2,
    'x_dim': None, # to be loaded from data
    'ncvx_capacity': None # None for standard newsvendor
}

EXP_NAME = 'basic' # experiment name is used to distinguish log files under the same output folder
METHODS = ['PADR', 'RKHS-DR', 'simopt', 'LDR', 'wSAA/kNN', 'wSAA/kernel', 'wSAA/cart', 'wSAA/rf', 'wSAA/sof', 'PO-PA', 'PO-L'] # methods to run, including benchmarks and PADR

PARAM_DICT = {}

# SIMOPT
PARAM_DICT['simopt'] = {'simulated_num': 1000} # number of the simulated scenarios on each data point

'''
For the following methods, the parameters are set as a dictionary.
The list of hyperparams will be used to generate all possible combinations to search the best one.
For some methods, hyperparams are already searched and a fine-tuned range is provided.
'''

# PADR
PARAM_DICT['PADR'] = {
    'K1K2': [(3,0)], # piece number tuple: (K1, K2)
    # ESMM params
    'epsilon': [0,], # use enhanced technique if epsilon > 0
    'shrinking_epsilon_ratio': [(i, j) for i in [0, 10, 100, 300, 1000, 3000] for j in [0.3]], # shrinking epsilon strategy for ESMM (a, b): use epsilon=a in the first b*iteration rounds
    'round': [10,], # number of rounds
    'iteration_alpha_beta_N0': [(10, 40, 1, 0)], # iteration = 10, sampling rate = alpha*t^beta + N0 
    'mu': [50], # scale of theta
    'eta': [0,],
    'sampling': True # sequential sampling strategy: False for EMM and standard MM
}

# RKHS-DR
PARAM_DICT['RKHS-DR'] = {
    'gamma': [1, 3, 10, 30, 100, 300], # kernel param
    'lbd': [1e-8, 1e-4, 1e-2, 1e-1], # regularization param
    'min_eigen': 1e-3
}

# LDR (no params)
PARAM_DICT['LDR'] = {}

# params for prescriptive methods, denoted by w(eighted)SAA
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
    'method': ['apx-soln', 'apx-risk',]} # 'oracle' costs too much time, so we do not include it here

# PO-PA (with PA model and ESMM params)
PARAM_DICT['PO-PA'] = {
    'K1K2': [(3,0)],
    # ESMM params
    'epsilon': [0,],
    'shrinking_epsilon_ratio': [(i, j) for i in [0, 10, 100, 300, 1000, 3000] for j in [0.3]],
    'round': [10,],
    'iteration_alpha_beta_N0': [(10, 40, 1, 0)],
    'mu': [50],
    'eta': [0,],
    'sampling': True
}

# PO-L (no params)
PARAM_DICT['PO-L'] = {}



if __name__ == '__main__':
    
    # for each data folder, list all subfolders
    DATA_ADDR_LIST = []
    for data_folder in DATA_FOLDER_LIST:
        folder_list = os.listdir(data_folder)
        folder_list.sort()
        DATA_ADDR_LIST += [check_folder_syntax(data_folder + '/' + data_name) for data_name in folder_list]
    
    for idx_data_addr, data_addr in enumerate(DATA_ADDR_LIST):
        for method in METHODS:
            _ = experiment_nv(data_addr, method, PROBLEM_DICT, PARAM_DICT[method], EXP_NAME, 
                            curr_exp_idx=idx_data_addr+1, total_exp_num=len(DATA_ADDR_LIST))