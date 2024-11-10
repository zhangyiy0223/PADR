'''
script to run experiments on product placement problem with varying settings
'''
from utils.experiments import *

DATA_FOLDER_LIST = [    
    # PART I: Varying feature dimensions
    './data/pp/SINPAk5denseRandom/ydim5arcdim10/xdim2',
    './data/pp/SINPAk5denseRandom/ydim5arcdim10/xdim4',
    './data/pp/SINPAk5denseRandom/ydim5arcdim10/xdim8',
    './data/pp/SINPAk5denseRandom/ydim5arcdim10/xdim12',
    './data/pp/SINPAk5denseRandom/ydim5arcdim10/xdim20',
    './data/pp/SINPAk5denseRandom/ydim5arcdim10/xdim30',
    './data/pp/SINPAk5denseRandom/ydim5arcdim10/xdim50',


    './data/pp/SINPAk5sparseRandom/ydim5arcdim10/xdim50',
    './data/pp/SINPAk5sparseRandom/ydim5arcdim10/xdim30',
    './data/pp/SINPAk5sparseRandom/ydim5arcdim10/xdim20',
    './data/pp/SINPAk5sparseRandom/ydim5arcdim10/xdim12',
    './data/pp/SINPAk5sparseRandom/ydim5arcdim10/xdim8',
    './data/pp/SINPAk5sparseRandom/ydim5arcdim10/xdim4',

    # PART II: Nonlinearity in demand model
    './data/pp/SINPAk1denseRandom/ydim5arcdim10/xdim20',
    './data/pp/SINPAk3denseRandom/ydim5arcdim10/xdim20',
    './data/pp/SINPAk7denseRandom/ydim5arcdim10/xdim20',
    './data/pp/SINPAk5denseRandom/ydim5arcdim10/xdim20',
    './data/pp/SINPAk10denseRandom/ydim5arcdim10/xdim20',
    './data/pp/SINPAk15denseRandom/ydim5arcdim10/xdim20',
]

PROBLEM_DICT = {
    'training_sample_size': [1000],
    'c': 3,
    'g': 1,
    'b': 5,
    'h': 0,
    'W': None, # node-arc matrix, to be loaded from data
    'x_dim': None, # to be loaded from data
}

EXP_NAME = 'varying_settings'
METHODS = ['LDR', 'PADR', 'RKHS-DR', 'wSAA/rf', 'wSAA/kNN', 'wSAA/kernel', 'wSAA/cart', 'PO-PA', 'PO-L', 'simopt']

PARAM_DICT = {}
PARAM_DICT['simopt'] = {'simulated_num': 500}

PARAM_DICT['wSAA/default'] = {'default': None}
PARAM_DICT['wSAA/kNN'] = {'k': [1, 3, 10, 30, 100, 300, 1000]}
PARAM_DICT['wSAA/kernel'] = {'gamma': [1, 3, 10, 30, 100, 300]}
PARAM_DICT['wSAA/cart'] = {
    'max_depth': [10, 30], 
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1],}
PARAM_DICT['wSAA/rf'] = {'n_estimators': [300]}
PARAM_DICT['wSAA/rf'].update(PARAM_DICT['wSAA/cart'])

PARAM_DICT['PADR'] = {
    'K1K2': [(2,2), (4, 2), (4, 4)],
    'epsilon': [0,],
    'shrinking_epsilon_ratio': [(i, j) for i in [0] for j in [0.3]],
    'round': [5,],
    'iteration_alpha_beta_N0': [(20, 40, 1, 0)],
    'mu': [100],
    'eta': [0,],
    'sampling': True
}

PARAM_DICT['PO-PA'] = {
    'K1K2': [(2, 2), (4, 2), (4, 4)],
    'epsilon': [0,],
    'shrinking_epsilon_ratio': [(i, j) for i in [0] for j in [0.3]],
    'round': [5,],
    'iteration_alpha_beta_N0': [(20, 40, 1, 0)],
    'mu': [100],
    'eta': [0,],
    'sampling': True
}

PARAM_DICT['PO-L'] = {}

PARAM_DICT['RKHS-DR'] = {
    'gamma': [1, 3, 10, 30, 100, 300],
    'lbd': [1e-6],
    'min_eigen': 1e-3,
}

PARAM_DICT['LDR'] = {}




if __name__ == '__main__':
    
    # for each folder, list all subfolders
    DATA_ADDR_LIST = []
    for data_folder in DATA_FOLDER_LIST:
        folder_list = os.listdir(data_folder)
        folder_list.sort()
        DATA_ADDR_LIST += [check_folder_syntax(data_folder + '/' + data_name) for data_name in folder_list[:5]]

    for idx_data_addr, data_addr in enumerate(DATA_ADDR_LIST):
        for method in METHODS:
            _ = experiment_pp(data_addr, method, PROBLEM_DICT, PARAM_DICT[method], EXP_NAME, 
                            curr_exp_idx=idx_data_addr+1, total_exp_num=len(DATA_ADDR_LIST))