from utils.data_generator import *


# generate simulated data used in the newsvendor problem
# 1. data with basic setting & varying feature dimensions
for sparse_or_dense in ['sparse', 'dense']:
    PARAM_DICT = {'xdim': [2, 4, 8, 12, 20, 30, 50]} if sparse_or_dense == 'dense' else {'xdim': [4, 8, 12, 20, 30, 50]} # for p=2, the sparse model is equivalent to the dense model, so we only keep the dense model
    data_generator(tr_size=1000, vl_size=1000, test_size=1000, problem_setting='nv', 
                   data_model_name = f'MA3k5{sparse_or_dense}', 
                   loop_dict=PARAM_DICT, save_data=True)

# 2. data with varying nonlinearity in demand model
for k in [1, 3, 7, 10, 15]:
    PARAM_DICT = {'xdim': [2, 20]}
    data_generator(tr_size=1000, vl_size=1000, test_size=1000, problem_setting='nv', 
                   data_model_name = f'MA3k{k}dense', 
                   loop_dict=PARAM_DICT, save_data=True)


# generate simulated data used in the product placement problem
REP_LOOP = 20

# 1. data with varying feature dimensions
for sparse_or_dense in ['sparse', 'dense']:
    PARAM_DICT = {'xdim': [2, 4, 8, 12, 20, 30, 50], 'ydim': 5} if sparse_or_dense == 'dense' else {'xdim': [4, 8, 12, 20, 30, 50], 'ydim': 5}
    for idx_loop in range(REP_LOOP):
        data_generator(tr_size=1000, vl_size=1000, test_size=1000, problem_setting='pp', 
                    data_model_name = f'SINPAk5{sparse_or_dense}Random',
                    loop_dict=PARAM_DICT, 
                    arc_dim = 10, save_data=True)
        
# 2. data with varying nonlinearity in demand model
for k in [1, 3, 7, 10, 15]:
    PARAM_DICT = {'xdim': [20], 'ydim': 5}
    for idx_loop in range(REP_LOOP):
        data_generator(tr_size=1000, vl_size=1000, test_size=1000, problem_setting='pp', 
                    data_model_name = f'SINPAk{k}denseRandom', 
                    loop_dict=PARAM_DICT, 
                    arc_dim = 10, save_data=True)


# generate simulated data used in the constrained newsvendor problem in Appendix
# 1. data with varying feature dimensions
for sparse_or_dense in ['sparse', 'dense']:
    PARAM_DICT = {'xdim': [2, 4, 8, 12, 20, 30, 50], 'ydim': 2} if sparse_or_dense == 'dense' else {'xdim': [4, 8, 12, 20, 30, 50], 'ydim': 2}
    data_generator(tr_size=1000, vl_size=1000, test_size=1000, problem_setting='nv2prod', 
                   data_model_name = f'linear{sparse_or_dense}', 
                   loop_dict=PARAM_DICT, save_data=True)