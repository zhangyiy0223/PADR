from utils.tools import *
from utils.methods.wSAA import *
from utils.methods.simopt import *
from utils.methods.LDR import *
from utils.methods.PA import *
from utils.methods.RKHS import *

def experiment_nv(data_addr, method, problem_dict, param_dict, exp_name='', curr_exp_idx=1, total_exp_num=1):

    # Load Data
    # data_addr: './data/problem_name/data_name/problem_dim/x_dim/datastamp/'
    _, _, problem_name, data_name, problem_dim_str, x_dim_str, data_stamp, _ = data_addr.split('/')

    data_addr = check_folder_syntax(data_addr)

    if data_name != 'realworlddata':
        data_npz = np.load(data_addr + 'data.npz')
        train_X, train_Y, val_X, val_Y, test_X, test_Y = data_npz['train_X'], data_npz['train_Y'], data_npz['val_X'], data_npz['val_Y'], data_npz['test_X'], data_npz['test_Y']
        data_info = read_from_file(data_addr + 'data_info.log')
    else:
        X = np.load(data_addr + 'X.npy')
        Y = np.load(data_addr + 'Y.npy')
        data_length = X.shape[0]
        train_length = int(data_length * 0.4)
        val_length = int(data_length * 0.1)
        train_X, train_Y = X[:train_length], Y[:train_length]
        val_X, val_Y = X[train_length:train_length+val_length], Y[train_length:train_length+val_length]
        test_X, test_Y = X[train_length+val_length:], Y[train_length+val_length:]
        data_info = 'realworld dataset dim_y=1'

    problem_dict['x_dim'] = train_X.shape[1]

    exp_log_name = f'exp_{exp_name}_{unique_stamp()}' if exp_name is not None else f'exp_{unique_stamp()}'
    output_addr = check_folder_syntax(f'./output/{problem_name}/{method}/{data_name}/{problem_dim_str}/{x_dim_str}/{data_stamp}')
    output_log_addr = output_addr + f'{exp_log_name}.log'

    if not os.path.exists(output_addr):
        os.makedirs(output_addr)
    save_ExpSettings(output_log_addr, data_addr, data_info, problem_dict, method, param_dict)

    data_model_name = data_name.split('_')[0]
    include_val = True if data_model_name == 'realworlddata' else False # before testing, train on all data including val if using real world data

    # Experiment
    if method=='simopt':
        if data_model_name == 'realworlddata':
            text = "Real world data is not supported for simopt."
            write_to_file(output_log_addr, text)
            return output_addr
        exp_nv_simopt(val_X, val_Y, test_X, test_Y, data_model_name, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num)
    elif method == 'RKHS-DR':
        exp_nv_RKHS(train_X, train_Y, val_X, val_Y, test_X, test_Y, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num, include_val=include_val)
    elif method=='LDR':
        exp_nv_LDR(train_X, train_Y, val_X, val_Y, test_X, test_Y, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num, include_val=include_val)
    elif method == 'PADR':
        exp_nv_PADR(train_X, train_Y, val_X, val_Y, test_X, test_Y, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num, include_val=include_val)
    elif method=='PO-PA':
        exp_nv_PO(train_X, train_Y, val_X, val_Y, test_X, test_Y, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num, include_val=include_val)
    elif method=='PO-L':
        param_dict.update({
            'K1K2': [(1,0)],
            'epsilon': [0,],
            'shrinking_epsilon_ratio': [(0, 0)],
            'round': [1,],
            'iteration_alpha_beta_N0': [(1, 0, 0, 0)],
            'mu': [100],
            'eta': [0,],
            'sampling': False})
        exp_nv_PO(train_X, train_Y, val_X, val_Y, test_X, test_Y, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num, name='PO-L', include_val=include_val)
    elif method.split('/')[0]=='wSAA':
        param_dict['weighting_method'] = method.split('/')[1]
        exp_nv_wSAA(train_X, train_Y, val_X, val_Y, test_X, test_Y, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num, include_val=include_val)
    else:
        raise ValueError(f'Invalid method [{method}].')
    
    print("Complete!")
    return output_addr

def experiment_pp(data_addr, method, problem_dict, param_dict, exp_name='', curr_exp_idx=1, total_exp_num=1):
    data_addr = check_folder_syntax(data_addr)
    data_npz = np.load(data_addr + 'data.npz')
    train_X, train_Y, val_X, val_Y, test_X, test_Y = data_npz['train_X'], data_npz['train_Y'], data_npz['val_X'], data_npz['val_Y'], data_npz['test_X'], data_npz['test_Y']
    W = data_npz['W']
    data_info = read_from_file(data_addr + 'data_info.log')

    problem_dict['W'] = W
    problem_dict['x_dim'] = train_X.shape[1]
    problem_dict['c'] = np.ones(W.shape[0]) * problem_dict['c']
    problem_dict['g'] = np.ones(W.shape[1]) * problem_dict['g']
    problem_dict['b'] = np.ones(W.shape[0]) * problem_dict['b']
    problem_dict['h'] = np.ones(W.shape[0]) * problem_dict['h']

    # './data/problem_name/data_name/problem_dim/x_dim/datastamp/'
    _, _, problem_name, data_name, problem_dim_str, x_dim_str, data_stamp, _ = data_addr.split('/')
    exp_log_name = f'exp_{exp_name}_{unique_stamp()}' if exp_name is not None else f'exp_{unique_stamp()}'
    output_addr = check_folder_syntax(f'./output/{problem_name}/{method}/{data_name}/{problem_dim_str}/{x_dim_str}/{data_stamp}')
    output_log_addr = output_addr + f'{exp_log_name}.log'

    if not os.path.exists(output_addr):
        os.makedirs(output_addr)
    save_ExpSettings(output_log_addr, data_addr, data_info, problem_dict, method, param_dict)

    if method=='simopt':
        data_model_name = data_name.split('_')[0]
        data_model_param_dict = np.load(data_addr + 'data_model.npz')
        exp_pp_simopt(val_X, val_Y, test_X, test_Y, data_model_name, data_model_param_dict, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num)
    elif method=='RKHS-DR':
        exp_pp_RKHS(train_X, train_Y, val_X, val_Y, test_X, test_Y, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num)
    elif method=='LDR':
        exp_pp_LDR(train_X, train_Y, val_X, val_Y, test_X, test_Y, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num)
    elif method=='PADR':
        exp_pp_PADR(train_X, train_Y, val_X, val_Y, test_X, test_Y, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num)
    elif method=='PO-PA':
        exp_pp_PO(train_X, train_Y, val_X, val_Y, test_X, test_Y, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num)
    elif method=='PO-L':
        # implement using PO-PA with K_1 = 1, K_2 = 0
        param_dict.update({
            'K1K2': [(1,0)],
            'epsilon': [0,],
            'shrinking_epsilon_ratio': [(0, 0)],
            'round': [1,],
            'iteration_alpha_beta_N0': [(1, 0, 0, 0)],
            'mu': [100],
            'eta': [0,],
            'sampling': False})
        exp_pp_PO(train_X, train_Y, val_X, val_Y, test_X, test_Y, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num, name='PO-L')
    elif method.split('/')[0]=='wSAA':
        param_dict['weighting_method'] = method.split('/')[1]
        exp_pp_wSAA(train_X, train_Y, val_X, val_Y, test_X, test_Y, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num)
    else:
        raise ValueError(f'Invalid method [{method}].')
    
    print("Complete!")
    return output_addr

def experiment_nv2prod(data_addr, method, problem_dict, param_dict, exp_name='', curr_exp_idx=1, total_exp_num=1):
    # './data/problem_name/data_name/problem_dim/x_dim/datastamp/'
    _, _, problem_name, data_name, problem_dim_str, x_dim_str, data_stamp, _ = data_addr.split('/')

    data_addr = check_folder_syntax(data_addr)

    data_npz = np.load(data_addr + 'data.npz')
    train_X, train_Y, val_X, val_Y, test_X, test_Y = data_npz['train_X'], data_npz['train_Y'], data_npz['val_X'], data_npz['val_Y'], data_npz['test_X'], data_npz['test_Y']
    data_info = read_from_file(data_addr + 'data_info.log')

    problem_dict['x_dim'] = train_X.shape[1]
    
    exp_log_name = f'exp_{exp_name}_{unique_stamp()}' if exp_name is not None else f'exp_{unique_stamp()}'
    output_addr = check_folder_syntax(f'./output/{problem_name}/{method}/{data_name}/{problem_dim_str}/{x_dim_str}/{data_stamp}')
    output_log_addr = output_addr + f'{exp_log_name}.log'

    if not os.path.exists(output_addr):
        os.makedirs(output_addr)
    save_ExpSettings(output_log_addr, data_addr, data_info, problem_dict, method, param_dict)

    if method=='simopt':
        data_model_name = data_name.split('_')[0]
        exp_nv2prod_simopt(val_X, val_Y, test_X, test_Y, data_model_name, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num)
    elif method=='RKHS-DR':
        exp_nv2prod_RKHS(train_X, train_Y, val_X, val_Y, test_X, test_Y, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num)
    elif method=='LDR':
        exp_nv2prod_LDR(train_X, train_Y, val_X, val_Y, test_X, test_Y, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num)
    elif method=='PADR':
        exp_nv2prod_PADR(train_X, train_Y, val_X, val_Y, test_X, test_Y, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num)
    elif method.split('/')[0]=='wSAA':
        param_dict['weighting_method'] = method.split('/')[1]
        exp_nv2prod_wSAA(train_X, train_Y, val_X, val_Y, test_X, test_Y, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num)
    else:
        raise ValueError(f'Invalid method [{method}].')
    
    print("Complete!")
    return output_addr

def experiment_ncvxconstr_nv2prod(data_addr, method, problem_dict, param_dict, exp_name='', curr_exp_idx=1, total_exp_num=1):
    _, _, problem_name, data_name, problem_dim_str, x_dim_str, data_stamp, _ = data_addr.split('/')
    data_addr = check_folder_syntax(data_addr)

    data_npz = np.load(data_addr + 'data.npz')
    train_X, train_Y, val_X, val_Y, test_X, test_Y = data_npz['train_X'], data_npz['train_Y'], data_npz['val_X'], data_npz['val_Y'], data_npz['test_X'], data_npz['test_Y']
    data_info = read_from_file(data_addr + 'data_info.log')

    problem_dict['x_dim'] = train_X.shape[1]

    exp_log_name = f'exp_{exp_name}_{unique_stamp()}' if exp_name is not None else f'exp_{unique_stamp()}'
    output_addr = check_folder_syntax(f'./output/{problem_name}/{method}/{data_name}/{problem_dim_str}/{x_dim_str}/{data_stamp}')
    output_log_addr = output_addr + f'{exp_log_name}.log'

    if not os.path.exists(output_addr):
        os.makedirs(output_addr)
    save_ExpSettings(output_log_addr, data_addr, data_info, problem_dict, method, param_dict)

    if method=='simopt':
        data_model_name = data_name.split('_')[0]
        exp_ncvxconstr_nv2prod_simopt(val_X, val_Y, test_X, test_Y, data_model_name, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num)
    elif method=='RKHS-DR':
        exp_ncvxconstr_nv2prod_RKHS(train_X, train_Y, val_X, val_Y, test_X, test_Y, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num)
    elif method=='PADR':
        exp_ncvxconstr_nv2prod_PADR(train_X, train_Y, val_X, val_Y, test_X, test_Y, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num)
    elif method.split('/')[0]=='wSAA':
        param_dict['weighting_method'] = method.split('/')[1]
        exp_ncvxconstr_nv2prod_wSAA(train_X, train_Y, val_X, val_Y, test_X, test_Y, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num)
    else:
        raise ValueError(f'Invalid method [{method}].')
    
    print("Complete!")
    return output_addr