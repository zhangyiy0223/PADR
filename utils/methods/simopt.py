from utils.tools import *
from utils.tools_nv import *
from utils.tools_pp import *
from utils.data_generator import *
import gurobipy as grb

def exp_nv_simopt(val_X, val_Y, test_X, test_Y, data_model_name, problem_dict, param_dict, 
                 output_log_addr, curr_exp_idx, total_exp_num):
    exp_start_time = time.time()
    cb, ch = problem_dict['c_b'], problem_dict['c_h']
    n_val = val_X.shape[0]
    n_sim = param_dict['simulated_num']
    z_arr = np.zeros(n_val)
    dim_y = val_Y.shape[1] if len(val_Y.shape) > 1 else 1
    for idx_n_val in range(n_val):
        sim_Y = true_data_model(
            np.ones((n_sim, 1)) @ val_X[idx_n_val, :].reshape((1, -1)), dim_y,
            model_name=data_model_name, load_model_param=None,)
        z_arr[idx_n_val] = solve_nv_saa(np.ones(n_sim)/n_sim, sim_Y, cb, ch)
        update_progress((idx_n_val+1)/n_val, exp_start_time, method='simopt', curr_exp_no=curr_exp_idx, total_exps=total_exp_num)
    
    val_cost = nv_cost(cb, ch, z_arr, val_Y)

    n_test = test_X.shape[0]
    z_arr = np.zeros(n_test)
    for idx_n_test in range(n_test):
        sim_Y = true_data_model(
            np.ones((n_sim, 1)) @ test_X[idx_n_test, :].reshape((1, -1)), dim_y,
            model_name=data_model_name, load_model_param=None,)
        z_arr[idx_n_test] = solve_nv_saa(np.ones(n_sim)/n_sim, sim_Y, cb, ch)
        update_progress((idx_n_test+1)/n_test, exp_start_time, method='simopt', curr_exp_no=curr_exp_idx, total_exps=total_exp_num)

    test_cost = nv_cost(cb, ch, z_arr, test_Y)

    exp_text = f'\nn_sim={n_sim:d} val_cost={val_cost:.4f} test_cost={test_cost:.4f} time={time.time()-exp_start_time:.4f} param=None'
    write_to_file(output_log_addr, exp_text)

def exp_pp_simopt(val_X, val_Y, test_X, test_Y, data_model_name, data_model_param_dict, problem_dict, param_dict, 
                 output_log_addr, curr_exp_idx, total_exp_num):
    exp_start_time = time.time()
    c, g, b, h = problem_dict['c'], problem_dict['g'], problem_dict['b'], problem_dict['h']
    W = problem_dict['W']

    n_val = val_X.shape[0]
    n_sim = param_dict['simulated_num']
    z_arr = np.zeros((n_val, c.shape[0]))
    for idx_n_val in range(n_val):
        sim_Y = true_data_model(
            np.ones((n_sim, 1)) @ val_X[idx_n_val, :].reshape((1, -1)), val_Y.shape[1],
            model_name=data_model_name, load_model_param=data_model_param_dict,)
        z_arr[idx_n_val, :] = solve_pp_saa(np.ones(n_sim)/n_sim, sim_Y, c, g, b, h, W)
        update_progress((idx_n_val+1)/n_val, exp_start_time, method='simopt', curr_exp_no=curr_exp_idx, total_exps=total_exp_num)
    
    val_cost = pp_cost(c, g, b, h, W, z_arr, val_Y)

    n_test = test_X.shape[0]
    z_arr_test = np.zeros((n_test, c.shape[0]))
    for idx_n_test in range(n_test):
        sim_Y = true_data_model(
            np.ones((n_sim, 1)) @ test_X[idx_n_test, :].reshape((1, -1)), val_Y.shape[1],
            model_name=data_model_name, load_model_param=data_model_param_dict,)
        z_arr_test[idx_n_test, :] = solve_pp_saa(np.ones(n_sim)/n_sim, sim_Y, c, g, b, h, W)
        update_progress((idx_n_test+1)/n_test, exp_start_time, method='simopt', curr_exp_no=curr_exp_idx, total_exps=total_exp_num)

    test_cost = pp_cost(c, g, b, h, W, z_arr_test, test_Y)

    exp_text = f'\nn_sim={n_sim:d} val_cost={val_cost:.4f} test_cost={test_cost:.4f} time={time.time()-exp_start_time:.4f} param=None'
    write_to_file(output_log_addr, exp_text)

def exp_nv2prod_simopt(val_X, val_Y, test_X, test_Y, data_model_name, problem_dict, param_dict, 
                    output_log_addr, curr_exp_idx, total_exp_num):
    exp_start_time = time.time()
    cb1, cb2, ch1, ch2, capacity = problem_dict['c_b_1'], problem_dict['c_b_2'], problem_dict['c_h_1'], problem_dict['c_h_2'], problem_dict['fixed_capacity']
    n_val = val_X.shape[0]
    n_sim = param_dict['simulated_num']
    z_arr = np.zeros((n_val, 2))
    for idx_n_val in range(n_val):
        sim_Y = true_data_model(
            np.ones((n_sim, 1)) @ val_X[idx_n_val, :].reshape((1, -1)), val_Y.shape[1],
            model_name=data_model_name, load_model_param=None,)
        z_arr[idx_n_val, :] = solve_nv2prod_saa(np.ones(n_sim)/n_sim, sim_Y, cb1, cb2, ch1, ch2, capacity)
        update_progress((idx_n_val+1)/n_val, exp_start_time, method='simopt', curr_exp_no=curr_exp_idx, total_exps=total_exp_num)
    
    val_cost, _ = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, z_arr, val_Y)

    n_test = test_X.shape[0]
    z_arr_test = np.zeros((n_test, 2))
    for idx_n_test in range(n_test):
        sim_Y = true_data_model(
            np.ones((n_sim, 1)) @ test_X[idx_n_test, :].reshape((1, -1)), val_Y.shape[1],
            model_name=data_model_name, load_model_param=None,)
        z_arr_test[idx_n_test, :] = solve_nv2prod_saa(np.ones(n_sim)/n_sim, sim_Y, cb1, cb2, ch1, ch2, capacity)
        update_progress((idx_n_test+1)/n_test, exp_start_time, method='simopt', curr_exp_no=curr_exp_idx, total_exps=total_exp_num)
    
    test_cost, _ = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, z_arr_test, test_Y)

    exp_text = f'\nn_sim={n_sim:d} val_cost={val_cost:.4f} test_cost={test_cost:.4f} time={time.time()-exp_start_time:.4f} param=None'
    write_to_file(output_log_addr, exp_text)

def exp_ncvxconstr_nv2prod_simopt(val_X, val_Y, test_X, test_Y, data_model_name, problem_dict, param_dict, 
        output_log_addr, curr_exp_idx, total_exp_num):
    exp_start_time = time.time()
    cb1, cb2, ch1, ch2, capacity = problem_dict['c_b_1'], problem_dict['c_b_2'], problem_dict['c_h_1'], problem_dict['c_h_2'], problem_dict['fixed_capacity']
    ncvx_constraint_arr = problem_dict['ncvx_constraint_arr']
    n_sim = param_dict['simulated_num']

    n_test = test_X.shape[0]
    z_arr_test = np.zeros((n_test, 2))
    for idx_n_test in range(n_test):
        sim_Y = true_data_model(
            np.ones((n_sim, 1)) @ test_X[idx_n_test, :].reshape((1, -1)), val_Y.shape[1],
            model_name=data_model_name, load_model_param=None,)
        
        z1 = np.linspace(0, np.max(sim_Y[:, 0]), 50)
        z2 = np.linspace(0, np.max(sim_Y[:, 1]), 50)

        z1_grid, z2_grid = np.meshgrid(z1, z2)
        z_grid = np.vstack((z1_grid.flatten(), z2_grid.flatten())).T

        capacity_cost = np.sum([capacost_function_oneprod(z_grid[:, idx], ncvx_constraint_arr, average=False) for idx in range(z_grid.shape[1])], axis=0)
        feasible_indices = (capacity_cost <= capacity)
        z_grid = z_grid[feasible_indices, :]
        sim_costs = np.sum([
            np.average(nv_cost([cb1, cb2][idx], [ch1, ch2][idx], z_grid[:, idx].reshape((-1, 1)), sim_Y[:, idx].reshape((1, -1)), average=False), axis=1)
            for idx in range(z_grid.shape[1])], axis=0)
        z_arr_test[idx_n_test, :] = z_grid[np.argmin(sim_costs), :]
        update_progress((idx_n_test+1)/n_test, exp_start_time, method='simopt-test', curr_exp_no=curr_exp_idx, total_exps=total_exp_num)

    test_cost, _ = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, z_arr_test, test_Y, constr_arr=ncvx_constraint_arr)
    exp_text = f'\nn_sim={n_sim:d} test_cost={test_cost:.4f} time={time.time()-exp_start_time:.4f} param=None'
    write_to_file(output_log_addr, exp_text)