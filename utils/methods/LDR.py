from utils.tools import *
from utils.tools_nv import *
from utils.tools_pp import *

# newsvendor probem
def single_loop_nv_LDR(train_X_, train_Y, cb, ch):
    n = train_X_.shape[0]
    m = grb.Model('LDR')
    m.Params.LogToConsole = 0
    param = m.addMVar(train_X_.shape[1], lb=-grb.GRB.INFINITY, name='param')
    vB = m.addMVar(n, lb=0, name='arti-var for back')
    vH = m.addMVar(n, lb=0, name='arti-var for hold')

    m.addConstr(vB >= train_Y - train_X_ @ param)
    m.addConstr(vH >= train_X_ @ param - train_Y)
    m.setObjective((np.ones(n)/n * cb) @ vB + (np.ones(n)/n * ch) @ vH)
    m.optimize()

    return param.X

def exp_nv_LDR(train_X_data, train_Y_data, val_X, val_Y, test_X, test_Y,
              problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num, include_val=False):
    exp_start_time = time.time()

    cb, ch = problem_dict['c_b'], problem_dict['c_h']

    train_X_data_ = np.hstack((train_X_data, np.ones((train_X_data.shape[0], 1))))
    val_X_ = np.hstack((val_X, np.ones((val_X.shape[0], 1)))) 
    test_X_ = np.hstack((test_X, np.ones((test_X.shape[0], 1)))) 

    outerLoopParams_dict, outerNumTotal = load_loop_params(problem_dict, 'training_sample_size')
    for idx_outerLoop in range(outerNumTotal):
        n, = pick_loop_params(idx_outerLoop, outerNumTotal, outerLoopParams_dict)
        assert n > 0
        if n > train_X_data.shape[0]:
            print(f'Warning: n={n} is larger than the training sample size. Set n to the training data size ({train_X_data.shape[0]}).')
            n = train_X_data.shape[0]
        # assert n <= train_X_data.shape[0]
        train_X_, train_Y = train_X_data_[:n], train_Y_data[:n]

        startTime_loop = time.time()
        param = single_loop_nv_LDR(train_X_, train_Y, cb, ch)

        train_cost = nv_cost(cb, ch, train_X_ @ param, train_Y)
        val_cost = nv_cost(cb, ch, val_X_ @ param, val_Y)

        param_final = param 
        if include_val:
            train_X_final_ = np.vstack((train_X_, val_X_))
            train_Y_final = np.vstack((train_Y, val_Y)) if len(train_Y.shape)>1 else np.hstack((train_Y, val_Y))
            param_final = single_loop_nv_LDR(train_X_final_, train_Y_final, cb, ch)

        test_cost = nv_cost(cb, ch, test_X_ @ param_final, test_Y)
        update_progress((idx_outerLoop+1)/outerNumTotal, exp_start_time, 'LDR', curr_exp_idx, total_exp_num)

        exp_text = f'\nn={n:d} val_cost={val_cost:.4f} test_cost={test_cost:.4f} time={time.time() - startTime_loop:.2f} train_cost={train_cost:.4f}'
        write_to_file(output_log_addr, exp_text)

# constrained newsvendor problem
def exp_nv2prod_LDR(train_X_data, train_Y_data, val_X, val_Y, test_X, test_Y, 
              problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num):
    exp_start_time = time.time()

    cb1, cb2, ch1, ch2, capacity = problem_dict['c_b_1'], problem_dict['c_b_2'], problem_dict['c_h_1'], problem_dict['c_h_2'], problem_dict['fixed_capacity']

    train_X_data_ = np.hstack((train_X_data, np.ones((train_X_data.shape[0], 1))))
    val_X_ = np.hstack((val_X, np.ones((val_X.shape[0], 1))))
    test_X_ = np.hstack((test_X, np.ones((test_X.shape[0], 1))))
    outerLoopParams_dict, outerNumTotal = load_loop_params(problem_dict, 'training_sample_size')
    for idx_outerLoop in range(outerNumTotal):
        n, = pick_loop_params(idx_outerLoop, outerNumTotal, outerLoopParams_dict)
        assert n > 0
        if n > train_X_data.shape[0]:
            print(f'Warning: n={n} is larger than the training sample size. Set n to the training data size ({train_X_data.shape[0]}).')
            n = train_X_data.shape[0]
        # assert n <= train_X_data.shape[0]
        train_X_, train_Y = train_X_data_[:n], train_Y_data[:n]

        innerLoopParams_dict, innerNumTotal = load_loop_params(param_dict, *param_dict.keys())
        val_costs, params = [], []
        for idx_innerLoop in range(innerNumTotal):
            startTime_loop = time.time()
            
            penalty_lbd, penalty_gamma = pick_loop_params(idx_innerLoop, innerNumTotal, innerLoopParams_dict)
            
            m = grb.Model('LDR')
            m.Params.LogToConsole = 0
            param = m.addMVar((2, train_X_.shape[1]), lb=-grb.GRB.INFINITY, name='param')
            v = m.addMVar((2, n), lb=0, name='arti-var for back and hold')

            m.addConstr(v[0] / ch1 >= train_X_ @ param[0] - train_Y[:, 0])
            m.addConstr(v[0] / cb1 >= train_Y[:, 0] - train_X_ @ param[0])
            m.addConstr(v[1] / ch2 >= train_X_ @ param[1] - train_Y[:, 1])
            m.addConstr(v[1] / cb2 >= train_Y[:, 1] - train_X_ @ param[1])

            penalty_var = m.addMVar(n, lb=0)
            m.addConstr(penalty_var + capacity - penalty_gamma >= train_X_ @ param[0] + train_X_ @ param[1])
            m.setObjective((np.ones(n)*1/n) @ v[0] + (np.ones(n)*1/n) @ v[1]
                +  penalty_var @ (np.diag(np.ones(n))*(penalty_lbd / n)) @ penalty_var)
            
            m.optimize()
            param = param.X
            train_z = train_X_ @ (param.T)
            val_z = val_X_ @ (param.T)
            train_cost, train_feasfreq = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, train_z, train_Y)
            val_cost, val_feasfreq = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, val_z, val_Y)
            val_costs.append(val_cost)
            params.append(param)
            update_progress((idx_outerLoop*innerNumTotal + idx_innerLoop + 1)/(outerNumTotal*innerNumTotal), exp_start_time, 'RKHS-DR', curr_exp_idx, total_exp_num)

            exp_text = f'\nn={n:d} val_cost={val_cost:.4f} val_feasfreq={val_feasfreq:.4f} time={time.time() - startTime_loop:.2f} train_cost={train_cost:.4f} penalty_gamma_lbd=({penalty_gamma},{penalty_lbd})'
            write_to_file(output_log_addr, exp_text)
        
        min_val_idx = np.argmin(val_costs)
        min_val_cost = val_costs[min_val_idx]
        param_final = params[min_val_idx]
        test_z = test_X_ @ (param_final.T)
        test_cost, test_feasfreq = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, test_z, test_Y)
        exp_text = f'\nOuterLoop{idx_outerLoop} (n={n}) with innerLoop{min_val_idx+1}: val_cost={min_val_cost:.4f} test_cost={test_cost:.4f} test_feasfreq={test_feasfreq:.4f}\n\n'
        write_to_file(output_log_addr, exp_text)

# product placement problem
def exp_pp_LDR(train_X_data, train_Y_data, val_X, val_Y, test_X, test_Y, 
              problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num):
    exp_start_time = time.time()

    c, g, b, h, W = problem_dict['c'], problem_dict['g'], problem_dict['b'], problem_dict['h'], problem_dict['W']
    dim_node, dim_arc = c.shape[0], g.shape[0]
    
    train_X_data_ = np.hstack((train_X_data, np.ones((train_X_data.shape[0], 1))))
    val_X_ = np.hstack((val_X, np.ones((val_X.shape[0], 1)))) 
    test_X_ = np.hstack((test_X, np.ones((test_X.shape[0], 1)))) 

    outerLoopParams_dict, outerNumTotal = load_loop_params(problem_dict, 'training_sample_size')
    for idx_outerLoop in range(outerNumTotal):
        n, = pick_loop_params(idx_outerLoop, outerNumTotal, outerLoopParams_dict)
        assert n > 0
        if n > train_X_data.shape[0]:
            print(f'Warning: n={n} is larger than the training sample size. Set n to the training data size ({train_X_data.shape[0]}).')
            n = train_X_data.shape[0]
        # assert n <= train_X_data.shape[0]
        train_X_, train_Y = train_X_data_[:n], train_Y_data[:n]

        startTime_loop = time.time()
        m = grb.Model('LDR')
        m.Params.LogToConsole = 0
        param = m.addMVar((train_X_.shape[1], dim_node), lb=-grb.GRB.INFINITY, name='param')
        f = m.addMVar((dim_arc, n), lb=0, name='f')
        aux_bmax0 = m.addMVar((dim_node, n), lb=0, name='aux_bmax0')
        aux_ldr_1 = m.addMVar((dim_node, n), lb=-grb.GRB.INFINITY, name='aux_ldr_1')
        aux_ldr_2 = m.addMVar((dim_node, n), lb=-grb.GRB.INFINITY, name='aux_ldr_2')
        aux_obj = m.addMVar((n,1), lb=-grb.GRB.INFINITY, name='aux_obj')

        for idx_node in range(dim_node):
            m.addConstr(aux_ldr_1[idx_node, :] >= train_X_ @ param[:, idx_node])
            m.addConstr(aux_ldr_2[idx_node, :] >= -train_X_ @ param[:, idx_node])
        
        for i in range(n):
            m.addConstr(aux_bmax0[:, i] >= train_Y[i, :] + W @ f[:, i] + aux_ldr_2[:, i])
            m.addConstr(aux_obj[i] >= c @ aux_ldr_1[:, i] + g @ f[:, i] + b @ aux_bmax0[:, i])
        
        m.setObjective((np.ones(n)/n) @ aux_obj[:, 0], grb.GRB.MINIMIZE)
        m.optimize()

        param = param.X
        train_z = train_X_ @ param
        val_z = val_X_ @ param
        train_cost = pp_cost(c, g, b, h, W, train_z, train_Y)
        val_cost = pp_cost(c, g, b, h, W, val_z, val_Y)
        test_z = test_X_ @ param
        test_cost = pp_cost(c, g, b, h, W, test_z, test_Y)

        update_progress((idx_outerLoop+1)/outerNumTotal, exp_start_time, 'LDR', curr_exp_idx, total_exp_num)

        exp_text = f'\nn={n:d} val_cost={val_cost:.4f} test_cost={test_cost:.4f} time={time.time() - startTime_loop:.2f} train_cost={train_cost:.4f}'
        write_to_file(output_log_addr, exp_text)