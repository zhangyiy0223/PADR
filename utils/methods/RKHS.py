from utils.tools import *
from utils.tools_nv import *
from utils.tools_pp import *
from utils.methods.PA import get_epsilon_nu, A_eps_ma, IJ, preprocessing_combinations, compute_trainobj_2prod

def GaussianKernel(x1, x2, gamma):
    """
    x1 (n1, n_features) and x2 (n2, n_features) should have same feature dimensions!
    return Arr with shape (n1, n2) and Arr[i, j] is RBFkernel element i-th row vec of x1 with j-th row vec of x2
    """
    if gamma <= 0:
        raise ValueError("the parameter gamma <= 0 is illegal!")

    x1 = x1.reshape((-1, x1.shape[-1])) # x1 (n1, dim_feature)
    x2 = x2.reshape((-1, x2.shape[-1])) # x2 (n2, dim_feature)

    # shape (n1, n2, dim_feature): Delta[i, j] = x1[i] - x2[j] (i-th row vec of x1 - j-th row vec of x2)
    x1_minus_x2 = x1.reshape(x1.shape[0], 1, x1.shape[1]) - x2.reshape(1, x2.shape[0], x2.shape[1])
    L2square_x1mx2 = np.sum(x1_minus_x2**2, axis=-1) # shape (n1, n2)
    return np.exp(-L2square_x1mx2 / gamma)
    
def RKHS_RBF_DR(X, Xbase, param, gamma):
    """
    param: (Xbase.shape[0],) or (Xbase.shape[0], dim_node)
    """
    return GaussianKernel(X, Xbase, gamma) @ param

def exp_pp_RKHS(train_X_data, train_Y_data, val_X, val_Y, test_X, test_Y, 
              problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num):
    exp_start_time = time.time()

    c, g, b, h, W = problem_dict['c'], problem_dict['g'], problem_dict['b'], problem_dict['h'], problem_dict['W']
    dim_node, dim_arc = c.shape[0], g.shape[0]

    outerLoopParams_dict, outerNumTotal = load_loop_params(problem_dict, 'training_sample_size')
    for idx_outerLoop in range(outerNumTotal):
        n, = pick_loop_params(idx_outerLoop, outerNumTotal, outerLoopParams_dict)
        assert n > 0
        if n > train_X_data.shape[0]:
            print(f'Warning: n={n} is larger than the training sample size. Set n to the training data size ({train_X_data.shape[0]}).')
            n = train_X_data.shape[0]
        # assert n <= train_X_data.shape[0]
        train_X, train_Y = train_X_data[:n], train_Y_data[:n]

        innerLoopParams_dict, innerNumTotal = load_loop_params(param_dict, *param_dict.keys())
        val_costs, params = [], []
        for idx_innerLoop in range(innerNumTotal):
            startTime_loop = time.time()
            gamma, lbd, min_eigen = pick_loop_params(idx_innerLoop, innerNumTotal, innerLoopParams_dict)
            Kernel_Mtx = GaussianKernel(train_X, train_X, gamma) + np.diag(np.ones(n) * min_eigen)

            m = grb.Model('RKHS-DR')
            m.Params.LogToConsole = 0  # do not output the log info
            param = m.addMVar((n, dim_node), lb=-grb.GRB.INFINITY, name='param')
            f = m.addMVar((dim_arc, n), lb=0, name='f')
            aux_bmax0 = m.addMVar((dim_node, n), lb=0, name='aux_bmax0')
            aux_rkhs_1 = m.addMVar((dim_node, n), lb=-grb.GRB.INFINITY, name='aux_rkhs_1')
            aux_rkhs_2 = m.addMVar((dim_node, n), lb=-grb.GRB.INFINITY, name='aux_rkhs_2')
            aux_obj = m.addMVar((n,1), lb=-grb.GRB.INFINITY, name='aux_obj')

            for idx_node in range(dim_node):
                m.addConstr(aux_rkhs_1[idx_node, :] >= Kernel_Mtx @ param[:, idx_node])
                m.addConstr(aux_rkhs_2[idx_node, :] >= -Kernel_Mtx @ param[:, idx_node])
            
            for i in range(n):
                m.addConstr(aux_bmax0[:, i] >= train_Y[i, :] + W @ f[:, i] + aux_rkhs_2[:, i])
                m.addConstr(aux_obj[i] >= c @ aux_rkhs_1[:, i] + g @ f[:, i] + b @ aux_bmax0[:, i])
            
            objective_quadratic = 0
            for i in range(dim_node):
                objective_quadratic += (param[:, i] @ (lbd * Kernel_Mtx) @ param[:, i])
            objective = (np.ones(n) * 1/n) @ aux_obj[:, 0] + objective_quadratic
            m.setObjective(objective, grb.GRB.MINIMIZE)
            m.optimize()
            
            try:
                param = param.X
            except:
                param = -1 * np.ones((n, dim_node))
            train_z = RKHS_RBF_DR(train_X, train_X, param, gamma)
            val_z = RKHS_RBF_DR(val_X, train_X, param, gamma)
            train_cost = pp_cost(c, g, b, h, W, train_z, train_Y)
            val_cost = pp_cost(c, g, b, h, W, val_z, val_Y)
            val_costs.append(val_cost)
            params.append(param)

            update_progress(
                (idx_outerLoop*innerNumTotal + idx_innerLoop + 1)/(outerNumTotal*innerNumTotal),
                exp_start_time, 'RKHS-DR', curr_exp_idx, total_exp_num)

            exp_text = f'\nn={n:d} param_idx={idx_innerLoop+1:d}/{innerNumTotal:d} val_cost={val_cost:.4f} time={time.time() - startTime_loop:.2f} train_cost={train_cost:.4f} gamma_lbd_min_eigen=({gamma},{lbd},{min_eigen})'
            write_to_file(output_log_addr, exp_text)

        min_val_idx = np.argmin(val_costs)
        min_val_cost = val_costs[min_val_idx]
        param_final = params[min_val_idx]
        gamma, lbd, min_eigen = pick_loop_params(min_val_idx, innerNumTotal, innerLoopParams_dict)
        test_z = RKHS_RBF_DR(test_X, train_X, param_final, gamma)
        test_cost = pp_cost(c, g, b, h, W, test_z, test_Y)
        exp_text = f'\nOuterLoop{idx_outerLoop} (n={n}) with innerLoop{min_val_idx+1}: val_cost={min_val_cost:.4f} test_cost={test_cost:.4f}\n\n'
        write_to_file(output_log_addr, exp_text)


def solve_mm_erm_nv_rkhs(train_X, train_Y, gamma, lbd, cb, ch, min_eigen, capacity_arr, output_log_addr):
    iteration = 10
    # eps, shrink_eps, shrink_quant = 0, 0, 0
    n = train_X.shape[0]

    Kernel_Mtx = GaussianKernel(train_X, train_X, gamma) + np.diag(np.ones(n) * min_eigen)

    param_init = np.zeros((n,))
    ## algorithm for one round (start)
    param_nu = param_init
    param_output = param_nu
    train_z = RKHS_RBF_DR(train_X, train_X, param_output, gamma)
    min_cost = nv_cost(cb, ch, train_z, train_Y) + capacost_function_oneprod(train_z, capacity_arr) # initial cost
    
    for nu in range(iteration):
        idcs_n_nu_uniq_freq = np.ones(n) / n
        train_X, train_Y = train_X.copy(), train_Y.copy()

        z_nu = RKHS_RBF_DR(train_X, train_X, param_nu, gamma)
        z_nu_ = np.hstack((z_nu.reshape((-1, 1)), np.ones((z_nu.shape[0], 1))))
        CJ_nu = A_eps_ma(capacity_arr, z_nu_, 0)
        CJcomb_num_for_sample = preprocessing_combinations(CJ_nu)
        CJ_nu = IJ(CJ_nu, CJcomb_num_for_sample)
        
        m = grb.Model('prob_nu')
        m.Params.LogToConsole = 0  # do not output the log info
        param_tmp = m.addMVar(n, lb=-grb.GRB.INFINITY, name='param')
        v = m.addMVar(n,)
        m.addConstr(v / ch >= Kernel_Mtx @ param_tmp - train_Y)
        m.addConstr(v / cb >= train_Y - Kernel_Mtx @ param_tmp)
        # if add_capacost:
        costparam_pos = capacity_arr[CJ_nu, :]
        kernel_mtx_remake = Kernel_Mtx * np.tile(-costparam_pos[:, 0].reshape((-1,1)), Kernel_Mtx.shape[1])
        # m.addConstr(vcapacost >= kernel_mtx_remake @ param - costparam_pos[:, 1])

        m.setObjective((idcs_n_nu_uniq_freq) @ v
                    + param_tmp @ (lbd * Kernel_Mtx) @ param_tmp
                    + idcs_n_nu_uniq_freq @ kernel_mtx_remake @ param_tmp - idcs_n_nu_uniq_freq @ costparam_pos[:, 1])
        m.optimize()

        try:
            param_nu = param_tmp.X
            cost_obj_nu = m.ObjVal
        except:
            param_nu = -1 * np.ones(n)
            cost_obj_nu = 1e10
        
        train_z_curr = RKHS_RBF_DR(train_X, train_X, param_nu, gamma)
        cost_curr = nv_cost(cb, ch, train_z_curr, train_Y) + capacost_function_oneprod(train_z_curr, capacity_arr)
        
        exp_text = f'\n - Iter.{nu+1}: n={n} train_cost={cost_curr:.4f} obj={cost_obj_nu:.4f} min_train_cost={min_cost:.4f}'
        write_to_file(output_log_addr, exp_text)
        
        if cost_curr < min_cost - 1e-3:
            min_cost = cost_curr
            param_output = param_nu
        else:
            break
    return param_output

def single_loop_nv_RKHS(gamma, lbd, min_eigen, train_X, train_Y, cb, ch, capacity_arr, output_log_addr):
    n = train_X.shape[0]
    if capacity_arr is None:
        Kernel_Mtx = GaussianKernel(train_X, train_X, gamma) + np.diag(np.ones(n) * min_eigen)
        m = grb.Model('RKHS-DR')
        m.Params.LogToConsole = 0  # do not output the log info
        param = m.addMVar(n, lb=-grb.GRB.INFINITY, name='param')
        v = m.addMVar(n,)
        m.addConstr(v / ch >= Kernel_Mtx @ param - train_Y)
        m.addConstr(v / cb >= train_Y - Kernel_Mtx @ param)
        m.setObjective((np.ones(n)*1/n) @ v + param @ (lbd * Kernel_Mtx) @ param)
        m.optimize()
        try:
            param = param.X
        except:
            param = -1 * np.ones(n)
    else:
        param = solve_mm_erm_nv_rkhs(train_X, train_Y, gamma, lbd, cb, ch, min_eigen, capacity_arr, output_log_addr)
    return param

def exp_nv_RKHS(train_X_data, train_Y_data, val_X, val_Y, test_X, test_Y,
              problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num, include_val=False):
    exp_start_time = time.time()

    cb, ch, capacity_arr = problem_dict['c_b'], problem_dict['c_h'], problem_dict['ncvx_capacity']
    
    outerLoopParams_dict, outerNumTotal = load_loop_params(problem_dict, 'training_sample_size')
    for idx_outerLoop in range(outerNumTotal):
        n, = pick_loop_params(idx_outerLoop, outerNumTotal, outerLoopParams_dict)
        assert n > 0
        if n > train_X_data.shape[0]:
            print(f'Warning: n={n} is larger than the training sample size. Set n to the training data size ({train_X_data.shape[0]}).')
            n = train_X_data.shape[0]
        # assert n <= train_X_data.shape[0]
        train_X, train_Y = train_X_data[:n], train_Y_data[:n]

        innerLoopParams_dict, innerNumTotal = load_loop_params(param_dict, *param_dict.keys())
        val_costs, params = [], []
        for idx_innerLoop in range(innerNumTotal):
            startTime_loop = time.time()
            gamma, lbd, min_eigen = pick_loop_params(idx_innerLoop, innerNumTotal, innerLoopParams_dict)
            param = single_loop_nv_RKHS(gamma, lbd, min_eigen, train_X, train_Y, cb, ch, capacity_arr, output_log_addr)

            train_z = RKHS_RBF_DR(train_X, train_X, param, gamma)
            val_z = RKHS_RBF_DR(val_X, train_X, param, gamma)
            train_cost = nv_cost(cb, ch, train_z, train_Y) + capacost_function_oneprod(train_z, capacity_arr)
            val_cost = nv_cost(cb, ch, val_z, val_Y) + capacost_function_oneprod(val_z, capacity_arr)
            val_costs.append(val_cost)
            params.append(param)
            
            update_progress((idx_outerLoop*innerNumTotal + idx_innerLoop + 1)/(outerNumTotal*innerNumTotal), exp_start_time, 'RKHS-DR', curr_exp_idx, total_exp_num)

            exp_text = f'\nn={n:d} param_idx={idx_innerLoop+1:d}/{innerNumTotal:d} val_cost={val_cost:.4f} time={time.time() - startTime_loop:.2f} train_cost={train_cost:.4f} gamma_lbd_min_eigen=({gamma},{lbd},{min_eigen})'
            write_to_file(output_log_addr, exp_text)
        
        min_val_idx = np.argmin(val_costs)
        min_val_cost = val_costs[min_val_idx]
        gamma_final, lbd_final, min_eigen_final = pick_loop_params(min_val_idx, innerNumTotal, innerLoopParams_dict)
        if not include_val:
            param_final = params[min_val_idx]
            train_X_final, train_Y_final = train_X, train_Y
        else:
            train_X_final = np.vstack((train_X, val_X))
            train_Y_final = np.hstack((train_Y, val_Y)) if len(train_Y.shape) == 1 else np.vstack((train_Y, val_Y))
            param_final = single_loop_nv_RKHS(gamma_final, lbd_final, min_eigen_final, train_X_final, train_Y_final, cb, ch, capacity_arr, output_log_addr)
        test_z = RKHS_RBF_DR(test_X, train_X_final, param_final, gamma_final)
        test_cost = nv_cost(cb, ch, test_z, test_Y) + capacost_function_oneprod(test_z, capacity_arr)

        exp_text = f'\nOuterLoop{idx_outerLoop} (n={n}) with innerLoop{min_val_idx+1}: val_cost={min_val_cost:.4f} test_cost={test_cost:.4f}\n\n'
        write_to_file(output_log_addr, exp_text)


def exp_nv2prod_RKHS(train_X_data, train_Y_data, val_X, val_Y, test_X, test_Y, 
              problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num):
    exp_start_time = time.time()

    cb1, cb2, ch1, ch2, capacity = problem_dict['c_b_1'], problem_dict['c_b_2'], problem_dict['c_h_1'], problem_dict['c_h_2'], problem_dict['fixed_capacity']

    outerLoopParams_dict, outerNumTotal = load_loop_params(problem_dict, 'training_sample_size')
    for idx_outerLoop in range(outerNumTotal):
        n, = pick_loop_params(idx_outerLoop, outerNumTotal, outerLoopParams_dict)
        assert n > 0
        if n > train_X_data.shape[0]:
            print(f'Warning: n={n} is larger than the training sample size. Set n to the training data size ({train_X_data.shape[0]}).')
            n = train_X_data.shape[0]
        # assert n <= train_X_data.shape[0]
        train_X, train_Y = train_X_data[:n], train_Y_data[:n]

        innerLoopParams_dict, innerNumTotal = load_loop_params(param_dict, *param_dict.keys())
        val_costs, params = [], []
        for idx_innerLoop in range(innerNumTotal):
            startTime_loop = time.time()

            gamma, lbd, min_eigen, penalty_lbd, penalty_gamma = pick_loop_params(idx_innerLoop, innerNumTotal, innerLoopParams_dict)
            Kernel_Mtx = GaussianKernel(train_X, train_X, gamma) + np.diag(np.ones(n) * min_eigen)

            m = grb.Model('RKHS-DR')
            m.Params.LogToConsole = 0
            param = m.addMVar((2, n), lb=-grb.GRB.INFINITY, name='param')
            v = m.addMVar((2, n), lb=0)

            m.addConstr(v[0] / ch1 >= Kernel_Mtx @ param[0] - train_Y[:, 0])
            m.addConstr(v[0] / cb1 >= train_Y[:, 0] - Kernel_Mtx @ param[0])
            m.addConstr(v[1] / ch2 >= Kernel_Mtx @ param[1] - train_Y[:, 1])
            m.addConstr(v[1] / cb2 >= train_Y[:, 1] - Kernel_Mtx @ param[1])

            penalty_var = m.addMVar(n, lb=0)
            m.addConstr(penalty_var + capacity - penalty_gamma >= Kernel_Mtx @ param[0] + Kernel_Mtx @ param[1])
            m.setObjective((np.ones(n)*1/n) @ v[0] + (np.ones(n)*1/n) @ v[1]
                +  penalty_var @ (np.diag(np.ones(n))*(penalty_lbd / n)) @ penalty_var
                + param[0] @ (lbd * Kernel_Mtx) @ param[0] + param[1] @ (lbd * Kernel_Mtx) @ param[1])
            
            m.optimize()
            end_time = time.time()
            try:
                param = param.X
            except:
                param = -1 * np.ones((2, n))

            train_z = np.hstack((RKHS_RBF_DR(train_X, train_X, param[0], gamma).reshape((-1, 1)), RKHS_RBF_DR(train_X, train_X, param[1], gamma).reshape((-1, 1))))
            val_z = np.hstack((RKHS_RBF_DR(val_X, train_X, param[0], gamma).reshape((-1, 1)), RKHS_RBF_DR(val_X, train_X, param[1], gamma).reshape((-1, 1))))
            train_cost, train_feasfreq = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, train_z, train_Y)
            val_cost, val_feasfreq = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, val_z, val_Y)
            val_costs.append(val_cost)
            params.append(param)
            update_progress((idx_outerLoop*innerNumTotal + idx_innerLoop + 1)/(outerNumTotal*innerNumTotal), exp_start_time, 'RKHS-DR', curr_exp_idx, total_exp_num)

            exp_text = f'\nn={n:d} param_idx={idx_innerLoop+1:d}/{innerNumTotal:d} val_cost={val_cost:.4f} val_feasfreq={val_feasfreq:.4f} time={end_time - startTime_loop:.2f} train_cost={train_cost:.4f} gamma_lbd_min_eigen=({gamma},{lbd},{min_eigen}) penalty_gamma_lbd=({penalty_gamma},{penalty_lbd})'
            write_to_file(output_log_addr, exp_text)

        min_val_idx = np.argmin(val_costs)
        min_val_cost = val_costs[min_val_idx]
        param_final = params[min_val_idx]
        gamma, lbd, min_eigen, penalty_lbd, penalty_gamma = pick_loop_params(min_val_idx, innerNumTotal, innerLoopParams_dict)
        test_z = np.hstack((RKHS_RBF_DR(test_X, train_X, param_final[0], gamma).reshape((-1, 1)), RKHS_RBF_DR(test_X, train_X, param_final[1], gamma).reshape((-1, 1))))
        test_cost, test_feasfreq = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, test_z, test_Y)
        exp_text = f'\nOuterLoop{idx_outerLoop} (n={n}) with innerLoop{min_val_idx+1}: val_cost={min_val_cost:.4f} val_feasfreq={val_feasfreq:.4f} test_cost={test_cost:.4f} test_feasfreq={test_feasfreq:.4f}\n\n'
        write_to_file(output_log_addr, exp_text)

def exp_ncvxconstr_nv2prod_RKHS(train_X_data, train_Y_data, val_X, val_Y, test_X, test_Y, 
              problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num):
    exp_start_time = time.time()
    cb1, cb2, ch1, ch2, capacity = problem_dict['c_b_1'], problem_dict['c_b_2'], problem_dict['c_h_1'], problem_dict['c_h_2'], problem_dict['fixed_capacity']
    ncvx_constraint_arr = problem_dict['ncvx_constraint_arr']

    outerLoopParams_dict, outerNumTotal = load_loop_params(problem_dict, 'training_sample_size')
    for idx_outerLoop in range(outerNumTotal):
        n, = pick_loop_params(idx_outerLoop, outerNumTotal, outerLoopParams_dict)
        assert n > 0
        if n > train_X_data.shape[0]:
            print(f'Warning: n={n} is larger than the training sample size. Set n to the training data size ({train_X_data.shape[0]}).')
            n = train_X_data.shape[0]
        # assert n <= train_X_data.shape[0]
        train_X, train_Y = train_X_data[:n], train_Y_data[:n]

        innerLoopParams_dict, innerNumTotal = load_loop_params(param_dict, *param_dict.keys())
        val_costs, feas_freqs, params = np.zeros(innerNumTotal), np.zeros(innerNumTotal), []
        for idx_innerLoop in range(innerNumTotal):
            startTime_loop = time.time()
            
            gamma, lbd, min_eigen, penalty_lbd, penalty_gamma = pick_loop_params(idx_innerLoop, innerNumTotal, innerLoopParams_dict)
            Kernel_Mtx = GaussianKernel(train_X, train_X, gamma) + np.diag(np.ones(n) * min_eigen)
            
            iteration = 10
            eps, shrink_eps, shrink_quant = 0, 0, 0
            startTime_loop = time.time()

            param1_init, param2_init = np.zeros((n,)), np.zeros((n,))

            param1_nu, param2_nu = param1_init, param2_init
            param1_output, param2_output = param1_nu, param2_nu

            train_z_output = np.hstack((RKHS_RBF_DR(train_X, train_X, param1_output, gamma).reshape((-1, 1)), 
                                 RKHS_RBF_DR(train_X, train_X, param2_output, gamma).reshape((-1, 1))))

            min_train_cost, _ = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, train_z_output, train_Y, constr_arr=ncvx_constraint_arr)
            min_train_obj = compute_trainobj_2prod(train_z_output, train_Y, [cb1, cb2], [ch1, ch2], penalty_lbd, penalty_gamma, capacity, constr_arr=ncvx_constraint_arr, rkhs={
                    'lbd': lbd, 'kernel': Kernel_Mtx, 'param1': param1_nu, 'param2': param2_nu
                })

            for nu in range(iteration):
                N_nu = n
                idcs_Nnu = np.arange(0, N_nu, 1)

                idcs_Nnu_uniq, idcs_Nnu_uniq_freq = np.unique(idcs_Nnu, return_counts=True) # calculate frequency and unique the sample index (_uniq is sorted)
                idcs_Nnu_uniq_freq = idcs_Nnu_uniq_freq / N_nu # turn counts into freq
                N_nu_uniq = idcs_Nnu_uniq.shape[0] # uniqued sample size
                X_nu, Y1_nu, Y2_nu = train_X[idcs_Nnu_uniq, :], train_Y[idcs_Nnu_uniq, 0], train_Y[idcs_Nnu_uniq, 1]

                # derive optimization model
                order1_nu, order2_nu = RKHS_RBF_DR(X_nu, train_X, param1_nu, gamma), RKHS_RBF_DR(X_nu, train_X, param2_nu, gamma)
                order1_nu_ = np.hstack((order1_nu.reshape((-1, 1)), np.ones((order1_nu.shape[0], 1))))
                order2_nu_ = np.hstack((order2_nu.reshape((-1, 1)), np.ones((order2_nu.shape[0], 1))))
                CJ1_nu, CJ2_nu = A_eps_ma(ncvx_constraint_arr, order1_nu_, 0), A_eps_ma(ncvx_constraint_arr, order2_nu_, 0)
                CJ1comb_num_for_sample = preprocessing_combinations(CJ1_nu)
                CJ2comb_num_for_sample = preprocessing_combinations(CJ2_nu)
                CJ1_nu, CJ2_nu = IJ(CJ1_nu, CJ1comb_num_for_sample), IJ(CJ2_nu, CJ2comb_num_for_sample)

                m = grb.Model('RKHS-MM')
                m.Params.LogToConsole = 0  # do not output the log info
                param1 = m.addMVar(n, lb=-grb.GRB.INFINITY, name='param1')
                param2 = m.addMVar(n, lb=-grb.GRB.INFINITY, name='param2')
                v1 = m.addMVar(N_nu_uniq, lb=0)
                v2 = m.addMVar(N_nu_uniq, lb=0)

                kernel_mtx = Kernel_Mtx[idcs_Nnu_uniq, :]
                m.addConstr(v1 / ch1 >= kernel_mtx @ param1 - Y1_nu)
                m.addConstr(v1 / cb1 >= Y1_nu - kernel_mtx @ param1)
                m.addConstr(v2 / ch2 >= kernel_mtx @ param2 - Y2_nu)
                m.addConstr(v2 / cb2 >= Y2_nu - kernel_mtx @ param2)

                if penalty_lbd==0:
                    m.setObjective(idcs_Nnu_uniq_freq @ v1 + idcs_Nnu_uniq_freq @ v2 
                            + param1 @ (lbd * Kernel_Mtx) @ param1 + param2 @ (lbd * Kernel_Mtx) @ param2)
                else:
                    penv = m.addMVar(N_nu_uniq, lb=0)
                    costparam_pos1, costparam_pos2 = ncvx_constraint_arr[CJ1_nu, :], ncvx_constraint_arr[CJ2_nu, :]
                    kernel_remake1 = kernel_mtx * np.tile(-costparam_pos1[:, 0].reshape((-1,1)), kernel_mtx.shape[1])
                    kernel_remake2 = kernel_mtx * np.tile(-costparam_pos2[:, 0].reshape((-1,1)), kernel_mtx.shape[1])
                    m.addConstr(penv + capacity - penalty_gamma >= 
                        kernel_remake1 @ param1 + kernel_remake2 @ param2 - costparam_pos1[:, 1] - costparam_pos2[:, 1])
                    m.setObjective(idcs_Nnu_uniq_freq @ v1 + idcs_Nnu_uniq_freq @ v2 
                        + penv @ (np.diag(idcs_Nnu_uniq_freq)*penalty_lbd) @ penv
                        + param1 @ (lbd * Kernel_Mtx) @ param1 + param2 @ (lbd * Kernel_Mtx) @ param2)
                    
                m.optimize()
                # # # # # #
                end_time = time.time()
                try:
                    param1_nu, param2_nu = param1.X, param2.X
                except:
                    param1_nu, param2_nu = -1 * np.ones(n), -1 * np.ones(n)

                order1_nu, order2_nu = RKHS_RBF_DR(X_nu, train_X, param1_nu, gamma), RKHS_RBF_DR(X_nu, train_X, param2_nu, gamma)
                train_z_nu = np.hstack((order1_nu.reshape((-1, 1)), order2_nu.reshape((-1, 1))))
                train_cost_nu, _ = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, train_z_nu, train_Y, constr_arr=ncvx_constraint_arr)
                train_obj_nu = compute_trainobj_2prod(train_z_nu, train_Y, [cb1, cb2], [ch1, ch2], penalty_lbd, penalty_gamma, capacity, constr_arr=ncvx_constraint_arr, rkhs={
                    'lbd': lbd, 'kernel': Kernel_Mtx, 'param1': param1_nu, 'param2': param2_nu
                })
                if train_obj_nu < min_train_obj - 1e-3:
                    min_train_obj = train_obj_nu
                    min_train_cost, _ = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, train_z_nu, train_Y, constr_arr=ncvx_constraint_arr)
                    param1_output, param2_output = param1_nu, param2_nu
                else:
                    # convergence
                    break
            
            if (param1_output is not None) and (param2_output is not None):
                train_z_output = np.hstack((RKHS_RBF_DR(train_X, train_X, param1_output, gamma).reshape((-1, 1)), RKHS_RBF_DR(train_X, train_X, param2_output, gamma).reshape((-1, 1))))
                val_z = np.hstack((RKHS_RBF_DR(val_X, train_X, param1_output, gamma).reshape((-1, 1)), RKHS_RBF_DR(val_X, train_X, param2_output, gamma).reshape((-1, 1))))

                train_cost, tr_fsbl_freq = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, train_z_output, train_Y, constr_arr=ncvx_constraint_arr)
                val_cost, val_fsbl_freq = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, val_z, val_Y, constr_arr=ncvx_constraint_arr)
            else:
                val_cost, train_cost, val_fsbl_freq, tr_fsbl_freq = 1e10, 1e10, 0, 0

            val_costs[idx_innerLoop] = val_cost
            feas_freqs[idx_innerLoop] = val_fsbl_freq
            params.append((param1_output, param2_output))

            update_progress((idx_outerLoop*innerNumTotal + idx_innerLoop + 1)/(outerNumTotal*innerNumTotal), exp_start_time, 'RKHS-DR', curr_exp_idx, total_exp_num)
            
            exp_text = f'\nn={n:d} param_idx={idx_innerLoop+1:d}/{innerNumTotal:d} val_cost={val_cost:.4f} val_feasfreq={val_fsbl_freq:.4f} time={end_time - startTime_loop:.2f} train_cost={train_cost:.4f} train_feasfreq={tr_fsbl_freq:.4f} gamma_lbd_min_eigen=({gamma},{lbd},{min_eigen}) penalty_gamma_lbd=({penalty_gamma},{penalty_lbd})'
            write_to_file(output_log_addr, exp_text)
            
        
        # min_val_idx = np.argmin(val_costs)
        if np.max(feas_freqs) >= 1:
            # val_costs = np.array(val_costs)
            val_costs[feas_freqs < 1] = 1e10
            min_val_idx = np.argmin(val_costs)
        else:
            min_val_idx = np.argmax(feas_freqs)
        gamma, lbd, min_eigen, penalty_lbd, penalty_gamma = pick_loop_params(min_val_idx, innerNumTotal, innerLoopParams_dict)
        min_val_cost = val_costs[min_val_idx]
        param1_final, param2_final = params[min_val_idx]
        test_z = np.hstack((RKHS_RBF_DR(test_X, train_X, param1_final, gamma).reshape((-1, 1)), RKHS_RBF_DR(test_X, train_X, param2_final, gamma).reshape((-1, 1))))
        
        test_cost, test_feasfreq = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, test_z, test_Y, constr_arr=ncvx_constraint_arr)
        exp_text = f'\nOuterLoop{idx_outerLoop} (n={n}) with innerLoop{min_val_idx+1}: val_cost={min_val_cost:.4f} test_cost={test_cost:.4f} test_feasfreq={test_feasfreq:.4f}\n\n'
        write_to_file(output_log_addr, exp_text)