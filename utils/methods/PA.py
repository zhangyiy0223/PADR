from utils.tools import *
from utils.tools_nv import *
from utils.tools_pp import *

# basic piecewise functions
def f_ma(theta, x):
    """
    input
    - - -
    theta: (dim_node, K, dim_x+1)
    x: (n, dim_x+1) (with interception)

    output
    - - -
    max-affine result, max(x theta^T): (n, dim_node)
    """
    return np.max(theta @ x.T, axis=-2).T
    # return np.max(x.dot(theta.T), axis=1)

def f_ma_1node(theta, x):
    return np.max(theta @ x.T, axis=-2)

def f_pa(samples_w, theta1, theta2):
    """
    Piecewise affine function
    """
    if theta1 is None:
        return - f_ma(theta2, samples_w)
    if theta2 is None:
        return f_ma(theta1, samples_w)
    return f_ma(theta1, samples_w) - f_ma(theta2, samples_w)

def f_ma_slice(theta, x, I):
    """
    input
    - - -
    theta: (dim_node, K, dim_x+1) the dim must be 2 even if K=1
    x: (n, dim_x+1) (with interception)
    I: (dim_node, n, ), the indices for each sample s \in [N]

    output
    - - -
    max-affine slice: (n, dim_node)
    """
    # theta_selected = theta[I] # (n, p+1)
    theta_selected = np.array([theta[i][I[i]] for i in range(theta.shape[0])]) # (dim_node, n, p+1)
    return np.sum(x.reshape((1,) + x.shape)*theta_selected, axis=2).T # (n, dim_node)

def f_pa_slice(x, theta1, theta2, I, J):
    """
    A slice of the piecewise affine function.
    """
    return f_ma_slice(theta1, x, I) - f_ma_slice(theta2, x, J) # (n, dim_node)

def sampling_size_strategy(idx, alpha=1, beta=1, n0=0):
    return max(int(alpha * (idx +1)** beta + n0), 1)

def get_epsilon_nu(nu, eps, shrinking_eps_list, ratio_list, iteration):
    if np.isscalar(shrinking_eps_list):
        shrinking_eps_list = [shrinking_eps_list]
    if np.isscalar(ratio_list):
        ratio_list = [ratio_list]
    assert len(shrinking_eps_list)==len(ratio_list)
    length = len(ratio_list)
    for idx in range(length):
        if nu < int(iteration * ratio_list[idx]):
            return shrinking_eps_list[idx]
        elif idx==length - 1:
            return eps

def A_eps_ma(theta, x, eps):
    """
    input
    - - -
    theta: (K, dim_x+1) the dim must be 2 even if K=1
    x: (N_uniq, dim_x+1) x should be preprocessed and sample idx has been uniqued
    
    output
    - - -
    indices_split: N length list [np.array([k1,k2]), np.array([k1,]), ...], 
    each list element is an eps-active index 1-d array for sample s \in [N]
    """
    max_value = f_ma_1node(theta, x)
    indices_all = np.argwhere(x.dot(theta.T) >= (max_value - eps - 1e-6).reshape((-1, 1)))
    _, indices_uniq_pos = np.unique(indices_all[:, 0], return_index=True)
    indices_split = np.split(indices_all[:,1], indices_uniq_pos[1:])
    return indices_split

def preprocessing_combinations(indices_split):
    """
    input
    - - -
    indices_split: list of length N, [np.array([x,x]), np.array([x,]), ...], 
    where each element is eps-active indices array for sample s \in [N].

    output
    - - -
    array (N,), each element is the num. of eps-active indices for each sample s \in [N]
    """
    return np.array([x.shape[0] for x in indices_split])

def IJ(indices_split, IorJcomb_num_for_sample):
    """
    input
    - - -
    indices_split: N length list [np.array([x,x]), np.array([x,]), ...], 
        each list element is an eps-active index 1-d array for sample s \in [N]
    IorJcomb_num_for_sample: array (N,), each element is the num. of eps-active indices for each sample s \in [N]
    dims_aux: array (N,), each element is an auxiliary unit which is used to locate the i-th combination in IorJcomb_num_for_sample (not used in the new version)
    occur: the order index of the wanted index combination
    
    output
    - - -
    (n_samples,) each element is an selected index of the eps-act index for that sample
    """
    ij = np.random.randint(np.zeros(IorJcomb_num_for_sample.shape[0]), IorJcomb_num_for_sample)
    output = np.zeros(IorJcomb_num_for_sample.shape[0], dtype=np.int64)
    for idx in range(IorJcomb_num_for_sample.shape[0]):
        output[idx] = indices_split[idx][ij[idx]]
    return output

def random_epsilon_active_indices_combinations(theta1, theta2, X, eps):
    n = X.shape[0]
    I_arr = np.zeros((2, n))
    for idx_theta, theta in enumerate([theta1, theta2]):
        if theta is not None:
            # print(theta.shape, X.shape)
            I_list = A_eps_ma(theta, X, eps)
            num_I = preprocessing_combinations(I_list)
            I_arr[idx_theta, :] = IJ(I_list, num_I)
    return I_arr.astype(np.int64)

def solve_esmm_erm_pp_padr(train_X_, train_Y, val_X_, val_Y, theta1_init, theta2_init, mu, eta,
    ITERATION, ALPHA, BETA, n0, EPSILON, SHRINK_EPSILON, SHRINK_RATIO, sampling, DIM_X, K1, K2, 
    c, g, b, h, W, output_log_addr):
    start_time = time.time()
    n = train_X_.shape[0]
    dim_node, dim_arc = c.shape[0], g.shape[0]

    theta1_nu, theta2_nu = theta1_init, theta2_init
    theta1_output, theta2_output = theta1_nu, theta2_nu

    train_z = f_pa(train_X_, theta1_output, theta2_output) # (n, dim_node)
    min_train_cost = pp_cost(c, g, b, h, W, train_z, train_Y)

    for nu in range(ITERATION):
        # sampling
        if sampling:
            n_nu = sampling_size_strategy(nu, ALPHA, BETA, n0)
            idcs_n_nu_uniq, idcs_n_nu_uniq_freq = np.unique(
                np.random.choice(n, n_nu, replace=True), return_counts=True)
            idcs_n_nu_uniq_freq = idcs_n_nu_uniq_freq / n_nu

            n_nu_uniq = idcs_n_nu_uniq.shape[0]
        else:
            n_nu = n
            idcs_n_nu_uniq, idcs_n_nu_uniq_freq = np.arange(n), np.ones(n) / n
            n_nu_uniq = n
            
        X_nu_, Y_nu = train_X_[idcs_n_nu_uniq, :], train_Y[idcs_n_nu_uniq, :]

        # epsilon-active indices
        eps_nu = get_epsilon_nu(nu, EPSILON, SHRINK_EPSILON, SHRINK_RATIO, ITERATION)
        ## find active indices
        mathcalI_nu_arr = np.zeros((dim_node, 2, n_nu_uniq))
        for idx_node in range(dim_node):
            theta1_nu_node = theta1_nu[idx_node, :, :] if K1 > 0 else None
            theta2_nu_node = theta2_nu[idx_node, :, :] if K2 > 0 else None
            mathcalI_nu_arr[idx_node, :, :] = random_epsilon_active_indices_combinations(
                theta1_nu_node, theta2_nu_node, X_nu_, eps_nu) # (2, n_nu_uniq)

        # build model
        # NOTICE: the following implementation only works for padrs with same K1 and K2 for all nodes
        m = grb.Model('esmm_padr_pp_subprob')
        m.Params.LogToConsole = 0
        if K1 > 0:
            theta1_var = m.addMVar((dim_node, K1, DIM_X+1), lb=-mu, ub=mu, name='theta1_var')

        if K2 > 0:
            theta2_var = m.addMVar((dim_node, K2, DIM_X+1), lb=-mu, ub=mu, name='theta2_var')
        aux_bmax0 = m.addMVar((dim_node, n_nu_uniq), lb=0, name='aux_bmax0')
        aux_padr_1 = m.addMVar((dim_node, n_nu_uniq), lb=-grb.GRB.INFINITY, name='aux_padr_1')
        aux_padr_2 = m.addMVar((dim_node, n_nu_uniq), lb=-grb.GRB.INFINITY, name='aux_padr_2')
        f = m.addMVar((dim_arc, n_nu_uniq), lb=0, name='f')

        # constraints (with surrogation)
        for idx_node in range(dim_node):
            # positive part
            if K2 > 0:
                I2_nu = mathcalI_nu_arr[idx_node, 1, :]
                I2_nu_argsort = np.argsort(I2_nu)
                I2_nu_sort = I2_nu[I2_nu_argsort].astype(np.int64)
                I2_nu_sort_uniq, I2_nu_sort_uniq_position, I2_nu_sort_uniq_counts = np.unique(I2_nu_sort, return_index=True, return_counts=True)

                X_nu_I2 = X_nu_[I2_nu_argsort]
                aux_padr_1_I2 = aux_padr_1[idx_node, :][I2_nu_argsort]

                for idx_K2 in range(I2_nu_sort_uniq.shape[0]):
                    position_start = I2_nu_sort_uniq_position[idx_K2]
                    position_end = position_start + I2_nu_sort_uniq_counts[idx_K2]
                    X_nu_I2_slice = X_nu_I2[position_start:position_end, :]
                    aux_padr_1_I2_slice = aux_padr_1_I2[position_start:position_end]
                    if K1 > 0:
                        for k1 in range(K1):
                            m.addConstr(aux_padr_1_I2_slice >= \
                                        X_nu_I2_slice @ theta1_var[idx_node, k1, :] 
                                        - X_nu_I2_slice @ theta2_var[idx_node, I2_nu_sort_uniq[idx_K2], :])
                    else:
                        m.addConstr(aux_padr_1_I2_slice >= - X_nu_I2_slice @ theta2_var[idx_node, I2_nu_sort_uniq[idx_K2], :])
            else:
                for k1 in range(K1):
                    m.addConstr(aux_padr_1[idx_node, :] >= X_nu_ @ theta1_var[idx_node, k1, :])

            # negative part
            if K1 > 0:
                I1_nu = mathcalI_nu_arr[idx_node, 0, :]
                I1_nu_argsort = np.argsort(I1_nu)
                I1_nu_sort = I1_nu[I1_nu_argsort].astype(np.int64)
                I1_nu_sort_uniq, I1_nu_sort_uniq_position, I1_nu_sort_uniq_counts = np.unique(I1_nu_sort, return_index=True, return_counts=True)

                X_nu_I1 = X_nu_[I1_nu_argsort]
                aux_padr_2_I1 = aux_padr_2[idx_node, :][I1_nu_argsort]

                for idx_K1 in range(I1_nu_sort_uniq.shape[0]):
                    position_start = I1_nu_sort_uniq_position[idx_K1]
                    position_end = position_start + I1_nu_sort_uniq_counts[idx_K1]
                    X_nu_I1_slice = X_nu_I1[position_start:position_end, :]
                    aux_padr_2_I1_slice = aux_padr_2_I1[position_start:position_end]
                    if K2 > 0:
                        for k2 in range(K2):
                            m.addConstr(aux_padr_2_I1_slice >= \
                                        X_nu_I1_slice @ theta2_var[idx_node, k2, :] 
                                        - X_nu_I1_slice @ theta1_var[idx_node, I1_nu_sort_uniq[idx_K1], :]) # -padr
                    else:
                        m.addConstr(aux_padr_2_I1_slice >= - X_nu_I1_slice @ theta1_var[idx_node, I1_nu_sort_uniq[idx_K1], :])
            else:
                for k2 in range(K2):
                    m.addConstr(aux_padr_2[idx_node, :] >= X_nu_ @ theta2_var[idx_node, k2, :])

        aux_obj = m.addMVar((n_nu_uniq, 1), lb=-grb.GRB.INFINITY, name='aux_obj')

        for i in range(n_nu_uniq):
            m.addConstr(aux_bmax0[:, i] >= Y_nu[i, :] + W @ f[:, i] + aux_padr_2[:, i])
            m.addConstr(aux_obj[i] >= c @ aux_padr_1[:, i] + g @ f[:, i] + b @ aux_bmax0[:, i])
        
        # objective
        if eta > 0:
            if K1 == 0:
                m.setObjective(idcs_n_nu_uniq_freq @ aux_obj[:, 0] 
                               + eta/2 * (sum(theta2_var[idx_node, k2,:] @ theta2_var[idx_node, k2, :] - 2 * theta2_nu[idx_node, k2, :] @ theta2_var[idx_node, k2, :] for k2 in range(K2) for idx_node in range(dim_node))))
            elif K2 == 0:
                m.setObjective(idcs_n_nu_uniq_freq @ aux_obj[:, 0] 
                               + eta/2 * (sum(theta1_var[idx_node, k1,:] @ theta1_var[idx_node, k1, :] - 2 * theta1_nu[idx_node, k1, :] @ theta1_var[idx_node, k1, :] for k1 in range(K1) for idx_node in range(dim_node))))
            else:
                m.setObjective(idcs_n_nu_uniq_freq @ aux_obj[:, 0] 
                               + eta/2 * (sum(theta1_var[idx_node, k1,:] @ theta1_var[idx_node, k1, :] - 2 * theta1_nu[idx_node, k1, :] @ theta1_var[idx_node, k1, :] for k1 in range(K1) for idx_node in range(dim_node))
                               + sum(theta2_var[idx_node, k2,:] @ theta2_var[idx_node, k2, :] - 2 * theta2_nu[idx_node, k2, :] @ theta2_var[idx_node, k2, :] for k2 in range(K2) for idx_node in range(dim_node))))
        else:
            m.setObjective(idcs_n_nu_uniq_freq @ aux_obj[:, 0])
        m.optimize()
        obj_nu = m.ObjVal
        theta1_prox = theta1_var.X if K1 > 0 else None
        theta2_prox = theta2_var.X if K2 > 0 else None

        prox_update = True
        if eps_nu == 0:
            theta1_nu, theta2_nu = theta1_prox, theta2_prox
        else:
            train_z_nu = f_pa(X_nu_, theta1_nu, theta2_nu)
            if obj_nu <= pp_cost(c, g, b, h, W, train_z_nu, Y_nu, weights=idcs_n_nu_uniq_freq):
                theta1_nu, theta2_nu = theta1_prox, theta2_prox
            else:
                prox_update = False
        
        train_z_nu = f_pa(train_X_, theta1_nu, theta2_nu)
        train_cost_nu = pp_cost(c, g, b, h, W, train_z_nu, train_Y)
        if train_cost_nu < min_train_cost:
            min_train_cost = train_cost_nu
            theta1_output, theta2_output = theta1_nu, theta2_nu
        
        exp_text = f'\n - Iter.{nu+1}: n={n_nu} eps={eps_nu:d} train_cost={train_cost_nu:.4f} obj={obj_nu:.4f} prox_update={prox_update} min_train_cost={min_train_cost:.4f}'
        write_to_file(output_log_addr, exp_text)
    
    val_cost = pp_cost(c, g, b, h, W, f_pa(val_X_, theta1_output, theta2_output), val_Y)
    exp_time = time.time() - start_time
    return min_train_cost, val_cost, exp_time, (theta1_output, theta2_output)

def exp_pp_PADR(train_X_data, train_Y_data, val_X, val_Y, test_X, test_Y, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num):
    exp_start_time = time.time()

    c, g, b, h, W = problem_dict['c'], problem_dict['g'], problem_dict['b'], problem_dict['h'], problem_dict['W']
    DIM_X = problem_dict['x_dim']
    DIM_NODE, DIM_ARC = c.shape[0], g.shape[0]

    train_X_data_ = np.hstack((train_X_data, np.ones((train_X_data.shape[0], 1)))) # adding intercept
    val_X_ = np.hstack((val_X, np.ones((val_X.shape[0], 1)))) 
    test_X_ = np.hstack((test_X, np.ones((test_X.shape[0], 1))))

    outerLoopParams_dict, outerNumTotal = load_loop_params(problem_dict, 'training_sample_size')
    for idx_outerLoop in range(outerNumTotal):
        n, = pick_loop_params(idx_outerLoop, outerNumTotal, outerLoopParams_dict)
        assert n > 0
        if n > train_X_data_.shape[0]:
            print(f'Warning: n={n} is larger than the training sample size. Set n to the training data size ({train_X_data_.shape[0]}).')
            n = train_X_data_.shape[0]
        # assert n <= train_X_data_.shape[0]
        train_X_, train_Y = train_X_data_[:n], train_Y_data[:n]
        exp_text = f'\n# OuterLoop{idx_outerLoop}: n={n}'
        write_to_file(output_log_addr, exp_text)

        innerLoopParams_dict, innerNumTotal = load_loop_params(param_dict, *param_dict.keys())

        val_costs, theta_tuples = [], []
        for idx_innerLoop in range(innerNumTotal):
            (K1, K2), EPSILON, (SHRINK_EPSILON, SHRINK_RATIO), ROUND, (ITERATION, ALPHA, BETA, n0), MU, ETA, SAMPLING = \
                pick_loop_params(idx_innerLoop, innerNumTotal, innerLoopParams_dict)
            exp_text = f'\n## InnerLoop{idx_innerLoop}: K1K2=({K1},{K2}), eps={EPSILON}, shrink_eps_ratio=({SHRINK_EPSILON},{SHRINK_RATIO}), round={ROUND}, iter_a_b_N0=({ITERATION},{ALPHA},{BETA},{n0}), mu={MU}, eta={ETA}, sampling={SAMPLING}'
            write_to_file(output_log_addr, exp_text)
            
            train_cost_log, val_cost_log, total_time, theta_tuple_log = 1e10, 1e10, 0, None
            for idx_round in range(ROUND):
                theta1_init = None if K1==0 else np.random.uniform(-MU/2, MU/2, (DIM_NODE, K1, DIM_X+1))
                theta2_init = None if K2==0 else np.random.uniform(-MU/2, MU/2, (DIM_NODE, K2, DIM_X+1))

                min_train_cost_round, val_cost_round, time_round, (theta1_output, theta2_output) = solve_esmm_erm_pp_padr(
                    train_X_, train_Y, val_X_, val_Y, theta1_init, theta2_init, MU, ETA,
                    ITERATION, ALPHA, BETA, n0, 
                    EPSILON, SHRINK_EPSILON, SHRINK_RATIO, SAMPLING,
                    DIM_X, K1, K2,
                    c, g, b, h, W,
                    output_log_addr)
                
                total_time += time_round
                if min_train_cost_round < train_cost_log:
                    train_cost_log, val_cost_log = min_train_cost_round, val_cost_round
                    theta_tuple_log = (theta1_output, theta2_output)
                
                exp_text = f'\n### Round{idx_round+1}: train_cost={min_train_cost_round:.4f}, val_cost={val_cost_round:.4f}, time={time_round:.2f}s\n'
                write_to_file(output_log_addr, exp_text)

                update_progress(
                    (idx_outerLoop*(innerNumTotal*ROUND) + idx_innerLoop*ROUND + idx_round+1)/(outerNumTotal*innerNumTotal*ROUND),
                    exp_start_time, 'PADR', curr_exp_idx, total_exp_num)
            val_costs.append(val_cost_log)
            theta_tuples.append(theta_tuple_log)
            exp_text = f'\nn={n:d} param_idx={idx_innerLoop+1:d}/{innerNumTotal:d} val_cost={val_cost_log:.4f} time={total_time:.2f} train_cost={train_cost_log:.4f}'
            write_to_file(output_log_addr, exp_text)
        
        min_val_cost, min_val_cost_idx = np.min(val_costs), np.argmin(val_costs)
        theta1_final, theta2_final = theta_tuples[min_val_cost_idx]
        test_z = f_pa(test_X_, theta1_final, theta2_final)
        test_cost = pp_cost(c, g, b, h, W, test_z, test_Y)
        exp_text = f'\nOuterLoop{idx_outerLoop} (n={n}) with innerLoop{min_val_cost_idx+1}: val_cost={min_val_cost:.4f} test_cost={test_cost:.4f}\n\n'
        write_to_file(output_log_addr, exp_text)

def solve_esmm_erm_nv_padr(train_X_, train_Y, val_X_, val_Y, theta1_init, theta2_init, mu, eta,
    ITERATION, ALPHA, BETA, n0, EPSILON, SHRINK_EPSILON, SHRINK_RATIO, sampling, DIM_X, K1, K2, cb, ch, capacity_arr, output_log_addr):
    start_time = time.time()
    n = train_X_.shape[0]

    theta1_nu, theta2_nu = theta1_init, theta2_init
    theta1_output, theta2_output = theta1_nu, theta2_nu

    train_z = f_pa(train_X_, theta1_output, theta2_output) # (n, dim_node)
    min_train_cost = nv_cost(cb, ch, train_z, train_Y) + capacost_function_oneprod(train_z, capacity_arr)

    for nu in range(ITERATION):
        # sampling strategy: determine the samples used in this iteration
        if sampling:
            n_nu = sampling_size_strategy(nu, ALPHA, BETA, n0)
            idcs_n_nu_uniq, idcs_n_nu_uniq_freq = np.unique(
                np.random.choice(n, n_nu, replace=True), return_counts=True)
            idcs_n_nu_uniq_freq = idcs_n_nu_uniq_freq / n_nu
            n_nu_uniq = idcs_n_nu_uniq.shape[0]
        else:
            n_nu = n
            idcs_n_nu_uniq, idcs_n_nu_uniq_freq = np.arange(n), np.ones(n) / n
            n_nu_uniq = n

        X_nu_, Y_nu = train_X_[idcs_n_nu_uniq, :], train_Y[idcs_n_nu_uniq]

        # epsilon-active indices
        eps_nu = get_epsilon_nu(nu, EPSILON, SHRINK_EPSILON, SHRINK_RATIO, ITERATION)
        ## find eps-active indices
        mathcalI_nu_arr = random_epsilon_active_indices_combinations(theta1_nu, theta2_nu, X_nu_, eps_nu)
        if capacity_arr is not None:
            # if the problem has a capacity cost in the objective, also find the active indices for the outer capacity cost function
            capacity_z = f_pa(X_nu_, theta1_nu, theta2_nu)
            capacity_z_ = np.hstack((capacity_z.reshape((-1, 1)), np.ones((capacity_z.shape[0], 1))))
            CJ_nu = A_eps_ma(capacity_arr, capacity_z_, 0)
            CJcomb_num_for_sample = preprocessing_combinations(CJ_nu)
            mathcalCJ_nu_arr = IJ(CJ_nu, CJcomb_num_for_sample)

        # build the optimization model
        m = grb.Model('esmm_padr_nv_subprob')
        m.Params.LogToConsole = 0
        if K1 > 0:
            theta1_var = m.addMVar((K1, DIM_X+1), lb=-mu, ub=mu, name='theta1_var')
        if K2 > 0:
            theta2_var = m.addMVar((K2, DIM_X+1), lb=-mu, ub=mu, name='theta2_var')
        vB = m.addMVar(n_nu_uniq, lb=0, name='arti-var for back') # auxiliary variables
        vH = m.addMVar(n_nu_uniq, lb=0, name='arti-var for hold')
        if capacity_arr is not None:
            vcapacost = m.addMVar(n_nu_uniq, lb = 0, name='arti-var for capacost')

        # adding constraints for the penalty part of the newsvendor problem (including surrogation)
        if K1 > 0:
            # For each sample X, we may have a different active index.
            # Adding constraints one by one for each sample is not efficient, so we group the samples by their active indices combinations, and then add constraints for each group.
            # Since the piece number of PADR (K1, K2) is relatively small, the number of groups is not large, and it can fasten the building process.
            I_nu_argsort = np.argsort(mathcalI_nu_arr[0], kind='mergesort')
            X_nu_I = X_nu_[I_nu_argsort]
            Y_nu_I = Y_nu[I_nu_argsort]
            I_nu_sort = mathcalI_nu_arr[0][I_nu_argsort]
            I_nu_sort_uniq, I_nu_sort_uniq_pos, I_nu_sort_uniq_counts = \
                np.unique(I_nu_sort, return_index=True, return_counts=True) # K1 index, start position in X_nu_I, length in X_nu_I

            for idx_K1 in range(I_nu_sort_uniq.shape[0]):
                pos_start = I_nu_sort_uniq_pos[idx_K1]
                pos_end = pos_start + I_nu_sort_uniq_counts[idx_K1]
                X_nu_I_remake = X_nu_I[pos_start:pos_end, :]
                Y_nu_I_remake = Y_nu_I[pos_start:pos_end]
                if K2 != 0:
                    for k2 in range(K2):
                        m.addConstr(vB[pos_start:pos_end] - Y_nu_I_remake >= \
                            X_nu_I_remake @ theta2_var[k2, :] - X_nu_I_remake @ theta1_var[I_nu_sort_uniq[idx_K1], :])
                else:
                    m.addConstr(
                        vB[pos_start:pos_end] - Y_nu_I_remake >= - X_nu_I_remake @ theta1_var[I_nu_sort_uniq[idx_K1], :])
        else:
            for k2 in range(K2):
                m.addConstr(vB - Y_nu >= X_nu_ @ theta2_var[k2, :])
        
        # adding constraints for the holding part of the newsvendor problem (including surrogation)
        if K2 > 0:
            J_nu_argsort = np.argsort(mathcalI_nu_arr[1], kind='mergesort')
            X_nu_J = X_nu_[J_nu_argsort]
            Y_nu_J = Y_nu[J_nu_argsort]
            J_nu_sort = mathcalI_nu_arr[1][J_nu_argsort]
            J_nu_sort_uniq, J_nu_sort_uniq_pos, J_nu_sort_uniq_counts = \
                np.unique(J_nu_sort, return_index=True, return_counts=True)
            if capacity_arr is not None:
                CJ_nu_sort = mathcalCJ_nu_arr[J_nu_argsort]
            for idx_K2 in range(J_nu_sort_uniq.shape[0]):
                pos_start = J_nu_sort_uniq_pos[idx_K2]
                pos_end = pos_start + J_nu_sort_uniq_counts[idx_K2]
                X_nu_J_remake = X_nu_J[pos_start:pos_end, :]
                Y_nu_J_remake = Y_nu_J[pos_start:pos_end]
                if K1 > 0:
                    for k1 in range(K1):
                        m.addConstr(
                            vH[pos_start:pos_end] + Y_nu_J_remake >= \
                                X_nu_J_remake @ theta1_var[k1, :] - X_nu_J_remake @ theta2_var[J_nu_sort_uniq[idx_K2], :])
                        if capacity_arr is not None:
                            costparam_pos = capacity_arr[CJ_nu_sort[pos_start:pos_end], :]
                            X_nu_J_remake = X_nu_J_remake * np.tile(-costparam_pos[:, 0].reshape((-1,1)), X_nu_J_remake.shape[1])
                            m.addConstr(
                                vcapacost[pos_start:pos_end] >= X_nu_J_remake @ theta1_var[k1, :] 
                                    - X_nu_J_remake @ theta2_var[J_nu_sort_uniq[idx_K2], :] - costparam_pos[:, 1])
                else:
                    m.addConstr(
                        vH[pos_start:pos_end] + Y_nu_J_remake >= - X_nu_J_remake @ theta2_var[J_nu_sort_uniq[idx_K2], :])
                    if capacity_arr is not None:
                        costparam_pos = capacity_arr[CJ_nu_sort[pos_start:pos_end], :]
                        X_nu_J_remake = X_nu_J_remake * np.tile(-costparam_pos[:, 0].reshape((-1,1)), X_nu_J_remake.shape[1])
                        m.addConstr(
                            vcapacost[pos_start:pos_end] >= - X_nu_J_remake @ theta2_var[J_nu_sort_uniq[idx_K2], :] - costparam_pos[:, 1])
        else:
            for k1 in range(K1):
                m.addConstr(vH + Y_nu >= X_nu_ @ theta1_var[k1, :])
                if capacity_arr is not None:
                    costparam_pos = capacity_arr[mathcalCJ_nu_arr, :]
                    X_nu_capa_remake = X_nu_ * np.tile(-costparam_pos[:, 0].reshape((-1,1)), X_nu_.shape[1])
                    m.addConstr(vcapacost[:] >= X_nu_capa_remake @ theta1_var[k1, :] - costparam_pos[:, 1])

        # set objective
        if capacity_arr is None:
            if eta > 0:
                if K1 == 0:
                    m.setObjective(
                        (idcs_n_nu_uniq_freq * cb) @ vB + (idcs_n_nu_uniq_freq * ch) @ vH 
                        + eta/2 * (sum(theta2_var[k2,:] @ theta2_var[k2, :] - 2 * theta2_nu[k2, :] @ theta2_var[k2, :] for k2 in range(K2))))
                elif K2 == 0:
                    m.setObjective(
                        (idcs_n_nu_uniq_freq * cb) @ vB + (idcs_n_nu_uniq_freq * ch) @ vH 
                        + eta/2 * (sum(theta1_var[k1,:] @ theta1_var[k1, :] - 2 * theta1_nu[k1, :] @ theta1_var[k1, :] for k1 in range(K1))))
                else:
                    m.setObjective(
                        (idcs_n_nu_uniq_freq * cb) @ vB + (idcs_n_nu_uniq_freq * ch) @ vH 
                        + eta/2 * (
                            sum(theta1_var[k1,:] @ theta1_var[k1, :] - 2 * theta1_nu[k1, :] @ theta1_var[k1, :] for k1 in range(K1))
                            + sum(theta2_var[k2,:] @ theta2_var[k2, :] - 2 * theta2_nu[k2, :] @ theta2_var[k2, :] for k2 in range(K2))))
            else:
                m.setObjective((idcs_n_nu_uniq_freq * cb) @ vB + (idcs_n_nu_uniq_freq * ch) @ vH)
        else:
            if eta > 0:
                m.setObjective(
                    (idcs_n_nu_uniq_freq * cb) @ vB + (idcs_n_nu_uniq_freq * ch) @ vH + idcs_n_nu_uniq_freq @ vcapacost 
                    + eta/2 * (
                    sum(theta1_var[k1,:] @ theta1_var[k1, :] - 2 * theta1_nu[k1, :] @ theta1_var[k1, :] for k1 in range(K1))
                    + sum(theta2_var[k2,:] @ theta2_var[k2, :] - 2 * theta2_nu[k2, :] @ theta2_var[k2, :] for k2 in range(K2))))
            else:
                m.setObjective((idcs_n_nu_uniq_freq * cb) @ vB + (idcs_n_nu_uniq_freq * ch) @ vH + idcs_n_nu_uniq_freq @ vcapacost)

        m.optimize()
        obj_nu = m.ObjVal
        theta1_prox = theta1_var.X if K1 > 0 else None
        theta2_prox = theta2_var.X if K2 > 0 else None

        prox_update = True
        if eps_nu == 0:
            theta1_nu, theta2_nu = theta1_prox, theta2_prox
        else:
            train_z_nu = f_pa(X_nu_, theta1_nu, theta2_nu)
            if obj_nu <= nv_cost(cb, ch, train_z_nu, Y_nu, weights=idcs_n_nu_uniq_freq) + capacost_function_oneprod(train_z_nu, capacity_arr, weights=idcs_n_nu_uniq_freq):
                theta1_nu, theta2_nu = theta1_prox, theta2_prox
            else:
                prox_update = False
        
        train_z_nu = f_pa(train_X_, theta1_nu, theta2_nu)
        train_cost_nu = nv_cost(cb, ch, train_z_nu, train_Y) + capacost_function_oneprod(train_z_nu, capacity_arr)
        if train_cost_nu < min_train_cost:
            min_train_cost = train_cost_nu
            theta1_output, theta2_output = theta1_nu, theta2_nu
        
        exp_text = f'\n - Iter.{nu+1}: n={n_nu} eps={eps_nu:d} train_cost={train_cost_nu:.4f} obj={obj_nu:.4f} prox_update={prox_update} min_train_cost={min_train_cost:.4f}'
        write_to_file(output_log_addr, exp_text)
    
    val_z = f_pa(val_X_, theta1_output, theta2_output)
    val_cost = nv_cost(cb, ch, val_z, val_Y) + capacost_function_oneprod(val_z, capacity_arr)
    exp_time = time.time() - start_time
    return min_train_cost, val_cost, exp_time, (theta1_output, theta2_output)

def exp_nv_PADR(train_X_data, train_Y_data, val_X, val_Y, test_X, test_Y, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num, include_val=False):
    exp_start_time = time.time()

    cb, ch = problem_dict['c_b'], problem_dict['c_h']
    DIM_X = problem_dict['x_dim']

    train_X_data_ = np.hstack((train_X_data, np.ones((train_X_data.shape[0], 1)))) # adding intercept for padr
    val_X_ = np.hstack((val_X, np.ones((val_X.shape[0], 1)))) 
    test_X_ = np.hstack((test_X, np.ones((test_X.shape[0], 1))))

    outerLoopParams_dict, outerNumTotal = load_loop_params(problem_dict, 'training_sample_size')
    # outer loop: training sample size
    for idx_outerLoop in range(outerNumTotal):
        n, = pick_loop_params(idx_outerLoop, outerNumTotal, outerLoopParams_dict)
        assert n > 0
        if n > train_X_data_.shape[0]:
            print(f'Warning: n={n} is larger than the training sample size. Set n to the training data size ({train_X_data_.shape[0]}).')
            n = train_X_data_.shape[0]
        train_X_, train_Y = train_X_data_[:n], train_Y_data[:n]
        exp_text = f'\n# OuterLoop{idx_outerLoop}: n={n}'
        write_to_file(output_log_addr, exp_text)

        innerLoopParams_dict, innerNumTotal = load_loop_params(param_dict, *param_dict.keys())
        
        val_costs = []
        theta_tuples = []
        # inner loop: hyperparameters
        for idx_innerLoop in range(innerNumTotal):
            (K1, K2), EPSILON, (SHRINK_EPSILON, SHRINK_RATIO), ROUND, (ITERATION, ALPHA, BETA, n0), MU, ETA, SAMPLING = \
                pick_loop_params(idx_innerLoop, innerNumTotal, innerLoopParams_dict)
            exp_text = f'\n## InnerLoop{idx_innerLoop}: K1K2=({K1},{K2}), eps={EPSILON}, shrink_eps_ratio=({SHRINK_EPSILON},{SHRINK_RATIO}), round={ROUND}, iter_a_b_N0=({ITERATION},{ALPHA},{BETA},{n0}), mu={MU}, eta={ETA}, sampling={SAMPLING}'
            write_to_file(output_log_addr, exp_text)

            train_cost_log, val_cost_log, total_time, theta_tuple_log = 1e10, 1e10, 0, None
            for idx_round in range(ROUND):
                theta1_init = None if K1==0 else np.random.uniform(-MU, MU, (K1, DIM_X+1))
                theta2_init = None if K2==0 else np.random.uniform(-MU, MU, (K2, DIM_X+1))
                # run ESMM in each round
                min_train_cost_round, val_cost_round, time_round, (theta1_output, theta2_output) = solve_esmm_erm_nv_padr(
                    train_X_, train_Y, val_X_, val_Y, theta1_init, theta2_init, MU, ETA,
                    ITERATION, ALPHA, BETA, n0, 
                    EPSILON, SHRINK_EPSILON, SHRINK_RATIO, SAMPLING,
                    DIM_X, K1, K2,
                    cb, ch, problem_dict['ncvx_capacity'],
                    output_log_addr)

                total_time += time_round
                if min_train_cost_round < train_cost_log:
                    train_cost_log, val_cost_log = min_train_cost_round, val_cost_round
                    theta_tuple_log = (theta1_output, theta2_output)

                exp_text = f'\n### Round{idx_round+1}: train_cost={min_train_cost_round:.4f}, val_cost={val_cost_round:.4f}, time={time_round:.2f}s\n'
                write_to_file(output_log_addr, exp_text)

                update_progress(
                    (idx_outerLoop*(innerNumTotal*ROUND) + idx_innerLoop*ROUND + idx_round+1)/(outerNumTotal*innerNumTotal*ROUND),
                    exp_start_time, 'PADR', curr_exp_idx, total_exp_num)

            val_costs.append(val_cost_log)
            theta_tuples.append(theta_tuple_log)
            exp_text = f'\nn={n:d} param_idx={idx_innerLoop+1:d}/{innerNumTotal:d} val_cost={val_cost_log:.4f} time={total_time:.2f} train_cost={train_cost_log:.4f}'
            write_to_file(output_log_addr, exp_text)
        
        # find the best hyperparameters and test
        min_val_cost, min_val_cost_idx = np.min(val_costs), np.argmin(val_costs)
        theta1_final, theta2_final = theta_tuples[min_val_cost_idx]
        if include_val:
            (K1, K2), EPSILON, (SHRINK_EPSILON, SHRINK_RATIO), ROUND, (ITERATION, ALPHA, BETA, n0), MU, ETA, SAMPLING = pick_loop_params(min_val_cost_idx, innerNumTotal, innerLoopParams_dict)
            exp_text = f'\n## Testing Stage: Using params of innerLoop{idx_innerLoop}: K1K2=({K1},{K2}), eps={EPSILON}, shrink_eps_ratio=({SHRINK_EPSILON},{SHRINK_RATIO}), round={ROUND}, iter_a_b_N0=({ITERATION},{ALPHA},{BETA},{n0}), mu={MU}, eta={ETA}, sampling={SAMPLING}'
            write_to_file(output_log_addr, exp_text)
            train_X_final_ = np.vstack((train_X_, val_X_))
            train_Y_final = np.vstack((train_Y, val_Y)) if len(train_Y.shape)>1 else np.hstack((train_Y, val_Y))
            train_cost_log, total_time, theta_tuple_log = 1e10, 0, None
            for idx_round in range(ROUND):
                theta1_init = None if K1==0 else np.random.uniform(-MU, MU, (K1, DIM_X+1))
                theta2_init = None if K2==0 else np.random.uniform(-MU, MU, (K2, DIM_X+1))

                min_train_cost_round, _, time_round, (theta1_output, theta2_output) = solve_esmm_erm_nv_padr(
                    train_X_final_, train_Y_final, val_X_, val_Y, theta1_init, theta2_init, MU, ETA,
                    ITERATION, ALPHA, BETA, n0, 
                    EPSILON, SHRINK_EPSILON, SHRINK_RATIO, SAMPLING,
                    DIM_X, K1, K2,
                    cb, ch, problem_dict['ncvx_capacity'],
                    output_log_addr)

                total_time += time_round
                if min_train_cost_round < train_cost_log:
                    train_cost_log = min_train_cost_round
                    theta1_final, theta2_final = theta1_output, theta2_output

                exp_text = f'\n### Round{idx_round+1}: train_cost={min_train_cost_round:.4f}, time={time_round:.2f}s\n'
                write_to_file(output_log_addr, exp_text)

        test_z_final = f_pa(test_X_, theta1_final, theta2_final)
        test_cost = nv_cost(cb, ch, test_z_final, test_Y) + capacost_function_oneprod(test_z_final, problem_dict['ncvx_capacity'])
        exp_text = f'\nOuterLoop{idx_outerLoop} (n={n}) with innerLoop{min_val_cost_idx+1}: val_cost={min_val_cost:.4f} test_cost={test_cost:.4f}\n\n'
        write_to_file(output_log_addr, exp_text)

def compute_trainobj_2prod(z_arr, Y, cb_list, ch_list, lbd, gamma, capacity, weights=None, constr_arr=None, rkhs=None):
    if constr_arr is None:
        z_arr = z_arr * (z_arr >= 0)
        z_overflow = np.sum(z_arr, axis=1) - capacity
        z_overflow = np.maximum(z_overflow, 0)
        real_z_arr = z_arr - (z_overflow/z_arr.shape[1]).reshape((-1, 1))
        penalty = lbd * np.max((np.sum(z_arr, axis=1) - capacity + gamma, np.zeros(z_arr.shape[0])), axis=0)
        obj = nv_cost(cb_list[0], ch_list[0], real_z_arr[:, 0], Y[:, 0]) + nv_cost(cb_list[1], ch_list[1], real_z_arr[:, 1], Y[:, 1]) + penalty
    else:
        if rkhs is None:
            z_arr = z_arr * (z_arr >= 0)
            capacity_cost = np.sum([capacost_function_oneprod(z_arr[:, idx], constr_arr, average=False) for idx in range(z_arr.shape[1])], axis=0)
            penalty = lbd * np.maximum(capacity_cost - capacity + gamma, 0)
        else:
            capacity_cost = np.sum([capacost_function_oneprod(z_arr[:, idx], constr_arr, average=False) for idx in range(z_arr.shape[1])], axis=0)
            penalty = lbd * np.maximum(capacity_cost - capacity + gamma, 0)**2 + rkhs['lbd'] * (np.sum(rkhs['param1'] * (rkhs['kernel'].dot(rkhs['param1']))) 
                        + np.sum(rkhs['param2'] * (rkhs['kernel'].dot(rkhs['param2']))))
        obj = nv_cost(cb_list[0], ch_list[0], z_arr[:, 0], Y[:, 0]) + nv_cost(cb_list[1], ch_list[1], z_arr[:, 1], Y[:, 1]) + penalty
    
    if weights is None:
        return np.mean(obj)
    else:
        return np.sum(obj * weights)

def solve_esmm_erm_nv2prod_padr(
    train_X_, train_Y, val_X_, val_Y, theta1_init, theta2_init, mu, eta,
    iteration, alpha, beta, n0, 
    epsilon, shrink_epsilon, shrink_ratio, sampling, penalty_lbd, penalty_gamma,
    dim_x, K1, K2, cb1, cb2, ch1, ch2, capacity,
    output_log_addr):
    
    start_time = time.time()
    n = train_X_.shape[0]
    dim_node = train_Y.shape[1]

    theta1_nu, theta2_nu = theta1_init, theta2_init
    theta1_output, theta2_output = theta1_nu, theta2_nu

    train_z_output = f_pa(train_X_, theta1_output, theta2_output)
    min_train_cost, _ = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, train_z_output, train_Y)
    # min_train_obj = compute_trainobj_2prod(train_z, train_Y, [cb1, cb2], [ch1, ch2], penalty_lbd, penalty_gamma, capacity)

    for nu in range(iteration):
        # sampling
        if sampling:
            n_nu = sampling_size_strategy(nu, alpha, beta, n0)
            idcs_n_nu_uniq, idcs_n_nu_uniq_freq = np.unique(
                np.random.choice(n, n_nu, replace=True), return_counts=True)
            idcs_n_nu_uniq_freq = idcs_n_nu_uniq_freq / n_nu
            n_nu_uniq = idcs_n_nu_uniq.shape[0]
        else:
            n_nu = n
            idcs_n_nu_uniq, idcs_n_nu_uniq_freq = np.arange(n), np.ones(n) / n
            n_nu_uniq = n
        
        X_nu_, Y_nu = train_X_[idcs_n_nu_uniq, :], train_Y[idcs_n_nu_uniq, :]

        # epsilon-active indices
        eps_nu = get_epsilon_nu(nu, epsilon, shrink_epsilon, shrink_ratio, iteration)
        
        mathcalI_nu_arr_tup = (
            random_epsilon_active_indices_combinations(theta1_nu[0], theta2_nu[0], X_nu_, eps_nu),
            random_epsilon_active_indices_combinations(theta1_nu[1], theta2_nu[1], X_nu_, eps_nu))

        m = grb.Model('esmm_padr_nv2prod_subprob')
        m.Params.LogToConsole = 0
        if K1 > 0:
            theta1_var = m.addMVar((dim_node, K1, dim_x+1), lb=-mu, ub=mu, name='theta1_var')
        if K2 > 0:
            theta2_var = m.addMVar((dim_node, K2, dim_x+1), lb=-mu, ub=mu, name='theta2_var')
        vB = m.addMVar((dim_node, n_nu_uniq), lb=0, name='arti-var for back') # auxiliary variables
        vH = m.addMVar((dim_node, n_nu_uniq), lb=0, name='arti-var for hold')
        vpenalty = m.addMVar(n_nu_uniq, lb=0, name='arti-var for penalty')
        vpenalty_node = m.addMVar((dim_node, n_nu_uniq), lb=0, name='arti-var for penalty of two products')

        # constraints for penalization
        X_nu_penremake = np.zeros_like(X_nu_)
        active_K2_combs, active_sampleaxis_pos, active_sampleaxis_counts = [], [], []
        J_stack = np.hstack(
            (mathcalI_nu_arr_tup[0][1].reshape((-1, 1)), mathcalI_nu_arr_tup[1][1].reshape((-1,1))))
        pos = 0
        for p1k2 in range(K2):
            for p2k2 in range(K2):
                chosen_idcs = np.all(J_stack[:]==np.array([p1k2, p2k2]), axis=1)
                num_p1k2p2k2activesamples = np.sum(chosen_idcs)
                if num_p1k2p2k2activesamples==0:
                    continue
                X_nu_penremake[pos:pos+num_p1k2p2k2activesamples] = X_nu_[chosen_idcs,:]
                active_K2_combs.append([p1k2, p2k2])
                active_sampleaxis_pos.append(pos)
                active_sampleaxis_counts.append(num_p1k2p2k2activesamples)
                pos += num_p1k2p2k2activesamples
        for idx_K2comb in range(len(active_K2_combs)):
            p1k2, p2k2 = active_K2_combs[idx_K2comb][0], active_K2_combs[idx_K2comb][1]
            pos_start = active_sampleaxis_pos[idx_K2comb]
            pos_end = pos_start + active_sampleaxis_counts[idx_K2comb]
            for p1k1 in range(K1):
                for p2k1 in range(K1):
                    m.addConstr(
                        vpenalty[pos_start:pos_end] + capacity - penalty_gamma >= \
                            X_nu_penremake[pos_start:pos_end,:] @ theta1_var[0, p1k1,:] - X_nu_penremake[pos_start:pos_end,:] @ theta2_var[0, p1k2,:] + \
                            X_nu_penremake[pos_start:pos_end,:] @ theta1_var[1, p2k1,:] - X_nu_penremake[pos_start:pos_end,:] @ theta2_var[1, p2k2,:]
                    )
        # constraints for nv objective components
        def generate_sorted_components(X_nu, Y_nu, I_nu,):
            I_nu_argsort = np.argsort(I_nu, kind='mergesort')
            X_nu_I, Y_nu_I = X_nu[I_nu_argsort], Y_nu[I_nu_argsort]
            I_nu_sorted = I_nu[I_nu_argsort]
            I_nu_sorteduniq, I_nu_sorteduniq_pos, I_nu_sorteduniq_counts = np.unique(I_nu_sorted, return_index=True, return_counts=True)
            return I_nu_sorteduniq, I_nu_sorteduniq_pos, I_nu_sorteduniq_counts, X_nu_I, Y_nu_I
        
        p1I_nu_sorteduniq, p1I_nu_sorteduniq_pos, p1I_nu_sorteduniq_counts, p1X_nu_I, p1Y_nu_I = generate_sorted_components(X_nu_, Y_nu[:,0], mathcalI_nu_arr_tup[0][0])
        p2I_nu_sorteduniq, p2I_nu_sorteduniq_pos, p2I_nu_sorteduniq_counts, p2X_nu_I, p2Y_nu_I = generate_sorted_components(X_nu_, Y_nu[:,1], mathcalI_nu_arr_tup[1][0])

        def add_nvconstr_max1(m, vB, vpenp, vartheta1, vartheta2, X_nu_remake, Y_nu_remake, I_nu_sorteduniq, I_nu_sorteduniq_pos, I_nu_sorteduniq_counts, K2):
            for idx_K1 in range(I_nu_sorteduniq.shape[0]):
                k1 = I_nu_sorteduniq[idx_K1]
                for k2 in range(K2):
                    pos_start = I_nu_sorteduniq_pos[idx_K1]
                    pos_end = pos_start + I_nu_sorteduniq_counts[idx_K1]
                    m.addConstr(
                        vB[pos_start:pos_end] - Y_nu_remake[pos_start:pos_end] >= \
                            X_nu_remake[pos_start:pos_end, :] @ vartheta2[k2, :] - \
                            X_nu_remake[pos_start:pos_end, :] @ vartheta1[k1, :])
                    m.addConstr(
                        vpenp[pos_start:pos_end] >= X_nu_remake[pos_start:pos_end, :] @ vartheta2[k2, :] - X_nu_remake[pos_start:pos_end, :] @ vartheta1[k1, :])
        add_nvconstr_max1(
            m, vB[0], vpenalty_node[0], theta1_var[0], theta2_var[0], p1X_nu_I, p1Y_nu_I, p1I_nu_sorteduniq, p1I_nu_sorteduniq_pos, p1I_nu_sorteduniq_counts, K2)
        add_nvconstr_max1(
            m, vB[1], vpenalty_node[1], theta1_var[1], theta2_var[1], p2X_nu_I, p2Y_nu_I, p2I_nu_sorteduniq, p2I_nu_sorteduniq_pos, p2I_nu_sorteduniq_counts, K2)
        
        p1J_nu_sorteduniq, p1J_nu_sorteduniq_pos, p1J_nu_sorteduniq_counts, p1X_nu_J, p1Y_nu_J = generate_sorted_components(X_nu_, Y_nu[:,0], mathcalI_nu_arr_tup[0][1])
        p2J_nu_sorteduniq, p2J_nu_sorteduniq_pos, p2J_nu_sorteduniq_counts, p2X_nu_J, p2Y_nu_J = generate_sorted_components(X_nu_, Y_nu[:,1], mathcalI_nu_arr_tup[1][1])

        def add_nvconstr_max2(m, vH, vartheta1, vartheta2, X_nu_remake, Y_nu_remake, J_nu_sorteduniq, J_nu_sorteduniq_pos, J_nu_sorteduniq_counts, K1):
            for idx_K2 in range(J_nu_sorteduniq.shape[0]):
                k2 = J_nu_sorteduniq[idx_K2]
                for k1 in range(K1):
                    pos_start = J_nu_sorteduniq_pos[idx_K2]
                    pos_end = pos_start + J_nu_sorteduniq_counts[idx_K2]
                    m.addConstr(
                        vH[pos_start:pos_end] + Y_nu_remake[pos_start:pos_end] >= \
                            X_nu_remake[pos_start:pos_end, :] @ vartheta1[k1, :] - \
                            X_nu_remake[pos_start:pos_end, :] @ vartheta2[k2, :]
                    )
        add_nvconstr_max2(
            m, vH[0], theta1_var[0], theta2_var[0], p1X_nu_J, p1Y_nu_J, p1J_nu_sorteduniq, p1J_nu_sorteduniq_pos, p1J_nu_sorteduniq_counts, K1)
        add_nvconstr_max2(
            m, vH[1], theta1_var[1], theta2_var[1], p2X_nu_J, p2Y_nu_J, p2J_nu_sorteduniq, p2J_nu_sorteduniq_pos, p2J_nu_sorteduniq_counts, K1)

        if eta > 0:
            m.setObjective((idcs_n_nu_uniq_freq * cb1) @ vB[0] + (idcs_n_nu_uniq_freq * ch1) @ vH[0] + \
                (idcs_n_nu_uniq_freq * cb2) @ vB[1] + (idcs_n_nu_uniq_freq * ch2) @ vH[1] + \
                (idcs_n_nu_uniq_freq * penalty_lbd) @ vpenalty + (idcs_n_nu_uniq_freq * penalty_lbd) @ vpenalty_node[0] + (idcs_n_nu_uniq_freq * penalty_lbd) @ vpenalty_node[1] + \
                eta/2 * (
                sum(theta1_var[0, k1,:] @ theta1_var[0, k1, :] - 2 * theta1_nu[0, k1, :] @ theta1_var[0, k1, :] for k1 in range(K1))
                + sum(theta2_var[0, k2,:] @ theta2_var[0, k2, :] - 2 * theta2_nu[0, k2, :] @ theta2_var[0, k2, :] for k2 in range(K2))
                + sum(theta1_var[1, k1,:] @ theta1_var[1, k1, :] - 2 * theta1_nu[1, k1, :] @ theta1_var[1, k1, :] for k1 in range(K1))
                + sum(theta2_var[1, k2,:] @ theta2_var[1, k2, :] - 2 * theta2_nu[1, k2, :] @ theta2_var[1, k2, :] for k2 in range(K2))))
        else:
            m.setObjective((idcs_n_nu_uniq_freq * cb1) @ vB[0] + (idcs_n_nu_uniq_freq * ch1) @ vH[0] + \
                (idcs_n_nu_uniq_freq * cb2) @ vB[1] + (idcs_n_nu_uniq_freq * ch2) @ vH[1] + \
                (idcs_n_nu_uniq_freq * penalty_lbd) @ vpenalty + (idcs_n_nu_uniq_freq * penalty_lbd) @ vpenalty_node[0] + (idcs_n_nu_uniq_freq * penalty_lbd) @ vpenalty_node[1])
        
        m.optimize()
        
        obj_nu = m.ObjVal
        theta1_prox = theta1_var.X if K1 > 0 else None
        theta2_prox = theta2_var.X if K2 > 0 else None
        # validation step and update theta_nu
        prox_update = True
        if eps_nu == 0:
            theta1_nu, theta2_nu = theta1_prox, theta2_prox
        else:
            train_z_nu = f_pa(X_nu_, theta1_nu, theta2_nu)
            if obj_nu <= compute_trainobj_2prod(train_z_nu, Y_nu, [cb1, cb2], [ch1, ch2], penalty_lbd, penalty_gamma, capacity, weights=idcs_n_nu_uniq_freq):
                theta1_nu, theta2_nu = theta1_prox, theta2_prox
            else:
                prox_update = False
        
        train_z_nu = f_pa(train_X_, theta1_nu, theta2_nu)
        train_cost_nu, _ = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, train_z_nu, train_Y)
        if train_cost_nu < min_train_cost:
            min_train_cost = train_cost_nu
            theta1_output, theta2_output = theta1_nu, theta2_nu
        
        exp_text = f'\n - Iter.{nu+1}: n={n_nu} eps={eps_nu:d} train_cost={train_cost_nu:.4f} obj={obj_nu:.4f} prox_update={prox_update} min_train_cost={min_train_cost:.4f}'
        write_to_file(output_log_addr, exp_text)
    
    val_z = f_pa(val_X_, theta1_output, theta2_output)
    val_cost, val_feasfreq = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, val_z, val_Y)
    exp_time = time.time() - start_time
    return min_train_cost, val_cost, val_feasfreq, exp_time, (theta1_output, theta2_output)

def exp_nv2prod_PADR(train_X_data, train_Y_data, val_X, val_Y, test_X, test_Y, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num):
    exp_start_time = time.time()

    cb1, cb2, ch1, ch2 = problem_dict['c_b_1'], problem_dict['c_b_2'], problem_dict['c_h_1'], problem_dict['c_h_2']
    capacity = problem_dict['fixed_capacity']
    DIM_X = problem_dict['x_dim']

    train_X_data_ = np.hstack((train_X_data, np.ones((train_X_data.shape[0], 1)))) # adding intercept for padr
    val_X_ = np.hstack((val_X, np.ones((val_X.shape[0], 1)))) 
    test_X_ = np.hstack((test_X, np.ones((test_X.shape[0], 1))))

    outerLoopParams_dict, outerNumTotal = load_loop_params(problem_dict, 'training_sample_size')
    for idx_outerLoop in range(outerNumTotal):
        n, = pick_loop_params(idx_outerLoop, outerNumTotal, outerLoopParams_dict)
        assert n > 0
        if n > train_X_data_.shape[0]:
            print(f'Warning: n={n} is larger than the training sample size. Set n to the training data size ({train_X_data_.shape[0]}).')
            n = train_X_data_.shape[0]
        # assert n <= train_X_data_.shape[0]
        train_X_, train_Y = train_X_data_[:n], train_Y_data[:n]
        exp_text = f'\n# OuterLoop{idx_outerLoop}: n={n}'
        write_to_file(output_log_addr, exp_text)

        innerLoopParams_dict, innerNumTotal = load_loop_params(param_dict, *param_dict.keys())
        val_costs, theta_tuples = [], []
        for idx_innerLoop in range(innerNumTotal):
            (K1, K2), EPSILON, (SHRINK_EPSILON, SHRINK_RATIO), ROUND, (ITERATION, ALPHA, BETA, n0), MU, ETA, SAMPLING, penalty_lbd, penalty_gamma = \
                pick_loop_params(idx_innerLoop, innerNumTotal, innerLoopParams_dict)
            exp_text = f'\n## InnerLoop{idx_innerLoop}: K1K2=({K1},{K2}), eps={EPSILON}, shrink_eps_ratio=({SHRINK_EPSILON},{SHRINK_RATIO}), round={ROUND}, iter_a_b_N0=({ITERATION},{ALPHA},{BETA},{n0}), mu={MU}, eta={ETA}, sampling={SAMPLING}, penalty_lbd={penalty_lbd}, penalty_gamma={penalty_gamma}'
            write_to_file(output_log_addr, exp_text)

            train_cost_log, val_cost_log, val_feasfreq_log, total_time, theta_tuple_log = 1e10, 1e10, 0, 0, None
            for idx_round in range(ROUND):
                theta1_init = None if K1==0 else np.random.uniform(-MU, MU, (2, K1, DIM_X+1))
                theta2_init = None if K2==0 else np.random.uniform(-MU, MU, (2, K2, DIM_X+1))

                min_train_cost_round, val_cost_round, val_feasfreq_round, time_round, (theta1_output, theta2_output) = solve_esmm_erm_nv2prod_padr(
                    train_X_, train_Y, val_X_, val_Y, theta1_init, theta2_init, MU, ETA,
                    ITERATION, ALPHA, BETA, n0, 
                    EPSILON, SHRINK_EPSILON, SHRINK_RATIO, SAMPLING, penalty_lbd, penalty_gamma,
                    DIM_X, K1, K2, cb1, cb2, ch1, ch2, capacity,
                    output_log_addr)
                
                total_time += time_round
                if min_train_cost_round < train_cost_log:
                    train_cost_log, val_cost_log, val_feasfreq_log = min_train_cost_round, val_cost_round, val_feasfreq_round
                    theta_tuple_log = (theta1_output, theta2_output)

                exp_text = f'\n### Round{idx_round+1}: train_cost={min_train_cost_round:.4f}, val_cost={val_cost_round:.4f}, val_feasfreq={val_feasfreq_round:.4f}, time={time_round:.2f}s\n'
                write_to_file(output_log_addr, exp_text)

                update_progress(
                    (idx_outerLoop*(innerNumTotal*ROUND) + idx_innerLoop*ROUND + idx_round+1)/(outerNumTotal*innerNumTotal*ROUND),
                    exp_start_time, 'PADR', curr_exp_idx, total_exp_num)
            
            val_costs.append(val_cost_log)
            theta_tuples.append(theta_tuple_log)
            exp_text = f'\nn={n:d} param_idx={idx_innerLoop+1:d}/{innerNumTotal:d} val_cost={val_cost_log:.4f} val_feasfreq={val_feasfreq_log:.4f} time={total_time:.2f} train_cost={train_cost_log:.4f}'
            write_to_file(output_log_addr, exp_text)
        
        min_val_cost, min_val_cost_idx = np.min(val_costs), np.argmin(val_costs)
        theta1_final, theta2_final = theta_tuples[min_val_cost_idx]
        test_z_final = f_pa(test_X_, theta1_final, theta2_final)
        test_cost, test_feasfreq = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, test_z_final, test_Y)
        exp_text = f'\nOuterLoop{idx_outerLoop} (n={n}) with innerLoop{min_val_cost_idx+1}: val_cost={min_val_cost:.4f} test_cost={test_cost:.4f} test_feasfreq={test_feasfreq:.4f}\n\n'
        write_to_file(output_log_addr, exp_text)

def solve_esmm_erm_ncvxconstr_nv2prod_padr(
        train_X_, train_Y, val_X_, val_Y, theta1_init, theta2_init, mu, eta,
        iteration, alpha, beta, n0, 
        epsilon, shrink_epsilon, shrink_ratio, sampling, penalty_lbd, penalty_gamma,
        dim_x, K1, K2, cb1, cb2, ch1, ch2, capacity, ncvx_constraint_arr,
        output_log_addr):

    start_time = time.time()
    n = train_X_.shape[0]
    dim_node = train_Y.shape[1] # num of products

    theta1_nu, theta2_nu = theta1_init, theta2_init
    theta1_output, theta2_output = theta1_nu, theta2_nu

    train_z_output = f_pa(train_X_, theta1_output, theta2_output)
    min_train_cost, _ = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, train_z_output, train_Y, constr_arr=ncvx_constraint_arr)
    min_train_obj = compute_trainobj_2prod(train_z_output, train_Y, [cb1, cb2], [ch1, ch2], penalty_lbd, penalty_gamma, capacity, constr_arr=ncvx_constraint_arr)

    for nu in range(iteration):
        # sampling
        if sampling:
            n_nu = sampling_size_strategy(nu, alpha, beta, n0)
            idcs_n_nu_uniq, idcs_n_nu_uniq_freq = np.unique(
                np.random.choice(n, n_nu, replace=True), return_counts=True)
            idcs_n_nu_uniq_freq = idcs_n_nu_uniq_freq / n_nu
            n_nu_uniq = idcs_n_nu_uniq.shape[0]
        else:
            n_nu = n
            idcs_n_nu_uniq, idcs_n_nu_uniq_freq = np.arange(n), np.ones(n) / n
            n_nu_uniq = n
        
        X_nu_, Y_nu = train_X_[idcs_n_nu_uniq, :], train_Y[idcs_n_nu_uniq, :]

        # epsilon-active indices
        eps_nu = get_epsilon_nu(nu, epsilon, shrink_epsilon, shrink_ratio, iteration)
        # randomly select eps-active indices (I1, I2) for two decisions, where both I1 and I2 are of (2,n)
        mathcalI_nu_arr_tup = (
            random_epsilon_active_indices_combinations(theta1_nu[0], theta2_nu[0], X_nu_, eps_nu),
            random_epsilon_active_indices_combinations(theta1_nu[1], theta2_nu[1], X_nu_, eps_nu))
        p1I_nu, p1J_nu = mathcalI_nu_arr_tup[0][0], mathcalI_nu_arr_tup[0][1]
        p2I_nu, p2J_nu = mathcalI_nu_arr_tup[1][0], mathcalI_nu_arr_tup[1][1]

        z_nu = f_pa(X_nu_, theta1_nu, theta2_nu)
        z1_nu_ = np.hstack((z_nu[:,0].reshape((-1, 1)), np.ones((z_nu.shape[0], 1))))
        z2_nu_ = np.hstack((z_nu[:,1].reshape((-1, 1)), np.ones((z_nu.shape[0], 1))))
        p1CJ_nu, p2CJ_nu = A_eps_ma(ncvx_constraint_arr, z1_nu_, 0), A_eps_ma(ncvx_constraint_arr, z2_nu_, 0)
        p1CJcomb_num_for_sample = preprocessing_combinations(p1CJ_nu)
        p2CJcomb_num_for_sample = preprocessing_combinations(p2CJ_nu)
        C1J_nu, C2J_nu = IJ(p1CJ_nu, p1CJcomb_num_for_sample), IJ(p2CJ_nu, p2CJcomb_num_for_sample)
    
        m = grb.Model('esmm_padr_ncvxconstr_nv2prod_subprob')
        m.Params.LogToConsole = 0
        if K1 > 0:
            theta1_var = m.addMVar((dim_node, K1, dim_x+1), lb=-mu, ub=mu, name='theta1_var')
        if K2 > 0:
            theta2_var = m.addMVar((dim_node, K2, dim_x+1), lb=-mu, ub=mu, name='theta2_var')
        vB = m.addMVar((dim_node, n_nu_uniq), lb=0, name='arti-var for back') # auxiliary variables
        vH = m.addMVar((dim_node, n_nu_uniq), lb=0, name='arti-var for hold')
        vpenalty = m.addMVar(n_nu_uniq, lb=0, name='arti-var for penalty')
        vpenalty_node = m.addMVar((dim_node, n_nu_uniq), lb=0, name='arti-var for penalty of two products')

        # construct constraints for penalization
        p1_cost_param_nu, p2_cost_param_nu = ncvx_constraint_arr[C1J_nu, :], ncvx_constraint_arr[C2J_nu, :] # shape: (N_nu_uniq, 2)
        p1_X_nu_cve_ = X_nu_ * np.tile(-p1_cost_param_nu[:, 0].reshape((-1,1)), X_nu_.shape[1])
        p2_X_nu_cve_ = X_nu_ * np.tile(-p2_cost_param_nu[:, 0].reshape((-1,1)), X_nu_.shape[1])
        # # generate active J-idcs pairs and corresponding sample idcs groups, then change the relative sample position of X and param
        X_nu_pen_p1, X_nu_pen_p2 = np.zeros_like(X_nu_), np.zeros_like(X_nu_)
        cost_param0_p1, cost_param0_p2 = np.zeros(n_nu_uniq), np.zeros(n_nu_uniq)
        active_K2_combs, active_sampleaxis_pos, active_sampleaxis_counts = [], [], []
        J_stack = np.hstack((p1J_nu.reshape((-1, 1)), p2J_nu.reshape((-1,1))))
        pos = 0
        for p1k2 in range(K2):
            for p2k2 in range(K2):
                chosen_idcs = np.all(J_stack[:]==np.array([p1k2, p2k2]), axis=1)
                num_p1k2p2k2activesamples = np.sum(chosen_idcs)
                if num_p1k2p2k2activesamples==0:
                    continue
                X_nu_pen_p1[pos:pos+num_p1k2p2k2activesamples] = p1_X_nu_cve_[chosen_idcs,:]
                X_nu_pen_p2[pos:pos+num_p1k2p2k2activesamples] = p2_X_nu_cve_[chosen_idcs,:]
                cost_param0_p1[pos:pos+num_p1k2p2k2activesamples] = p1_cost_param_nu[chosen_idcs, 1]
                cost_param0_p2[pos:pos+num_p1k2p2k2activesamples] = p2_cost_param_nu[chosen_idcs, 1]

                active_K2_combs.append([p1k2, p2k2])
                active_sampleaxis_pos.append(pos)
                active_sampleaxis_counts.append(num_p1k2p2k2activesamples)
                pos += num_p1k2p2k2activesamples
        for idx_K2comb in range(len(active_K2_combs)):
            p1k2, p2k2 = active_K2_combs[idx_K2comb][0], active_K2_combs[idx_K2comb][1]
            pos_start = active_sampleaxis_pos[idx_K2comb]
            pos_end = pos_start + active_sampleaxis_counts[idx_K2comb]
            for p1k1 in range(K1):
                for p2k1 in range(K1):
                    m.addConstr(
                        vpenalty[pos_start:pos_end] + capacity - penalty_gamma >= \
                            X_nu_pen_p1[pos_start:pos_end,:] @ theta1_var[0, p1k1,:] - X_nu_pen_p1[pos_start:pos_end,:] @ theta2_var[0, p1k2,:] + \
                            X_nu_pen_p2[pos_start:pos_end,:] @ theta1_var[1, p2k1,:] - X_nu_pen_p2[pos_start:pos_end,:] @ theta2_var[1, p2k2,:]
                            - cost_param0_p1[pos_start:pos_end] - cost_param0_p2[pos_start:pos_end]
                    )

        # constraints for nv objective components
        def generate_sorted_components(X_nu, Y_nu, I_nu,):
            I_nu_argsort = np.argsort(I_nu, kind='mergesort')
            X_nu_I, Y_nu_I = X_nu[I_nu_argsort], Y_nu[I_nu_argsort]
            I_nu_sorted = I_nu[I_nu_argsort]
            I_nu_sorteduniq, I_nu_sorteduniq_pos, I_nu_sorteduniq_counts = np.unique(I_nu_sorted, return_index=True, return_counts=True)
            return I_nu_sorteduniq, I_nu_sorteduniq_pos, I_nu_sorteduniq_counts, X_nu_I, Y_nu_I


        p1I_nu_sorteduniq, p1I_nu_sorteduniq_pos, p1I_nu_sorteduniq_counts, p1X_nu_I, p1Y_nu_I = generate_sorted_components(X_nu_, Y_nu[:,0], mathcalI_nu_arr_tup[0][0])
        p2I_nu_sorteduniq, p2I_nu_sorteduniq_pos, p2I_nu_sorteduniq_counts, p2X_nu_I, p2Y_nu_I = generate_sorted_components(X_nu_, Y_nu[:,1], mathcalI_nu_arr_tup[1][0])

        def add_nvconstr_max1(m, vB, vpenp, vartheta1, vartheta2, X_nu_remake, Y_nu_remake, I_nu_sorteduniq, I_nu_sorteduniq_pos, I_nu_sorteduniq_counts, K2):
            for idx_K1 in range(I_nu_sorteduniq.shape[0]):
                k1 = I_nu_sorteduniq[idx_K1]
                for k2 in range(K2):
                    pos_start = I_nu_sorteduniq_pos[idx_K1]
                    pos_end = pos_start + I_nu_sorteduniq_counts[idx_K1]
                    m.addConstr(
                        vB[pos_start:pos_end] - Y_nu_remake[pos_start:pos_end] >= \
                            X_nu_remake[pos_start:pos_end, :] @ vartheta2[k2, :] - \
                            X_nu_remake[pos_start:pos_end, :] @ vartheta1[k1, :])
                    m.addConstr(
                        vpenp[pos_start:pos_end] >= X_nu_remake[pos_start:pos_end, :] @ vartheta2[k2, :] - X_nu_remake[pos_start:pos_end, :] @ vartheta1[k1, :])
        add_nvconstr_max1(
            m, vB[0], vpenalty_node[0], theta1_var[0], theta2_var[0], p1X_nu_I, p1Y_nu_I, p1I_nu_sorteduniq, p1I_nu_sorteduniq_pos, p1I_nu_sorteduniq_counts, K2)
        add_nvconstr_max1(
            m, vB[1], vpenalty_node[1], theta1_var[1], theta2_var[1], p2X_nu_I, p2Y_nu_I, p2I_nu_sorteduniq, p2I_nu_sorteduniq_pos, p2I_nu_sorteduniq_counts, K2)
        
        p1J_nu_sorteduniq, p1J_nu_sorteduniq_pos, p1J_nu_sorteduniq_counts, p1X_nu_J, p1Y_nu_J = generate_sorted_components(X_nu_, Y_nu[:,0], mathcalI_nu_arr_tup[0][1])
        p2J_nu_sorteduniq, p2J_nu_sorteduniq_pos, p2J_nu_sorteduniq_counts, p2X_nu_J, p2Y_nu_J = generate_sorted_components(X_nu_, Y_nu[:,1], mathcalI_nu_arr_tup[1][1])

        def add_nvconstr_max2(m, vH, vartheta1, vartheta2, X_nu_remake, Y_nu_remake, J_nu_sorteduniq, J_nu_sorteduniq_pos, J_nu_sorteduniq_counts, K1):
            for idx_K2 in range(J_nu_sorteduniq.shape[0]):
                k2 = J_nu_sorteduniq[idx_K2]
                for k1 in range(K1):
                    pos_start = J_nu_sorteduniq_pos[idx_K2]
                    pos_end = pos_start + J_nu_sorteduniq_counts[idx_K2]
                    m.addConstr(
                        vH[pos_start:pos_end] + Y_nu_remake[pos_start:pos_end] >= \
                            X_nu_remake[pos_start:pos_end, :] @ vartheta1[k1, :] - \
                            X_nu_remake[pos_start:pos_end, :] @ vartheta2[k2, :]
                    )
        add_nvconstr_max2(
            m, vH[0], theta1_var[0], theta2_var[0], p1X_nu_J, p1Y_nu_J, p1J_nu_sorteduniq, p1J_nu_sorteduniq_pos, p1J_nu_sorteduniq_counts, K1)
        add_nvconstr_max2(
            m, vH[1], theta1_var[1], theta2_var[1], p2X_nu_J, p2Y_nu_J, p2J_nu_sorteduniq, p2J_nu_sorteduniq_pos, p2J_nu_sorteduniq_counts, K1)        

        if eta > 0:
            m.setObjective((idcs_n_nu_uniq_freq * cb1) @ vB[0] + (idcs_n_nu_uniq_freq * ch1) @ vH[0] + \
                (idcs_n_nu_uniq_freq * cb2) @ vB[1] + (idcs_n_nu_uniq_freq * ch2) @ vH[1] + \
                (idcs_n_nu_uniq_freq * penalty_lbd) @ vpenalty + (idcs_n_nu_uniq_freq * penalty_lbd) @ vpenalty_node[0] + (idcs_n_nu_uniq_freq * penalty_lbd) @ vpenalty_node[1] + \
                eta/2 * (
                sum(theta1_var[0, k1,:] @ theta1_var[0, k1, :] - 2 * theta1_nu[0, k1, :] @ theta1_var[0, k1, :] for k1 in range(K1))
                + sum(theta2_var[0, k2,:] @ theta2_var[0, k2, :] - 2 * theta2_nu[0, k2, :] @ theta2_var[0, k2, :] for k2 in range(K2))
                + sum(theta1_var[1, k1,:] @ theta1_var[1, k1, :] - 2 * theta1_nu[1, k1, :] @ theta1_var[1, k1, :] for k1 in range(K1))
                + sum(theta2_var[1, k2,:] @ theta2_var[1, k2, :] - 2 * theta2_nu[1, k2, :] @ theta2_var[1, k2, :] for k2 in range(K2))))
        else:
            m.setObjective((idcs_n_nu_uniq_freq * cb1) @ vB[0] + (idcs_n_nu_uniq_freq * ch1) @ vH[0] + \
                (idcs_n_nu_uniq_freq * cb2) @ vB[1] + (idcs_n_nu_uniq_freq * ch2) @ vH[1] + \
                (idcs_n_nu_uniq_freq * penalty_lbd) @ vpenalty + (idcs_n_nu_uniq_freq * penalty_lbd) @ vpenalty_node[0] + (idcs_n_nu_uniq_freq * penalty_lbd) @ vpenalty_node[1])
        
        m.optimize()            
        obj_nu = m.ObjVal
        theta1_prox = theta1_var.X if K1 > 0 else None
        theta2_prox = theta2_var.X if K2 > 0 else None
        # validation step and update theta_nu
        prox_update = True
        if eps_nu == 0:
            theta1_nu, theta2_nu = theta1_prox, theta2_prox
        else:
            train_z_nu = f_pa(X_nu_, theta1_nu, theta2_nu)
            if obj_nu <= compute_trainobj_2prod(train_z_nu, Y_nu, [cb1, cb2], [ch1, ch2], penalty_lbd, penalty_gamma, capacity, weights=idcs_n_nu_uniq_freq, constr_arr=ncvx_constraint_arr):
                theta1_nu, theta2_nu = theta1_prox, theta2_prox
            else:
                prox_update = False

        train_z_nu = f_pa(train_X_, theta1_nu, theta2_nu)
        train_cost_nu, _ = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, train_z_nu, train_Y, constr_arr=ncvx_constraint_arr)
        train_obj_nu = compute_trainobj_2prod(train_z_nu, train_Y, [cb1, cb2], [ch1, ch2], penalty_lbd, penalty_gamma, capacity, constr_arr=ncvx_constraint_arr)
        if train_obj_nu < min_train_obj:
            min_train_obj = train_obj_nu
            min_train_cost, _ = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, train_z_nu, train_Y, constr_arr=ncvx_constraint_arr)
            theta1_output, theta2_output = theta1_nu, theta2_nu

        exp_text = f'\n - Iter.{nu+1}: n={n_nu} eps={eps_nu:d} train_cost={train_cost_nu:.4f} obj={train_obj_nu:.4f} prox_update={prox_update} min_train_cost={min_train_cost:.4f}'
        write_to_file(output_log_addr, exp_text)
    
    val_z = f_pa(val_X_, theta1_output, theta2_output)
    val_cost, val_feasfreq = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, val_z, val_Y, constr_arr=ncvx_constraint_arr)
    exp_time = time.time() - start_time
    return min_train_cost, val_cost, val_feasfreq, exp_time, (theta1_output, theta2_output)

def exp_ncvxconstr_nv2prod_PADR(train_X_data, train_Y_data, val_X, val_Y, test_X, test_Y, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num):
    exp_start_time = time.time()
    cb1, cb2, ch1, ch2 = problem_dict['c_b_1'], problem_dict['c_b_2'], problem_dict['c_h_1'], problem_dict['c_h_2']
    capacity = problem_dict['fixed_capacity']
    ncvx_constraint_arr = problem_dict['ncvx_constraint_arr']
    DIM_X = problem_dict['x_dim']

    train_X_data_ = np.hstack((train_X_data, np.ones((train_X_data.shape[0], 1)))) # adding intercept for padr
    val_X_ = np.hstack((val_X, np.ones((val_X.shape[0], 1)))) 
    test_X_ = np.hstack((test_X, np.ones((test_X.shape[0], 1))))

    outerLoopParams_dict, outerNumTotal = load_loop_params(problem_dict, 'training_sample_size')
    for idx_outerLoop in range(outerNumTotal):
        n, = pick_loop_params(idx_outerLoop, outerNumTotal, outerLoopParams_dict)
        assert n > 0
        if n > train_X_data_.shape[0]:
            print(f'Warning: n={n} is larger than the training sample size. Set n to the training data size ({train_X_data_.shape[0]}).')
            n = train_X_data_.shape[0]
        # assert n <= train_X_data_.shape[0]
        train_X_, train_Y = train_X_data_[:n], train_Y_data[:n]
        exp_text = f'\n# OuterLoop{idx_outerLoop}: n={n}'
        write_to_file(output_log_addr, exp_text)

        innerLoopParams_dict, innerNumTotal = load_loop_params(param_dict, *param_dict.keys())
        val_costs, feas_freqs, theta_tuples = np.zeros(innerNumTotal), np.zeros(innerNumTotal), []
        for idx_innerLoop in range(innerNumTotal):
            (K1, K2), EPSILON, (SHRINK_EPSILON, SHRINK_RATIO), ROUND, (ITERATION, ALPHA, BETA, n0), MU, ETA, SAMPLING, penalty_lbd, penalty_gamma = \
                pick_loop_params(idx_innerLoop, innerNumTotal, innerLoopParams_dict)
            exp_text = f'\n## InnerLoop{idx_innerLoop}: K1K2=({K1},{K2}), eps={EPSILON}, shrink_eps_ratio=({SHRINK_EPSILON},{SHRINK_RATIO}), round={ROUND}, iter_a_b_N0=({ITERATION},{ALPHA},{BETA},{n0}), mu={MU}, eta={ETA}, sampling={SAMPLING}, penalty_lbd={penalty_lbd}, penalty_gamma={penalty_gamma}'
            write_to_file(output_log_addr, exp_text)

            train_cost_log, val_cost_log, val_feasfreq_log, total_time, theta_tuple_log = 1e10, 1e10, 0, 0, None
            for idx_round in range(ROUND):
                theta1_init = None if K1==0 else np.random.uniform(-MU, MU, (2, K1, DIM_X+1))
                theta2_init = None if K2==0 else np.random.uniform(-MU, MU, (2, K2, DIM_X+1))

                min_train_cost_round, val_cost_round, val_feasfreq_round, time_round, (theta1_output, theta2_output) = solve_esmm_erm_ncvxconstr_nv2prod_padr(
                    train_X_, train_Y, val_X_, val_Y, theta1_init, theta2_init, MU, ETA,
                    ITERATION, ALPHA, BETA, n0, 
                    EPSILON, SHRINK_EPSILON, SHRINK_RATIO, SAMPLING, penalty_lbd, penalty_gamma,
                    DIM_X, K1, K2, cb1, cb2, ch1, ch2, capacity, ncvx_constraint_arr,
                    output_log_addr)
                
                total_time += time_round
                if min_train_cost_round < train_cost_log:
                    train_cost_log, val_cost_log, val_feasfreq_log = min_train_cost_round, val_cost_round, val_feasfreq_round
                    theta_tuple_log = (theta1_output, theta2_output)

                exp_text = f'\n### Round{idx_round+1}: train_cost={min_train_cost_round:.4f}, val_cost={val_cost_round:.4f}, val_feasfreq={val_feasfreq_round:.4f}, time={time_round:.2f}s\n'
                write_to_file(output_log_addr, exp_text)

                update_progress(
                    (idx_outerLoop*(innerNumTotal*ROUND) + idx_innerLoop*ROUND + idx_round+1)/(outerNumTotal*innerNumTotal*ROUND),
                    exp_start_time, 'PADR', curr_exp_idx, total_exp_num)
                
            val_costs[idx_innerLoop] = val_cost_log
            feas_freqs[idx_innerLoop] = val_feasfreq_log
            theta_tuples.append(theta_tuple_log)
            exp_text = f'\nn={n:d} param_idx={idx_innerLoop+1:d}/{innerNumTotal:d} val_cost={val_cost_log:.4f} val_feasfreq={val_feasfreq_log:.4f} time={total_time:.2f} train_cost={train_cost_log:.4f}'
            write_to_file(output_log_addr, exp_text)


        # min_val_cost, min_val_cost_idx = np.min(val_costs), np.argmin(val_costs)
        if np.max(feas_freqs) >= 1:
            val_costs[feas_freqs < 1] = 1e10
            min_val_cost_idx = np.argmin(val_costs)
        else:
            min_val_cost_idx = np.argmax(feas_freqs)
        min_val_cost = val_costs[min_val_cost_idx]
        theta1_final, theta2_final = theta_tuples[min_val_cost_idx]
        test_z_final = f_pa(test_X_, theta1_final, theta2_final)
        test_cost, test_feasfreq = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, test_z_final, test_Y, constr_arr=ncvx_constraint_arr)
        exp_text = f'\nOuterLoop{idx_outerLoop} (n={n}) with innerLoop{min_val_cost_idx+1}: val_cost={min_val_cost:.4f} test_cost={test_cost:.4f} test_feasfreq={test_feasfreq:.4f}\n\n'
        write_to_file(output_log_addr, exp_text)

def opt_subproblems_l2(N_nu_uniq, idcs_Nnu_uniq_freq, 
                              X_nu_, Y_nu, theta1_nu, theta2_nu, 
                              eps_nu, padr_K1, padr_K2, dim_x, mu, eta=0):
    # the following code selects eps-active index combination I and J
    if padr_K1==0:
        assert theta1_nu is None
    else:
        AI_nu = A_eps_ma(theta1_nu, X_nu_, eps_nu)
        Icomb_num_for_sample = preprocessing_combinations(AI_nu)
        I_nu = IJ(AI_nu, Icomb_num_for_sample)
    if padr_K2==0:
        assert theta2_nu is None
    else:
        AJ_nu = A_eps_ma(theta2_nu, X_nu_, eps_nu)
        Jcomb_num_for_sample = preprocessing_combinations(AJ_nu)
        J_nu = IJ(AJ_nu, Jcomb_num_for_sample)
            
    # construct and solve the optimization sub-problem (surrogation is built in constraints)
    m = grb.Model('prob_nu')
    m.Params.LogToConsole = 0  # do not output the log info
    
    if padr_K1 != 0:
        vartheta1 = m.addMVar((padr_K1, dim_x+1), lb=-mu, ub=mu, name='vartheta1')
    if padr_K2 != 0:
        vartheta2 = m.addMVar((padr_K2, dim_x+1), lb=-mu, ub=mu, name='vartheta2') 
    vB = m.addMVar(N_nu_uniq, lb=0, name='arti-var for back') # auxiliary variables
    vH = m.addMVar(N_nu_uniq, lb=0, name='arti-var for hold')

    if padr_K1 != 0:
        I_nu_argsort = np.argsort(I_nu, kind='mergesort')
        X_nu_I = X_nu_[I_nu_argsort]
        Y_nu_I = Y_nu[I_nu_argsort]
        I_nu_sort = I_nu[I_nu_argsort]
        I_nu_sort_uniq, I_nu_sort_uniq_pos, I_nu_sort_uniq_counts = \
            np.unique(I_nu_sort, return_index=True, return_counts=True) # K1 index, start position in X_nu_I, length in X_nu_I

        for idx_K1 in range(I_nu_sort_uniq.shape[0]):
            pos_start = I_nu_sort_uniq_pos[idx_K1]
            pos_end = pos_start + I_nu_sort_uniq_counts[idx_K1]
            X_nu_I_remake = X_nu_I[pos_start:pos_end, :]
            Y_nu_I_remake = Y_nu_I[pos_start:pos_end]
            if padr_K2 != 0:
                for k2 in range(padr_K2):
                    m.addConstr(
                        vB[pos_start:pos_end] - Y_nu_I_remake >= \
                            X_nu_I_remake @ vartheta2[k2, :] - X_nu_I_remake @ vartheta1[I_nu_sort_uniq[idx_K1], :])
            else:
                m.addConstr(
                    vB[pos_start:pos_end] - Y_nu_I_remake >= - X_nu_I_remake @ vartheta1[I_nu_sort_uniq[idx_K1], :])
    else:
        for k2 in range(padr_K2):
            m.addConstr(vB - Y_nu >= X_nu_ @ vartheta2[k2, :])

    if padr_K2 != 0:
        J_nu_argsort = np.argsort(J_nu, kind='mergesort')
        X_nu_J = X_nu_[J_nu_argsort]
        Y_nu_J = Y_nu[J_nu_argsort]
        J_nu_sort = J_nu[J_nu_argsort]
        J_nu_sort_uniq, J_nu_sort_uniq_pos, J_nu_sort_uniq_counts = \
            np.unique(J_nu_sort, return_index=True, return_counts=True)

        for idx_K2 in range(J_nu_sort_uniq.shape[0]):
            pos_start = J_nu_sort_uniq_pos[idx_K2]
            pos_end = pos_start + J_nu_sort_uniq_counts[idx_K2]
            X_nu_J_remake = X_nu_J[pos_start:pos_end, :]
            Y_nu_J_remake = Y_nu_J[pos_start:pos_end]
            if padr_K1 != 0:
                for k1 in range(padr_K1):
                    m.addConstr(
                        vH[pos_start:pos_end] + Y_nu_J_remake >= \
                            X_nu_J_remake @ vartheta1[k1, :] - X_nu_J_remake @ vartheta2[J_nu_sort_uniq[idx_K2], :])
            else:
                m.addConstr(
                    vH[pos_start:pos_end] + Y_nu_J_remake >= - X_nu_J_remake @ vartheta2[J_nu_sort_uniq[idx_K2], :])
    else:
        for k1 in range(padr_K1):
            m.addConstr(vH + Y_nu >= X_nu_ @ vartheta1[k1, :])

    uniq_freq_diag = np.diag(np.ones(N_nu_uniq) * idcs_Nnu_uniq_freq)
    if padr_K1 == 0:
        m.setObjective(
            (vB @ uniq_freq_diag @ vB + vH @ uniq_freq_diag @ vH)
            + eta/2 * (sum(vartheta2[k1,:] @ vartheta2[k1, :] - 2 * theta2_nu[k1, :] @ vartheta2[k1, :] for k1 in range(padr_K2))))
    elif padr_K2 == 0:
        m.setObjective(
            (vB @ uniq_freq_diag @ vB + vH @ uniq_freq_diag @ vH)
            + eta/2 * (sum(vartheta1[k1,:] @ vartheta1[k1, :] - 2 * theta1_nu[k1, :] @ vartheta1[k1, :] for k1 in range(padr_K1))))
    else:
        m.setObjective(
            (vB @ uniq_freq_diag @ vB + vH @ uniq_freq_diag @ vH)
            + eta/2 * (
                sum(vartheta1[k1,:] @ vartheta1[k1, :] - 2 * theta1_nu[k1, :] @ vartheta1[k1, :] for k1 in range(padr_K1))
                + sum(vartheta2[k2,:] @ vartheta2[k2, :] - 2 * theta2_nu[k2, :] @ vartheta2[k2, :] for k2 in range(padr_K2))
            )
        )
    m.optimize()
    obj_nu = m.ObjVal
    theta1_prox = vartheta1.X if padr_K1 > 0 else None
    theta2_prox = vartheta2.X if padr_K2 > 0 else None
    return obj_nu, theta1_prox, theta2_prox

def solve_esmm_erm_l2_padr(
    train_X, train_Y,
    theta1_init, theta2_init, mu, eta,
    iteration, alpha, beta, n0, eps, shrink_eps, shrink_ratio, sampling=True):
    
    n = train_X.shape[0] # total training data size
    dim_x = train_X.shape[1]-1

    padr_K1 = 0 if theta1_init is None else theta1_init.shape[1]
    padr_K2 = 0 if theta2_init is None else theta2_init.shape[1]
    dim_node = train_Y.shape[1]
        
    # initialization
    theta1_nu, theta2_nu = theta1_init, theta2_init # iterates
    theta1_output, theta2_output = theta1_nu, theta2_nu # the final output iterate
        
    train_pred = f_pa(train_X, theta1_output, theta2_output)
    min_loss = np.sum(np.average((train_pred - train_Y)**2, axis=0))

    for nu in range(iteration):
        eps_nu = get_epsilon_nu(nu, eps, shrink_eps, shrink_ratio, iteration)
        N_nu = sampling_size_strategy(nu, alpha, beta, n0)
        if sampling:
            idcs_Nnu = np.random.randint(0, n, N_nu) # i.i.d. draw N_nu samples \xi^s, s \in [N_nu]
        else:
            N_nu = n
            idcs_Nnu = np.arange(0, N_nu, 1) # only for LDR
        idcs_Nnu_uniq, idcs_Nnu_uniq_freq = np.unique(idcs_Nnu, return_counts=True) # calculate frequency and unique the sample index (_uniq is sorted)
        idcs_Nnu_uniq_freq = idcs_Nnu_uniq_freq / N_nu # turn counts into freq
        N_nu_uniq = idcs_Nnu_uniq.shape[0] # uniqued sample size
        X_nu_, Y_nu = train_X[idcs_Nnu_uniq, :], train_Y[idcs_Nnu_uniq] # corresponding samples used in this iteration
        
        loss_nu = 0
        theta1_prox = theta1_nu.copy() if theta1_nu is not None else None
        theta2_prox = theta2_nu.copy() if theta2_nu is not None else None
        for idx_node in range(dim_node):
            theta1_nu_node = theta1_nu[idx_node, :, :] if padr_K1 > 0 else None
            theta2_nu_node = theta2_nu[idx_node, :, :] if padr_K2 > 0 else None

            loss_nu_node, theta1_prox_node, theta2_prox_node = opt_subproblems_l2(
                N_nu_uniq, idcs_Nnu_uniq_freq, X_nu_, Y_nu[:, idx_node], theta1_nu_node, theta2_nu_node, 
                eps_nu, padr_K1, padr_K2, dim_x, mu, eta)
            
            loss_nu += loss_nu_node
            if theta1_nu is not None:
                theta1_prox[idx_node, :, :] = theta1_prox_node
            if theta2_nu is not None:
                theta2_prox[idx_node, :, :] = theta2_prox_node

        if eps_nu == 0:
            theta1_nu, theta2_nu = theta1_prox, theta2_prox
        else:
            pred_nu = f_pa(X_nu_, theta1_nu, theta2_nu)
            if loss_nu <= np.sum(np.average((pred_nu - Y_nu)**2, weights=idcs_Nnu_uniq_freq, axis=0)): # update the proximal mapping point
                theta1_nu, theta2_nu = theta1_prox, theta2_prox
            else:
                theta1_nu, theta2_nu = theta1_nu, theta2_nu

        # update the final output and print info
        curr_pred = f_pa(train_X, theta1_nu, theta2_nu)
        curr_loss = np.sum(np.average((curr_pred - train_Y)**2, axis=0))
        
        # min cost update
        if curr_loss < min_loss: # choose the output theta (with the min train loss)
            theta1_output, theta2_output = theta1_nu, theta2_nu
            min_loss = curr_loss

    return theta1_output, theta2_output, min_loss

def exp_nv_PO(train_X_data, train_Y_data, val_X, val_Y, test_X, test_Y, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num, name='PO-PA', include_val=False):
    exp_start_time = time.time()
    cb, ch = problem_dict['c_b'], problem_dict['c_h']
    DIM_X = problem_dict['x_dim']
    DIM_NODE = 1

    train_X_data_ = np.hstack((train_X_data, np.ones((train_X_data.shape[0], 1)))) # adding intercept for padr
    val_X_ = np.hstack((val_X, np.ones((val_X.shape[0], 1))))
    test_X_ = np.hstack((test_X, np.ones((test_X.shape[0], 1))))

    outerLoopParams_dict, outerNumTotal = load_loop_params(problem_dict, 'training_sample_size')
    for idx_outerLoop in range(outerNumTotal):
        n, = pick_loop_params(idx_outerLoop, outerNumTotal, outerLoopParams_dict)
        assert n > 0
        if n > train_X_data_.shape[0]:
            print(f'Warning: n={n} is larger than the training sample size. Set n to the training data size ({train_X_data_.shape[0]}).')
            n = train_X_data_.shape[0]
        # assert n <= train_X_data_.shape[0]
        train_X_, train_Y = train_X_data_[:n], train_Y_data[:n]
        exp_text = f'\n# OuterLoop{idx_outerLoop}: n={n}'
        write_to_file(output_log_addr, exp_text)

        innerLoopParams_dict, innerNumTotal = load_loop_params(param_dict, *param_dict.keys())
        val_costs, theta_tuples = [], []
        for idx_innerLoop in range(innerNumTotal):
            round_start_time = time.time()
            (K1, K2), EPSILON, (SHRINK_EPSILON, SHRINK_RATIO), ROUND, (ITERATION, ALPHA, BETA, n0), MU, ETA, SAMPLING = \
                pick_loop_params(idx_innerLoop, innerNumTotal, innerLoopParams_dict)
            exp_text = f'\n## InnerLoop{idx_innerLoop}: K1K2=({K1},{K2}), eps={EPSILON}, shrink_eps_ratio=({SHRINK_EPSILON},{SHRINK_RATIO}), round={ROUND}, iter_a_b_N0=({ITERATION},{ALPHA},{BETA},{n0}), mu={MU}, eta={ETA}, sampling={SAMPLING}'
            write_to_file(output_log_addr, exp_text)

            min_pred_loss_loop = 1e10
            theta1_loop, theta2_loop = None, None
            for idx_round in range(ROUND):
                theta1_init = None if K1==0 else np.random.uniform(-MU/2, MU/2, (DIM_NODE, K1, DIM_X+1))
                theta2_init = None if K2==0 else np.random.uniform(-MU/2, MU/2, (DIM_NODE, K2, DIM_X+1))

                theta1_round, theta2_round, loss_round = solve_esmm_erm_l2_padr(
                    train_X_, train_Y.reshape((-1, 1)), theta1_init, theta2_init,
                    MU, ETA, ITERATION, ALPHA, BETA, n0, EPSILON, SHRINK_EPSILON, SHRINK_RATIO, SAMPLING)
                
                if loss_round <= min_pred_loss_loop:
                    min_pred_loss_loop = loss_round
                    theta1_loop, theta2_loop = theta1_round, theta2_round
                
                exp_text = f'\n### Round{idx_round+1}: prediction loss={loss_round:.4f}'
                write_to_file(output_log_addr, exp_text)

                update_progress(
                    (idx_outerLoop*(innerNumTotal*ROUND) + idx_innerLoop*ROUND + idx_round+1)/(outerNumTotal*innerNumTotal*ROUND),
                    exp_start_time, name, curr_exp_idx, total_exp_num)
            
            train_z, val_z = f_pa(train_X_, theta1_loop, theta2_loop), f_pa(val_X_, theta1_loop, theta2_loop)
            train_z = train_z[:,0] if train_z.shape[1] == 1 else train_z
            val_z = val_z[:,0] if val_z.shape[1] == 1 else val_z

            train_cost = nv_cost(cb, ch, train_z, train_Y) + capacost_function_oneprod(train_z, problem_dict['ncvx_capacity'])
            val_cost = nv_cost(cb, ch, val_z, val_Y) + capacost_function_oneprod(val_z, problem_dict['ncvx_capacity'])
            val_costs.append(val_cost)
            theta_tuples.append((theta1_loop, theta2_loop))
            exp_text = f'\nn={n:d} param_idx={idx_innerLoop+1:d}/{innerNumTotal:d} val_cost={val_cost:.4f} time={time.time()-round_start_time:.2f} train_cost={train_cost:.4f} prediction_loss={min_pred_loss_loop:.4f}'
            write_to_file(output_log_addr, exp_text)
        
        min_val_cost, min_val_cost_idx = np.min(val_costs), np.argmin(val_costs)
        theta1_final, theta2_final = theta_tuples[min_val_cost_idx]
        if include_val:
            (K1, K2), EPSILON, (SHRINK_EPSILON, SHRINK_RATIO), ROUND, (ITERATION, ALPHA, BETA, n0), MU, ETA, SAMPLING = \
                pick_loop_params(min_val_cost_idx, innerNumTotal, innerLoopParams_dict)
            exp_text = f'\n## Testing Stage: Using params of innerLoop{idx_innerLoop}: K1K2=({K1},{K2}), eps={EPSILON}, shrink_eps_ratio=({SHRINK_EPSILON},{SHRINK_RATIO}), round={ROUND}, iter_a_b_N0=({ITERATION},{ALPHA},{BETA},{n0}), mu={MU}, eta={ETA}, sampling={SAMPLING}'
            write_to_file(output_log_addr, exp_text)
            train_X_final_ = np.vstack((train_X_, val_X_))
            train_Y_final = np.vstack((train_Y, val_Y)) if len(train_Y.shape)>1 else np.hstack((train_Y, val_Y))

            min_pred_loss_loop = 1e10
            theta1_final, theta2_final = None, None
            for idx_round in range(ROUND):
                theta1_init = None if K1==0 else np.random.uniform(-MU/2, MU/2, (DIM_NODE, K1, DIM_X+1))
                theta2_init = None if K2==0 else np.random.uniform(-MU/2, MU/2, (DIM_NODE, K2, DIM_X+1))

                theta1_round, theta2_round, loss_round = solve_esmm_erm_l2_padr(
                    train_X_final_, train_Y_final.reshape((-1, 1)), theta1_init, theta2_init,
                    MU, ETA, ITERATION, ALPHA, BETA, n0, EPSILON, SHRINK_EPSILON, SHRINK_RATIO, SAMPLING)
                
                if loss_round <= min_pred_loss_loop:
                    min_pred_loss_loop = loss_round
                    theta1_final, theta2_final = theta1_round, theta2_round
                
                exp_text = f'\n### Round{idx_round+1}: prediction loss={loss_round:.4f}'
                write_to_file(output_log_addr, exp_text)

        test_z_final = f_pa(test_X_, theta1_final, theta2_final)
        test_z_final = test_z_final[:,0] if test_z_final.shape[1] == 1 else test_z_final
        test_cost = nv_cost(cb, ch, test_z_final, test_Y) + capacost_function_oneprod(test_z_final, problem_dict['ncvx_capacity'])
        exp_text = f'\nOuterLoop{idx_outerLoop} (n={n}) with innerLoop{min_val_cost_idx+1}: val_cost={min_val_cost:.4f} test_cost={test_cost:.4f}\n\n'
        write_to_file(output_log_addr, exp_text)


def exp_pp_PO(train_X_data, train_Y_data, val_X, val_Y, test_X, test_Y, problem_dict, param_dict, output_log_addr, curr_exp_idx, total_exp_num, name='PO-PA'):
    exp_start_time = time.time()

    c, g, b, h, W = problem_dict['c'], problem_dict['g'], problem_dict['b'], problem_dict['h'], problem_dict['W']
    DIM_X = problem_dict['x_dim']
    DIM_NODE, DIM_ARC = c.shape[0], g.shape[0]

    train_X_data_ = np.hstack((train_X_data, np.ones((train_X_data.shape[0], 1)))) # adding intercept for padr
    val_X_ = np.hstack((val_X, np.ones((val_X.shape[0], 1))))
    test_X_ = np.hstack((test_X, np.ones((test_X.shape[0], 1))))

    outerLoopParams_dict, outerNumTotal = load_loop_params(problem_dict, 'training_sample_size')
    for idx_outerLoop in range(outerNumTotal):
        n, = pick_loop_params(idx_outerLoop, outerNumTotal, outerLoopParams_dict)
        assert n > 0
        if n > train_X_data_.shape[0]:
            print(f'Warning: n={n} is larger than the training sample size. Set n to the training data size ({train_X_data_.shape[0]}).')
            n = train_X_data_.shape[0]
        # assert n <= train_X_data_.shape[0]
        train_X_, train_Y = train_X_data_[:n], train_Y_data[:n]
        exp_text = f'\n# OuterLoop{idx_outerLoop}: n={n}'
        write_to_file(output_log_addr, exp_text)

        innerLoopParams_dict, innerNumTotal = load_loop_params(param_dict, *param_dict.keys())
        val_costs, theta_tuples = [], []
        for idx_innerLoop in range(innerNumTotal):
            round_start_time = time.time()
            (K1, K2), EPSILON, (SHRINK_EPSILON, SHRINK_RATIO), ROUND, (ITERATION, ALPHA, BETA, n0), MU, ETA, SAMPLING = \
                pick_loop_params(idx_innerLoop, innerNumTotal, innerLoopParams_dict)
            exp_text = f'\n## InnerLoop{idx_innerLoop}: K1K2=({K1},{K2}), eps={EPSILON}, shrink_eps_ratio=({SHRINK_EPSILON},{SHRINK_RATIO}), round={ROUND}, iter_a_b_N0=({ITERATION},{ALPHA},{BETA},{n0}), mu={MU}, eta={ETA}, sampling={SAMPLING}'
            write_to_file(output_log_addr, exp_text)

            min_pred_loss_loop = 1e10
            theta1_loop, theta2_loop = None, None
            for idx_round in range(ROUND):
                theta1_init = None if K1==0 else np.random.uniform(-MU/2, MU/2, (DIM_NODE, K1, DIM_X+1))
                theta2_init = None if K2==0 else np.random.uniform(-MU/2, MU/2, (DIM_NODE, K2, DIM_X+1))

                theta1_round, theta2_round, loss_round = solve_esmm_erm_l2_padr(
                    train_X_, train_Y, theta1_init, theta2_init,
                    MU, ETA, ITERATION, ALPHA, BETA, n0, EPSILON, SHRINK_EPSILON, SHRINK_RATIO, SAMPLING)

                if loss_round <= min_pred_loss_loop:
                    min_pred_loss_loop = loss_round
                    theta1_loop, theta2_loop = theta1_round, theta2_round
            
                exp_text = f'\n### Round{idx_round+1}: prediction loss={loss_round:.4f}'
                write_to_file(output_log_addr, exp_text)

                update_progress(
                    (idx_outerLoop*(innerNumTotal*ROUND) + idx_innerLoop*ROUND + idx_round+1)/(outerNumTotal*innerNumTotal*ROUND),
                    exp_start_time, name, curr_exp_idx, total_exp_num)
            
            train_order, val_order = f_pa(train_X_, theta1_loop, theta2_loop), f_pa(val_X_, theta1_loop, theta2_loop)
            train_cost = pp_cost(c, g, b, h, W, train_order, train_Y)
            val_cost = pp_cost(c, g, b, h, W, val_order, val_Y)
            val_costs.append(val_cost)
            theta_tuples.append((theta1_loop, theta2_loop))
            exp_text = f'\nn={n:d} param_idx={idx_innerLoop+1:d}/{innerNumTotal:d} val_cost={val_cost:.4f} time={time.time()-round_start_time:.2f} train_cost={train_cost:.4f} prediction_loss={min_pred_loss_loop:.4f}'
            write_to_file(output_log_addr, exp_text)

        min_val_cost, min_val_cost_idx = np.min(val_costs), np.argmin(val_costs)
        theta1_final, theta2_final = theta_tuples[min_val_cost_idx]
        test_order_final = f_pa(test_X_, theta1_final, theta2_final)
        test_cost = pp_cost(c, g, b, h, W, test_order_final, test_Y)
        exp_text = f'\nOuterLoop{idx_outerLoop} (n={n}) with innerLoop{min_val_cost_idx+1}: val_cost={min_val_cost:.4f} test_cost={test_cost:.4f}\n\n'
        write_to_file(output_log_addr, exp_text)