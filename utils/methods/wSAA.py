from utils.tools import *
from utils.tools_nv import *
from utils.tools_pp import *
from utils.methods.StochOptForest_tree import *
from utils.methods.StochOptForest_nv_tree_utilities import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

def exp_nv_wSAA(train_X_data, train_Y_data, val_X, val_Y, test_X, test_Y, problem_dict, param_dict, output_log_addr, 
             curr_exp_idx, total_exp_num, include_val=False):
    exp_start_time = time.time()

    cb, ch = problem_dict['c_b'], problem_dict['c_h']

    outerLoopParams_dict, outerNumTotal = load_loop_params(problem_dict, 'training_sample_size')
    for idx_outerLoop in range(outerNumTotal):
        n, = pick_loop_params(idx_outerLoop, outerNumTotal, outerLoopParams_dict)
        assert n > 0
        if n > train_X_data.shape[0]:
            print(f'Warning: n={n} is larger than the training sample size. Set n to the training data size ({train_X_data.shape[0]}).')
            n = train_X_data.shape[0]
        # assert n <= train_X_data.shape[0]
        train_X, train_Y = train_X_data[:n], train_Y_data[:n]

        if problem_dict['ncvx_capacity'] is None:
            grbModel, grbz, grbvB, grbvH = build_nv_grb_model(train_Y)
        innerLoopParams_dict, innerNumTotal = load_loop_params(param_dict, *param_dict.keys())

        val_costs = []
        for idx_innerLoop in range(innerNumTotal):
            method_param_dict = pick_loop_params_dict(idx_innerLoop, innerNumTotal, innerLoopParams_dict)
            startTime_loop = time.time()
            weighting_method = method_param_dict['weighting_method']
            # before the loop for val_X, train the tree-based model for saving time
            tree_based_model = None
            if weighting_method in ['cart', 'rf']:
                max_depth = method_param_dict['max_depth']
                min_samples_split = method_param_dict['min_samples_split']
                min_samples_leaf = method_param_dict['min_samples_leaf']
                if weighting_method == 'cart':
                    tree_based_model = DecisionTreeRegressor(
                        max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                else:
                    n_estimators = method_param_dict['n_estimators']
                    tree_based_model = RandomForestRegressor(
                        n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                tree_based_model.fit(train_X, train_Y)
            elif weighting_method == 'sof':
                sof_max_depth, sof_min_samples_leaf, sof_n_estimators, sof_method = method_param_dict['max_depth'], method_param_dict['min_samples_leaf'], method_param_dict['n_estimators'], method_param_dict['method']
                sof_h, sof_b = np.array([ch]), np.array([cb])
                C_param = 100000 # set C to be a large number to ignore capacity constraints
                opt_solver = partial(solve_multi_nv, h_list = sof_h, b_list = sof_b, C = C_param, verbose = False)
                crit_method_dict = {
                    'oracle': partial(compute_crit_oracle, solver = opt_solver),
                    'apx-soln': partial(compute_crit_approx_sol, h_list = sof_h, b_list = sof_b),
                    'apx-risk': compute_crit_approx_risk,
                }
                impurity_method_dict = {
                    'oracle':   partial(impurity_oracle, h_list=sof_h, b_list=sof_b, C=C_param),
                    'apx-soln': partial(impurity_approx_sol, h_list=sof_h, b_list=sof_b, C=C_param),
                    'apx-risk': partial(impurity_approx_risk, h_list=sof_h, b_list=sof_b, C=C_param),
                }
                tree_based_model = forest(
                    opt_solver = opt_solver, 
                    hessian_computer = partial(compute_hessian, h_list = sof_h, b_list = sof_b, C = C_param),
                    gradient_computer = partial(compute_gradient, h_list = sof_h, b_list = sof_b, C = C_param), 
                    search_active_constraint = partial(search_active_constraint, C = C_param),
                    compute_update_step = partial(compute_update_step, constraint = True),
                    crit_computer = crit_method_dict[sof_method],
                    impurity_computer = impurity_method_dict[sof_method],
                    subsample_ratio=1.0, bootstrap = True, n_trees = sof_n_estimators, 
                    honesty = False, mtry = train_X.shape[1],
                    min_leaf_size = sof_min_samples_leaf, max_depth = sof_max_depth, 
                    n_proposals = 200, balancedness_tol = 0.0,
                    verbose = False, seed = None
                )
                tree_based_model.fit(train_Y.reshape((-1, 1)), train_X, train_Y.reshape((-1, 1)), train_X)
            
            # for each val_x, compute weights of training samples
            n_val = val_Y.shape[0]
            z_arr = np.zeros(n_val)
            for idx_val in range(n_val):
                x = val_X[idx_val]
                if weighting_method == 'default':
                    weight_arr = np.ones(n) / n
                    if problem_dict['ncvx_capacity'] is None:
                        z, _ = opt_nv_grb_model(grbModel, grbz, grbvB, grbvH, cb, ch, weight_arr)
                    else:
                        z = solve_ncvx_nv_saa(weight_arr, train_Y, cb, ch, problem_dict['ncvx_capacity'])
                    z_arr = np.ones(n_val) * z
                    sys.stdout.write(f'SAA({curr_exp_idx}/{total_exp_num}) ')
                    sys.stdout.flush()
                    break
                
                elif weighting_method == 'kNN':
                    k = min(method_param_dict['k'], train_Y.shape[0])    
                    weight_arr = np.zeros(n)
                    dist_arr = np.linalg.norm(train_X - x, axis=1)
                    idx_kNN = np.argsort(dist_arr)[:k]
                    weight_arr[idx_kNN] = 1 / k
                
                elif weighting_method == 'kernel':
                    gamma = method_param_dict['gamma']
                    weight_arr = np.zeros(n)
                    dist_arr = np.linalg.norm(train_X - x, axis=1)
                    weight_arr = np.exp(dist_arr/gamma)
                    weight_arr /= np.sum(weight_arr)

                elif weighting_method == 'cart':
                    leaf_idx = tree_based_model.apply(x.reshape(1, -1))[0]
                    weight_arr = np.zeros(n)
                    weight_arr = (tree_based_model.apply(train_X) == leaf_idx).astype(float)
                    weight_arr /= np.sum(weight_arr)
                
                elif weighting_method == 'rf':
                    weight_arr = np.zeros(n)
                    for tree in tree_based_model.estimators_:
                        leaf_idx = tree.apply(x.reshape(1, -1))[0]
                        weight_arr_curr = (tree.apply(train_X) == leaf_idx).astype(float)
                        weight_arr_curr /= np.sum(weight_arr_curr)
                        weight_arr += weight_arr_curr
                    weight_arr /= tree_based_model.n_estimators
                
                elif weighting_method == 'sof':
                    weight_arr = tree_based_model.get_weights(x)
                else:
                    raise ValueError(f'Invalid weighting method [{weighting_method}].')
                
                if problem_dict['ncvx_capacity'] is None:
                    z_arr[idx_val], _ = opt_nv_grb_model(grbModel, grbz, grbvB, grbvH, cb, ch, weight_arr)
                else:
                    z_arr[idx_val] = solve_ncvx_nv_saa(weight_arr, train_Y, cb, ch, problem_dict['ncvx_capacity'])

                update_progress(
                    (idx_outerLoop*innerNumTotal*n_val + idx_innerLoop*n_val + idx_val+1) / (outerNumTotal*innerNumTotal*n_val),
                    exp_start_time, method=f'wSAA/{weighting_method}', curr_exp_no=curr_exp_idx, total_exps=total_exp_num)
            val_cost = nv_cost(cb, ch, z_arr, val_Y) + capacost_function_oneprod(z_arr, problem_dict['ncvx_capacity'])
            val_costs.append(val_cost)
            exp_text = f'\nn={n:d} param_idx={idx_innerLoop+1:d}/{innerNumTotal:d} val_cost={val_cost:.4f} time={time.time()-startTime_loop:.4f} params={method_param_dict}'
            write_to_file(output_log_addr, exp_text)
        
        # compute test cost
        best_idx = np.argmin(val_costs)
        best_param_dict = pick_loop_params_dict(best_idx, innerNumTotal, innerLoopParams_dict)
        weighting_method = best_param_dict['weighting_method']
        if include_val:
            train_X_final = np.vstack((train_X, val_X))
            train_Y_final = np.vstack((train_Y, val_Y)) if len(train_Y.shape)>1 else np.hstack((train_Y, val_Y))
        else:
            train_X_final, train_Y_final = train_X, train_Y
        n_old = n
        n = train_X_final.shape[0]
        if problem_dict['ncvx_capacity'] is None:
            grbModel, grbz, grbvB, grbvH = build_nv_grb_model(train_Y_final)

        # before the loop for test_X
        tree_based_model = None
        if weighting_method in ['cart', 'rf']:
            max_depth = best_param_dict['max_depth']
            min_samples_split = best_param_dict['min_samples_split']
            min_samples_leaf = best_param_dict['min_samples_leaf']
            if weighting_method == 'cart':
                tree_based_model = DecisionTreeRegressor(
                    max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            else:
                n_estimators = best_param_dict['n_estimators']
                tree_based_model = RandomForestRegressor(
                    n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            tree_based_model.fit(train_X_final, train_Y_final)
        elif weighting_method == 'sof':
            sof_max_depth, sof_min_samples_leaf, sof_n_estimators, sof_method = best_param_dict['max_depth'], best_param_dict['min_samples_leaf'], best_param_dict['n_estimators'], best_param_dict['method']
            sof_h, sof_b = np.array([ch]), np.array([cb])
            C_param = 100000 # set C to be a large number to ignore capacity constraints
            opt_solver = partial(solve_multi_nv, h_list = sof_h, b_list = sof_b, C = C_param, verbose = False)
            crit_method_dict = {
                'oracle': partial(compute_crit_oracle, solver = opt_solver),
                'apx-soln': partial(compute_crit_approx_sol, h_list = sof_h, b_list = sof_b),
                'apx-risk': compute_crit_approx_risk,
            }
            impurity_method_dict = {
                'oracle':   partial(impurity_oracle, h_list=sof_h, b_list=sof_b, C=C_param),
                'apx-soln': partial(impurity_approx_sol, h_list=sof_h, b_list=sof_b, C=C_param),
                'apx-risk': partial(impurity_approx_risk, h_list=sof_h, b_list=sof_b, C=C_param),
            }
            tree_based_model = forest(
                opt_solver = opt_solver, 
                hessian_computer = partial(compute_hessian, h_list = sof_h, b_list = sof_b, C = C_param),
                gradient_computer = partial(compute_gradient, h_list = sof_h, b_list = sof_b, C = C_param), 
                search_active_constraint = partial(search_active_constraint, C = C_param),
                compute_update_step = partial(compute_update_step, constraint = True),
                crit_computer = crit_method_dict[sof_method],
                impurity_computer = impurity_method_dict[sof_method],
                subsample_ratio=1.0, bootstrap = True, n_trees = sof_n_estimators, 
                honesty = False, mtry = train_X_final.shape[1],
                min_leaf_size = sof_min_samples_leaf, max_depth = sof_max_depth, 
                n_proposals = 200, balancedness_tol = 0.0,
                verbose = False, seed = None
            )
            tree_based_model.fit(train_Y_final.reshape((-1, 1)), train_X_final, train_Y_final.reshape((-1, 1)), train_X_final)
        
        # for each test_x, compute weights of training samples
        n_test = test_Y.shape[0]
        z_arr_test = np.zeros(n_test)
        for idx_test in range(n_test):
            x = test_X[idx_test]
            if weighting_method == 'default':
                weight_arr = np.ones(n) / n
                if problem_dict['ncvx_capacity'] is None:
                    z, _ = opt_nv_grb_model(grbModel, grbz, grbvB, grbvH, cb, ch, weight_arr)
                else:
                    z = solve_ncvx_nv_saa(weight_arr, train_Y_final, cb, ch, problem_dict['ncvx_capacity'])
                z_arr_test = np.ones(n_test) * z
                sys.stdout.write(f'SAA({curr_exp_idx}/{total_exp_num}) ')
                sys.stdout.flush()
                break
            
            elif weighting_method == 'kNN':
                k = min(best_param_dict['k'], train_Y_final.shape[0])    
                weight_arr = np.zeros(n)
                dist_arr = np.linalg.norm(train_X_final - x, axis=1)
                idx_kNN = np.argsort(dist_arr)[:k]
                weight_arr[idx_kNN] = 1 / k
            
            elif weighting_method == 'kernel':
                gamma = best_param_dict['gamma']
                weight_arr = np.zeros(n)
                dist_arr = np.linalg.norm(train_X_final - x, axis=1)
                weight_arr = np.exp(dist_arr/gamma)
                weight_arr /= np.sum(weight_arr)
            elif weighting_method == 'cart':
                leaf_idx = tree_based_model.apply(x.reshape(1, -1))[0]
                weight_arr = np.zeros(n)
                weight_arr = (tree_based_model.apply(train_X_final) == leaf_idx).astype(float)
                weight_arr /= np.sum(weight_arr)
            
            elif weighting_method == 'rf':
                weight_arr = np.zeros(n)
                for tree in tree_based_model.estimators_:
                    leaf_idx = tree.apply(x.reshape(1, -1))[0]
                    weight_arr_curr = (tree.apply(train_X_final) == leaf_idx).astype(float)
                    weight_arr_curr /= np.sum(weight_arr_curr)
                    weight_arr += weight_arr_curr
                weight_arr /= tree_based_model.n_estimators
            
            elif weighting_method == 'sof':
                weight_arr = tree_based_model.get_weights(x)
            else:
                raise ValueError(f'Invalid weighting method [{weighting_method}].')
            
            if problem_dict['ncvx_capacity'] is None:
                z_arr_test[idx_test], _ = opt_nv_grb_model(grbModel, grbz, grbvB, grbvH, cb, ch, weight_arr)
            else:
                z_arr_test[idx_test] = solve_ncvx_nv_saa(weight_arr, train_Y_final, cb, ch, problem_dict['ncvx_capacity'])
        test_cost = nv_cost(cb, ch, z_arr_test, test_Y) + capacost_function_oneprod(z_arr_test, problem_dict['ncvx_capacity'])
        exp_text = f'\nOuterLoop{idx_outerLoop} (n={n_old}) with innerLoop{best_idx+1}: val_cost={val_costs[best_idx]:.4f} test_cost={test_cost:.4f}\n\n'
        write_to_file(output_log_addr, exp_text)

def exp_pp_wSAA(train_X_data, train_Y_data, val_X, val_Y, test_X, test_Y, problem_dict, param_dict, output_log_addr, 
             curr_exp_idx, total_exp_num):
    exp_start_time = time.time()

    c, g, b, h = problem_dict['c'], problem_dict['g'], problem_dict['b'], problem_dict['h']
    W = problem_dict['W']

    outerLoopParams_dict, outerNumTotal = load_loop_params(problem_dict, 'training_sample_size')
    for idx_outerLoop in range(outerNumTotal):
        n, = pick_loop_params(idx_outerLoop, outerNumTotal, outerLoopParams_dict)
        assert n > 0
        if n > train_X_data.shape[0]:
            print(f'Warning: n={n} is larger than the training sample size. Set n to the training data size ({train_X_data.shape[0]}).')
            n = train_X_data.shape[0]
        # assert n <= train_X_data.shape[0]
        train_X, train_Y = train_X_data[:n], train_Y_data[:n]
        
        grbModel, grbz, grbT = build_pp_grb_model(train_Y, g, b, h, W)
        innerLoopParams_dict, innerNumTotal = load_loop_params(param_dict, *param_dict.keys())
        val_costs = []
        for idx_innerLoop in range(innerNumTotal):
            method_param_dict = pick_loop_params_dict(idx_innerLoop, innerNumTotal, innerLoopParams_dict)
            startTime_loop = time.time()
            weighting_method = method_param_dict['weighting_method']
            # before the loop for val_X
            tree_based_model = None
            if weighting_method in ['cart', 'rf']:
                max_depth = method_param_dict['max_depth']
                min_samples_split = method_param_dict['min_samples_split']
                min_samples_leaf = method_param_dict['min_samples_leaf']
                if weighting_method == 'cart':
                    tree_based_model = DecisionTreeRegressor(
                        max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                else:
                    n_estimators = method_param_dict['n_estimators']
                    tree_based_model = RandomForestRegressor(
                        n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                tree_based_model.fit(train_X, train_Y)

            # for each val_x, compute weights of training samples
            n_val = val_Y.shape[0]
            z_arr = np.zeros((n_val, c.shape[0]))
            for idx_val in range(n_val):
                x = val_X[idx_val]
                if weighting_method == 'default':
                    weight_arr = np.ones(n) / n
                    z, _ = opt_pp_grb_model(grbModel, grbz, grbT, c, weight_arr)
                    z_arr = np.ones((n_val, 1)) @ z.reshape((1, -1))
                    sys.stdout.write(f'SAA({curr_exp_idx}/{total_exp_num}) ')
                    sys.stdout.flush()
                    break

                elif weighting_method == 'kNN':
                    k = min(method_param_dict['k'], train_Y.shape[0])    
                    weight_arr = np.zeros(n)
                    dist_arr = np.linalg.norm(train_X - x, axis=1)
                    idx_kNN = np.argsort(dist_arr)[:k]
                    weight_arr[idx_kNN] = 1 / k

                elif weighting_method == 'kernel':
                    gamma = method_param_dict['gamma']
                    weight_arr = np.zeros(n)
                    dist_arr = np.linalg.norm(train_X - x, axis=1)
                    weight_arr = np.exp(dist_arr/gamma)
                    weight_arr /= np.sum(weight_arr)

                elif weighting_method == 'cart':
                    leaf_idx = tree_based_model.apply(x.reshape(1, -1))[0]
                    weight_arr = np.zeros(n)
                    weight_arr = (tree_based_model.apply(train_X) == leaf_idx).astype(float)
                    weight_arr /= np.sum(weight_arr)
                
                elif weighting_method == 'rf':
                    weight_arr = np.zeros(n)
                    for tree in tree_based_model.estimators_:
                        leaf_idx = tree.apply(x.reshape(1, -1))[0]
                        weight_arr_curr = (tree.apply(train_X) == leaf_idx).astype(float)
                        weight_arr_curr /= np.sum(weight_arr_curr)
                        weight_arr += weight_arr_curr
                    weight_arr /= tree_based_model.n_estimators

                else:
                    raise ValueError(f'Invalid weighting method [{weighting_method}].')
            
                z_arr[idx_val, :], _ = opt_pp_grb_model(grbModel, grbz, grbT, c, weight_arr)
                update_progress(
                    (idx_outerLoop*innerNumTotal*n_val + idx_innerLoop*n_val + idx_val+1) / (outerNumTotal*innerNumTotal*n_val),
                    exp_start_time, method=f'wSAA/{weighting_method}', curr_exp_no=curr_exp_idx, total_exps=total_exp_num)

            val_cost = pp_cost(c, g, b, h, W, z_arr, val_Y)
            val_costs.append(val_cost)
            exp_text = f'\nn={n:d} param_idx={idx_innerLoop+1:d}/{innerNumTotal:d} val_cost={val_cost:.4f} time={time.time()-startTime_loop:.4f} params={method_param_dict}'
            write_to_file(output_log_addr, exp_text)
        
        # compute test cost
        best_idx = np.argmin(val_costs)
        best_param_dict = pick_loop_params_dict(best_idx, innerNumTotal, innerLoopParams_dict)
        weighting_method = best_param_dict['weighting_method']
        train_X_final, train_Y_final = train_X, train_Y

        # before the loop for test_X
        tree_based_model = None
        if weighting_method in ['cart', 'rf']:
            max_depth = best_param_dict['max_depth']
            min_samples_split = best_param_dict['min_samples_split']
            min_samples_leaf = best_param_dict['min_samples_leaf']
            if weighting_method == 'cart':
                tree_based_model = DecisionTreeRegressor(
                    max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            else:
                n_estimators = best_param_dict['n_estimators']
                tree_based_model = RandomForestRegressor(
                    n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            tree_based_model.fit(train_X_final, train_Y_final)
        
        # for each test_x, compute weights of training samples
        n_test = test_Y.shape[0]
        z_arr_test = np.zeros((n_test, c.shape[0]))
        for idx_test in range(n_test):
            x = test_X[idx_test]
            if weighting_method == 'default':
                weight_arr = np.ones(n) / n
                z, _ = opt_pp_grb_model(grbModel, grbz, grbT, c, weight_arr)
                z_arr_test = np.ones((n_test, 1)) @ z.reshape((1, -1))
                sys.stdout.write(f'SAA({curr_exp_idx}/{total_exp_num}) ')
                sys.stdout.flush()
                break
            
            elif weighting_method == 'kNN':
                k = min(best_param_dict['k'], train_Y.shape[0])    
                weight_arr = np.zeros(n)
                dist_arr = np.linalg.norm(train_X - x, axis=1)
                idx_kNN = np.argsort(dist_arr)[:k]
                weight_arr[idx_kNN] = 1 / k
            
            elif weighting_method == 'kernel':
                gamma = best_param_dict['gamma']
                weight_arr = np.zeros(n)
                dist_arr = np.linalg.norm(train_X - x, axis=1)
                weight_arr = np.exp(dist_arr/gamma)
                weight_arr /= np.sum(weight_arr)
            elif weighting_method == 'cart':
                leaf_idx = tree_based_model.apply(x.reshape(1, -1))[0]
                weight_arr = np.zeros(n)
                weight_arr = (tree_based_model.apply(train_X) == leaf_idx).astype(float)
                weight_arr /= np.sum(weight_arr)
            
            elif weighting_method == 'rf':
                weight_arr = np.zeros(n)
                for tree in tree_based_model.estimators_:
                    leaf_idx = tree.apply(x.reshape(1, -1))[0]
                    weight_arr_curr = (tree.apply(train_X) == leaf_idx).astype(float)
                    weight_arr_curr /= np.sum(weight_arr_curr)
                    weight_arr += weight_arr_curr
                weight_arr /= tree_based_model.n_estimators
            
            else:
                raise ValueError(f'Invalid weighting method [{weighting_method}].')
            
            z_arr_test[idx_test, :], _ = opt_pp_grb_model(grbModel, grbz, grbT, c, weight_arr)
        test_cost = pp_cost(c, g, b, h, W, z_arr_test, test_Y)
        exp_text = f'\nOuterLoop{idx_outerLoop} (n={n}) with innerLoop{best_idx+1}: val_cost={val_costs[best_idx]:.4f} test_cost={test_cost:.4f}\n\n'
        write_to_file(output_log_addr, exp_text)


def exp_nv2prod_wSAA(train_X_data, train_Y_data, val_X, val_Y, test_X, test_Y, 
                     problem_dict, param_dict, output_log_addr, 
                     curr_exp_idx, total_exp_num):
    exp_start_time = time.time()

    cb1, cb2, ch1, ch2 = problem_dict['c_b_1'], problem_dict['c_b_2'], problem_dict['c_h_1'], problem_dict['c_h_2']
    capacity = problem_dict['fixed_capacity']

    outerLoopParams_dict, outerNumTotal = load_loop_params(problem_dict, 'training_sample_size')
    for idx_outerLoop in range(outerNumTotal):
        n, = pick_loop_params(idx_outerLoop, outerNumTotal, outerLoopParams_dict)
        assert n > 0
        if n > train_X_data.shape[0]:
            print(f'Warning: n={n} is larger than the training sample size. Set n to the training data size ({train_X_data.shape[0]}).')
            n = train_X_data.shape[0]
        # assert n <= train_X_data.shape[0]
        train_X, train_Y = train_X_data[:n], train_Y_data[:n]

        grbModel, grbz_tup, grbvB_tup, grbvH_tup = build_nv2prod_grb_model(train_Y, capacity)
        innerLoopParams_dict, innerNumTotal = load_loop_params(param_dict, *param_dict.keys())
        val_costs = []
        for idx_innerLoop in range(innerNumTotal):
            method_param_dict = pick_loop_params_dict(idx_innerLoop, innerNumTotal, innerLoopParams_dict)
            startTime_loop = time.time()
            weighting_method = method_param_dict['weighting_method']
            # before the loop for val_X
            tree_based_model = None
            if weighting_method in ['cart', 'rf']:
                max_depth = method_param_dict['max_depth']
                min_samples_split = method_param_dict['min_samples_split']
                min_samples_leaf = method_param_dict['min_samples_leaf']
                if weighting_method == 'cart':
                    tree_based_model = DecisionTreeRegressor(
                        max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                else:
                    n_estimators = method_param_dict['n_estimators']
                    tree_based_model = RandomForestRegressor(
                        n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                tree_based_model.fit(train_X, train_Y)
            elif weighting_method == 'sof':
                sof_max_depth, sof_min_samples_leaf, sof_n_estimators, sof_method = method_param_dict['max_depth'], method_param_dict['min_samples_leaf'], method_param_dict['n_estimators'], method_param_dict['method']
                sof_h, sof_b = np.array([ch1, ch2]), np.array([cb1, cb2])
                C_param = capacity # set C to be a large number to ignore capacity constraints
                opt_solver = partial(solve_multi_nv, h_list = sof_h, b_list = sof_b, C = C_param, verbose = False)
                crit_method_dict = {
                    'oracle': partial(compute_crit_oracle, solver = opt_solver),
                    'apx-soln': partial(compute_crit_approx_sol, h_list = sof_h, b_list = sof_b),
                    'apx-risk': compute_crit_approx_risk,
                }
                impurity_method_dict = {
                    'oracle':   partial(impurity_oracle, h_list=sof_h, b_list=sof_b, C=C_param),
                    'apx-soln': partial(impurity_approx_sol, h_list=sof_h, b_list=sof_b, C=C_param),
                    'apx-risk': partial(impurity_approx_risk, h_list=sof_h, b_list=sof_b, C=C_param),
                }
                tree_based_model = forest(
                    opt_solver = opt_solver, 
                    hessian_computer = partial(compute_hessian, h_list = sof_h, b_list = sof_b, C = C_param),
                    gradient_computer = partial(compute_gradient, h_list = sof_h, b_list = sof_b, C = C_param), 
                    search_active_constraint = partial(search_active_constraint, C = C_param),
                    compute_update_step = partial(compute_update_step, constraint = True),
                    crit_computer = crit_method_dict[sof_method],
                    impurity_computer = impurity_method_dict[sof_method],
                    subsample_ratio=1.0, bootstrap = True, n_trees = sof_n_estimators, 
                    honesty = False, mtry = train_X.shape[1],
                    min_leaf_size = sof_min_samples_leaf, max_depth = sof_max_depth, 
                    n_proposals = 200, balancedness_tol = 0.0,
                    verbose = False, seed = None
                )
                tree_based_model.fit(train_Y, train_X, train_Y, train_X)
            # for each val_x, compute weights of training samples
            n_val = val_Y.shape[0]
            z_arr = np.zeros((n_val, 2))
            for idx_val in range(n_val):
                x = val_X[idx_val]
                if weighting_method == 'default':
                    weight_arr = np.ones(n) / n
                    z, _ = opt_nv2prod_grb_model(grbModel, grbz_tup, grbvB_tup, grbvH_tup, (cb1, cb2), (ch1, ch2), weight_arr)
                    z_arr = np.ones((n_val, 1)) @ z.reshape((1, -1))
                    sys.stdout.write(f'SAA({curr_exp_idx}/{total_exp_num}) ')
                    sys.stdout.flush()
                    break
                
                elif weighting_method == 'kNN':
                    k = min(method_param_dict['k'], train_Y.shape[0])    
                    weight_arr = np.zeros(n)
                    dist_arr = np.linalg.norm(train_X - x, axis=1)
                    idx_kNN = np.argsort(dist_arr)[:k]
                    weight_arr[idx_kNN] = 1 / k
                
                elif weighting_method == 'kernel':
                    gamma = method_param_dict['gamma']
                    weight_arr = np.zeros(n)
                    dist_arr = np.linalg.norm(train_X - x, axis=1)
                    weight_arr = np.exp(dist_arr/gamma)
                    weight_arr /= np.sum(weight_arr)

                elif weighting_method == 'cart':
                    leaf_idx = tree_based_model.apply(x.reshape(1, -1))[0]
                    weight_arr = np.zeros(n)
                    weight_arr = (tree_based_model.apply(train_X) == leaf_idx).astype(float)
                    weight_arr /= np.sum(weight_arr)
                
                elif weighting_method == 'rf':
                    weight_arr = np.zeros(n)
                    for tree in tree_based_model.estimators_:
                        leaf_idx = tree.apply(x.reshape(1, -1))[0]
                        weight_arr_curr = (tree.apply(train_X) == leaf_idx).astype(float)
                        weight_arr_curr /= np.sum(weight_arr_curr)
                        weight_arr += weight_arr_curr
                    weight_arr /= tree_based_model.n_estimators
                
                elif weighting_method == 'sof':
                    weight_arr = tree_based_model.get_weights(x)

                else:
                    raise ValueError(f'Invalid weighting method [{weighting_method}].')
                
                z_arr[idx_val], _ = opt_nv2prod_grb_model(grbModel, grbz_tup, grbvB_tup, grbvH_tup, (cb1, cb2), (ch1, ch2), weight_arr)

                update_progress(
                    (idx_outerLoop*innerNumTotal*n_val + idx_innerLoop*n_val + idx_val+1) / (outerNumTotal*innerNumTotal*n_val),
                    exp_start_time, method=f'wSAA/{weighting_method}', curr_exp_no=curr_exp_idx, total_exps=total_exp_num)
            val_cost, _ = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, z_arr, val_Y)
            val_costs.append(val_cost)
            exp_text = f'\nn={n:d} param_idx={idx_innerLoop+1:d}/{innerNumTotal:d} val_cost={val_cost:.4f} time={time.time()-startTime_loop:.4f} params={method_param_dict}'
            write_to_file(output_log_addr, exp_text)
        
        # compute test cost
        best_idx = np.argmin(val_costs)
        best_param_dict = pick_loop_params_dict(best_idx, innerNumTotal, innerLoopParams_dict)
        weighting_method = best_param_dict['weighting_method']
        train_X_final, train_Y_final = train_X, train_Y

        # before the loop for test_X
        tree_based_model = None
        if weighting_method in ['cart', 'rf']:
            max_depth = best_param_dict['max_depth']
            min_samples_split = best_param_dict['min_samples_split']
            min_samples_leaf = best_param_dict['min_samples_leaf']
            if weighting_method == 'cart':
                tree_based_model = DecisionTreeRegressor(
                    max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            else:
                n_estimators = best_param_dict['n_estimators']
                tree_based_model = RandomForestRegressor(
                    n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            tree_based_model.fit(train_X_final, train_Y_final)
        elif weighting_method == 'sof':
            sof_max_depth, sof_min_samples_leaf, sof_n_estimators, sof_method = best_param_dict['max_depth'], best_param_dict['min_samples_leaf'], best_param_dict['n_estimators'], best_param_dict['method']
            sof_h, sof_b = np.array([ch1, ch2]), np.array([cb1, cb2])
            C_param = capacity # set C to be a large number to ignore capacity constraints
            opt_solver = partial(solve_multi_nv, h_list = sof_h, b_list = sof_b, C = C_param, verbose = False)
            crit_method_dict = {
                'oracle': partial(compute_crit_oracle, solver = opt_solver),
                'apx-soln': partial(compute_crit_approx_sol, h_list = sof_h, b_list = sof_b),
                'apx-risk': compute_crit_approx_risk,
            }
            impurity_method_dict = {
                'oracle':   partial(impurity_oracle, h_list=sof_h, b_list=sof_b, C=C_param),
                'apx-soln': partial(impurity_approx_sol, h_list=sof_h, b_list=sof_b, C=C_param),
                'apx-risk': partial(impurity_approx_risk, h_list=sof_h, b_list=sof_b, C=C_param),
            }
            tree_based_model = forest(
                opt_solver = opt_solver, 
                hessian_computer = partial(compute_hessian, h_list = sof_h, b_list = sof_b, C = C_param),
                gradient_computer = partial(compute_gradient, h_list = sof_h, b_list = sof_b, C = C_param), 
                search_active_constraint = partial(search_active_constraint, C = C_param),
                compute_update_step = partial(compute_update_step, constraint = True),
                crit_computer = crit_method_dict[sof_method],
                impurity_computer = impurity_method_dict[sof_method],
                subsample_ratio=1.0, bootstrap = True, n_trees = sof_n_estimators, 
                honesty = False, mtry = train_X_final.shape[1],
                min_leaf_size = sof_min_samples_leaf, max_depth = sof_max_depth, 
                n_proposals = 200, balancedness_tol = 0.0,
                verbose = False, seed = None
            )
            tree_based_model.fit(train_Y_final, train_X_final, train_Y_final, train_X_final)
        # for each test_x, compute weights of training samples
        n_test = test_Y.shape[0]
        z_arr_test = np.zeros((n_test, 2))
        for idx_test in range(n_test):
            x = test_X[idx_test]
            if weighting_method == 'default':
                weight_arr = np.ones(n) / n
                z, _ = opt_nv2prod_grb_model(grbModel, grbz_tup, grbvB_tup, grbvH_tup, (cb1, cb2), (ch1, ch2), weight_arr)
                z_arr_test = np.ones((n_test, 1)) @ z.reshape((1, -1))
                sys.stdout.write(f'SAA({curr_exp_idx}/{total_exp_num}) ')
                sys.stdout.flush()
                break
            
            elif weighting_method == 'kNN':
                k = min(best_param_dict['k'], train_Y.shape[0])    
                weight_arr = np.zeros(n)
                dist_arr = np.linalg.norm(train_X - x, axis=1)
                idx_kNN = np.argsort(dist_arr)[:k]
                weight_arr[idx_kNN] = 1 / k
            
            elif weighting_method == 'kernel':
                gamma = best_param_dict['gamma']
                weight_arr = np.zeros(n)
                dist_arr = np.linalg.norm(train_X - x, axis=1)
                weight_arr = np.exp(dist_arr/gamma)
                weight_arr /= np.sum(weight_arr)
            elif weighting_method == 'cart':
                leaf_idx = tree_based_model.apply(x.reshape(1, -1))[0]
                weight_arr = np.zeros(n)
                weight_arr = (tree_based_model.apply(train_X) == leaf_idx).astype(float)
                weight_arr /= np.sum(weight_arr)
            
            elif weighting_method == 'rf':
                weight_arr = np.zeros(n)
                for tree in tree_based_model.estimators_:
                    leaf_idx = tree.apply(x.reshape(1, -1))[0]
                    weight_arr_curr = (tree.apply(train_X) == leaf_idx).astype(float)
                    weight_arr_curr /= np.sum(weight_arr_curr)
                    weight_arr += weight_arr_curr
                weight_arr /= tree_based_model.n_estimators
            
            elif weighting_method == 'sof':
                weight_arr = tree_based_model.get_weights(x)
            else:
                raise ValueError(f'Invalid weighting method [{weighting_method}].')
            
            z_arr_test[idx_test], _ = opt_nv2prod_grb_model(grbModel, grbz_tup, grbvB_tup, grbvH_tup, (cb1, cb2), (ch1, ch2), weight_arr)
        test_cost, _ = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, z_arr_test, test_Y)
        exp_text = f'\nOuterLoop{idx_outerLoop} (n={n}) with innerLoop{best_idx+1}: val_cost={val_costs[best_idx]:.4f} test_cost={test_cost:.4f}\n\n'
        write_to_file(output_log_addr, exp_text)

def exp_ncvxconstr_nv2prod_wSAA(train_X_data, train_Y_data, val_X, val_Y, test_X, test_Y, 
                     problem_dict, param_dict, output_log_addr, 
                     curr_exp_idx, total_exp_num):
    exp_start_time = time.time()

    cb1, cb2, ch1, ch2 = problem_dict['c_b_1'], problem_dict['c_b_2'], problem_dict['c_h_1'], problem_dict['c_h_2']
    capacity = problem_dict['fixed_capacity']
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
        val_costs = []
        for idx_innerLoop in range(innerNumTotal):
            method_param_dict = pick_loop_params_dict(idx_innerLoop, innerNumTotal, innerLoopParams_dict)
            startTime_loop = time.time()
            weighting_method = method_param_dict['weighting_method']
            # before the loop for val_X
            tree_based_model = None
            if weighting_method in ['cart', 'rf']:
                max_depth = method_param_dict['max_depth']
                min_samples_split = method_param_dict['min_samples_split']
                min_samples_leaf = method_param_dict['min_samples_leaf']
                if weighting_method == 'cart':
                    tree_based_model = DecisionTreeRegressor(
                        max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                else:
                    n_estimators = method_param_dict['n_estimators']
                    tree_based_model = RandomForestRegressor(
                        n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
                tree_based_model.fit(train_X, train_Y)

            # for each val_x, compute weights of training samples
            n_val = val_Y.shape[0]
            z_arr = np.zeros((n_val, 2))
            for idx_val in range(n_val):
                x = val_X[idx_val]
                if weighting_method == 'default':
                    weight_arr = np.ones(n) / n
                    if idx_val > 0:
                        z_arr[idx_val] = z_arr[0]
                        update_progress((idx_outerLoop*innerNumTotal*n_val + idx_innerLoop*n_val + idx_val+1) / (outerNumTotal*innerNumTotal*n_val),
                                        exp_start_time, method=f'wSAA/{weighting_method}', curr_exp_no=curr_exp_idx, total_exps=total_exp_num)
                        continue
                
                elif weighting_method == 'kNN':
                    k = min(method_param_dict['k'], train_Y.shape[0])    
                    weight_arr = np.zeros(n)
                    dist_arr = np.linalg.norm(train_X - x, axis=1)
                    idx_kNN = np.argsort(dist_arr)[:k]
                    weight_arr[idx_kNN] = 1 / k
                
                elif weighting_method == 'kernel':
                    gamma = method_param_dict['gamma']
                    weight_arr = np.zeros(n)
                    dist_arr = np.linalg.norm(train_X - x, axis=1)
                    weight_arr = np.exp(dist_arr/gamma)
                    weight_arr /= np.sum(weight_arr)

                elif weighting_method == 'cart':
                    leaf_idx = tree_based_model.apply(x.reshape(1, -1))[0]
                    weight_arr = np.zeros(n)
                    weight_arr = (tree_based_model.apply(train_X) == leaf_idx).astype(float)
                    weight_arr /= np.sum(weight_arr)
                
                elif weighting_method == 'rf':
                    weight_arr = np.zeros(n)
                    for tree in tree_based_model.estimators_:
                        leaf_idx = tree.apply(x.reshape(1, -1))[0]
                        weight_arr_curr = (tree.apply(train_X) == leaf_idx).astype(float)
                        weight_arr_curr /= np.sum(weight_arr_curr)
                        weight_arr += weight_arr_curr
                    weight_arr /= tree_based_model.n_estimators

                else:
                    raise ValueError(f'Invalid weighting method [{weighting_method}].')
                
                z1 = np.linspace(0, np.max(train_Y[:, 0]), 50)
                z2 = np.linspace(0, np.max(train_Y[:, 1]), 50)
                z1_grid, z2_grid = np.meshgrid(z1, z2)
                z_grid = np.vstack((z1_grid.flatten(), z2_grid.flatten())).T
                capacity_cost = np.sum([capacost_function_oneprod(z_grid[:, idx], ncvx_constraint_arr, average=False) for idx in range(z_grid.shape[1])], axis=0)
                z_grid = z_grid[capacity_cost <= capacity, :]

                grid_costs = np.sum([
                    np.average(nv_cost([cb1, cb2][idx], [ch1, ch2][idx], z_grid[:, idx].reshape((-1, 1)), train_Y[:, idx].reshape((1, -1)), average=False), axis=1, weights=weight_arr)
                     for idx in range(z_grid.shape[1])], axis=0)
                z_arr[idx_val] = z_grid[np.argmin(grid_costs)]
                update_progress(
                    (idx_outerLoop*innerNumTotal*n_val + idx_innerLoop*n_val + idx_val+1) / (outerNumTotal*innerNumTotal*n_val),
                    exp_start_time, method=f'wSAA/{weighting_method}', curr_exp_no=curr_exp_idx, total_exps=total_exp_num)
                
            val_cost, _ = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, z_arr, val_Y, constr_arr=ncvx_constraint_arr)
            val_costs.append(val_cost)
            exp_text = f'\nn={n:d} param_idx={idx_innerLoop+1:d}/{innerNumTotal:d} val_cost={val_cost:.4f} time={time.time()-startTime_loop:.4f} params={method_param_dict}'
            write_to_file(output_log_addr, exp_text)
        
        # compute test cost
        best_idx = np.argmin(val_costs)
        best_param_dict = pick_loop_params_dict(best_idx, innerNumTotal, innerLoopParams_dict)
        weighting_method = best_param_dict['weighting_method']
        train_X_final, train_Y_final = train_X, train_Y

        # before the loop for test_X
        tree_based_model = None
        if weighting_method in ['cart', 'rf']:
            max_depth = best_param_dict['max_depth']
            min_samples_split = best_param_dict['min_samples_split']
            min_samples_leaf = best_param_dict['min_samples_leaf']
            if weighting_method == 'cart':
                tree_based_model = DecisionTreeRegressor(
                    max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            else:
                n_estimators = best_param_dict['n_estimators']
                tree_based_model = RandomForestRegressor(
                    n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            tree_based_model.fit(train_X_final, train_Y_final)
        
        # for each test_x, compute weights of training samples
        n_test = test_Y.shape[0]
        z_arr_test = np.zeros((n_test, 2))
        for idx_test in range(n_test):
            x = test_X[idx_test]
            if weighting_method == 'default':
                weight_arr = np.ones(n) / n
                if idx_test > 0:
                    z_arr_test[idx_test] = z_arr_test[0]
                    update_progress((idx_outerLoop*innerNumTotal*n_test + idx_innerLoop*n_test + idx_test+1) / (outerNumTotal*innerNumTotal*n_test),
                                    exp_start_time, method=f'wSAA/{weighting_method}-test', curr_exp_no=curr_exp_idx, total_exps=total_exp_num)
                    continue
            
            elif weighting_method == 'kNN':
                k = min(best_param_dict['k'], train_Y.shape[0])    
                weight_arr = np.zeros(n)
                dist_arr = np.linalg.norm(train_X - x, axis=1)
                idx_kNN = np.argsort(dist_arr)[:k]
                weight_arr[idx_kNN] = 1 / k
            
            elif weighting_method == 'kernel':
                gamma = best_param_dict['gamma']
                weight_arr = np.zeros(n)
                dist_arr = np.linalg.norm(train_X - x, axis=1)
                weight_arr = np.exp(dist_arr/gamma)
                weight_arr /= np.sum(weight_arr)
            elif weighting_method == 'cart':
                leaf_idx = tree_based_model.apply(x.reshape(1, -1))[0]
                weight_arr = np.zeros(n)
                weight_arr = (tree_based_model.apply(train_X) == leaf_idx).astype(float)
                weight_arr /= np.sum(weight_arr)
            
            elif weighting_method == 'rf':
                weight_arr = np.zeros(n)
                for tree in tree_based_model.estimators_:
                    leaf_idx = tree.apply(x.reshape(1, -1))[0]
                    weight_arr_curr = (tree.apply(train_X) == leaf_idx).astype(float)
                    weight_arr_curr /= np.sum(weight_arr_curr)
                    weight_arr += weight_arr_curr
                weight_arr /= tree_based_model.n_estimators
            else:
                raise ValueError(f'Invalid weighting method [{weighting_method}].')
        
            # find the best z using grid search
            z1 = np.linspace(0, np.max(train_Y[:, 0]), 50)
            z2 = np.linspace(0, np.max(train_Y[:, 1]), 50)
            z1_grid, z2_grid = np.meshgrid(z1, z2)
            z_grid = np.vstack((z1_grid.flatten(), z2_grid.flatten())).T
            capacity_cost = np.sum([capacost_function_oneprod(z_grid[:, idx], ncvx_constraint_arr, average=False) for idx in range(z_grid.shape[1])], axis=0)
            z_grid = z_grid[capacity_cost <= capacity, :]

            grid_costs = np.sum([
                np.average(nv_cost([cb1, cb2][idx], [ch1, ch2][idx], z_grid[:, idx].reshape((-1, 1)), train_Y[:, idx].reshape((1, -1)), average=False), axis=1, weights=weight_arr)
                for idx in range(z_grid.shape[1])], axis=0)
            z_arr_test[idx_test] = z_grid[np.argmin(grid_costs)]
            update_progress(
                (idx_outerLoop*innerNumTotal*n_test + idx_innerLoop*n_test + idx_test+1) / (outerNumTotal*innerNumTotal*n_test),
                exp_start_time, method=f'wSAA/{weighting_method}-test', curr_exp_no=curr_exp_idx, total_exps=total_exp_num)
            
        test_cost, _ = feasible_2prodnv_cost([cb1, cb2], [ch1, ch2], capacity, z_arr_test, test_Y, constr_arr=ncvx_constraint_arr)
        exp_text = f'\nOuterLoop{idx_outerLoop} (n={n}) with innerLoop{best_idx+1}: val_cost={val_costs[best_idx]:.4f} test_cost={test_cost:.4f}\n\n'
        write_to_file(output_log_addr, exp_text)