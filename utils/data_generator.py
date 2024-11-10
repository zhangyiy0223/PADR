from utils.tools import *
from utils.tools_pp import *
import inspect

def true_data_model(X, dim_y, model_name='linear', output_info=False, load_model_param=None, output_model_param=False, ):
    """
    True demand models used in the method 'simopt' and data generation.
    """

    # linear model for the 2-product constrained newsvendor problem
    def linear_model(X_input, load_model=None, sparse=True):
        if load_model is not None:
            coef = load_model['coef']
        else:
            coef = np.array([[15, 15], [-5, 5]])
        X = X_input[:, :2] if sparse else np.hstack((
                np.average(X_input[:, 0:X_input.shape[1]//2], axis=1).reshape((-1,1)), 
                np.average(X_input[:, X_input.shape[1]//2:], axis=1).reshape((-1,1))))
        Y = X @ coef + np.random.normal(30, 1, (X.shape[0], dim_y))
        Y = Y * (Y > 0)
        return Y, {'coef': coef}

    # max-affine model with 3 pieces
    def MA3(X_input, k=1, sparse=True, std=1):
        X = X_input[:, :2] if sparse else np.hstack((
            np.average(X_input[:, 0:X_input.shape[1]//2], axis=1).reshape((-1,1)), 
            np.average(X_input[:, X_input.shape[1]//2:], axis=1).reshape((-1,1))))
        Y = k * np.max((-2*X[:, 0]+X[:, 1], X[:, 0]-2*X[:, 1], 3*X[:, 0]), axis=0) + np.random.normal(10, std, (X.shape[0], ))
        Y = Y * (Y > 0)
        return Y, None

    # sine plus piecewise affine model
    def SINPA(X_input, load_model=None, k=1, sparse=True, std=1, random=False):
        if not random:
            model_dict = {
                'coef_sin': [4, np.pi],
                'coef_ma': np.array([[0, 16], [0, -20]]).reshape((2, 1, 2))}
            X = X_input[:, :2] if sparse else np.hstack((
                np.average(X_input[:, 0:X_input.shape[1]//2], axis=1).reshape((-1,1)), 
                np.average(X_input[:, X_input.shape[1]//2:], axis=1).reshape((-1,1))))

            Y = k * (model_dict['coef_sin'][0] * np.sin(model_dict['coef_sin'][1] * X[:, 0])).reshape((-1, 1)) + \
                k * np.max((model_dict['coef_ma']* np.ones((2, dim_y, X.shape[1]))) @ X.T, axis=0).T + np.random.normal(k * model_dict['coef_sin'][0] + 5, std, (X.shape[0], dim_y))
        else:
            if load_model is None:
                model_dict = {
                    'coef_sin': [np.random.uniform(3, 5, dim_y), np.random.uniform(0.8*np.pi, 1.2*np.pi, dim_y)],
                    'coef_ma': np.zeros((2, dim_y, 2))}
                model_dict['coef_ma'][0, :, 1] = np.random.uniform(10, 20, dim_y) # dim_piece, dim_y, dim_x
                model_dict['coef_ma'][1, :, 1] = np.random.uniform(-10, -30, dim_y)
            else:
                model_dict = load_model

            X = X_input[:, :2] if sparse else np.hstack((
                np.average(X_input[:, 0:X_input.shape[1]//2], axis=1).reshape((-1,1)), 
                np.average(X_input[:, X_input.shape[1]//2:], axis=1).reshape((-1,1))))
            Y = k * (model_dict['coef_sin'][0].reshape((1, -1)) * np.sin(model_dict['coef_sin'][1].reshape((1, -1)) * X[:, 0].reshape((-1, 1)))) + \
                k * np.max(model_dict['coef_ma'] @ X.T, axis=0).T + np.random.normal(k * model_dict['coef_sin'][0] + 5, std, (X.shape[0], dim_y))
            
        Y = Y * (Y > 0)
        return Y, model_dict        
    
    model_function = None
    # linear
    if model_name == 'linearsparse':
        model_function = lambda x, load_model_param: linear_model(x, load_model_param, sparse=True)
    elif model_name == 'lineardense':
        model_function = lambda x, load_model_param: linear_model(x, load_model_param, sparse=False)
    # MA3
    elif model_name == 'MA3k1dense':
        model_function = lambda x: MA3(x, k=1, sparse=False)
    elif model_name == 'MA3k3dense':
        model_function = lambda x: MA3(x, k=3, sparse=False)
    elif model_name == 'MA3k5dense':
        model_function = lambda x: MA3(x, k=5, sparse=False)
    elif model_name == 'MA3k7dense':
        model_function = lambda x: MA3(x, k=7, sparse=False)
    elif model_name == 'MA3k10dense':
        model_function = lambda x: MA3(x, k=10, sparse=False)
    elif model_name == 'MA3k15dense':
        model_function = lambda x: MA3(x, k=15, sparse=False)

    elif model_name == 'MA3k1sparse':
        model_function = lambda x: MA3(x, k=1, sparse=True)
    elif model_name == 'MA3k3sparse':
        model_function = lambda x: MA3(x, k=3, sparse=True)
    elif model_name == 'MA3k5sparse':
        model_function = lambda x: MA3(x, k=5, sparse=True)
    elif model_name == 'MA3k7sparse':
        model_function = lambda x: MA3(x, k=7, sparse=True)
    elif model_name == 'MA3k10sparse':
        model_function = lambda x: MA3(x, k=10, sparse=True)
    elif model_name == 'MA3k15sparse':
        model_function = lambda x: MA3(x, k=15, sparse=True)

    # SINPA
    elif model_name == 'SINPAk5sparse':
        model_function = lambda x, load_model_param: SINPA(x, load_model_param, k=5/5, sparse=True, random=False)
    elif model_name == 'SINPAk5dense':
        model_function = lambda x, load_model_param: SINPA(x, load_model_param, k=5/5, sparse=False, random=False)

    elif model_name == 'SINPAk1dense':
        model_function = lambda x, load_model_param: SINPA(x, load_model_param, k=1/5, sparse=False, random=False)
    elif model_name == 'SINPAk3dense':
        model_function = lambda x, load_model_param: SINPA(x, load_model_param, k=3/5, sparse=False, random=False)
    elif model_name == 'SINPAk7dense':
        model_function = lambda x, load_model_param: SINPA(x, load_model_param, k=7/5, sparse=False, random=False)
    elif model_name == 'SINPAk10dense':
        model_function = lambda x, load_model_param: SINPA(x, load_model_param, k=10/5, sparse=False, random=False)
    elif model_name == 'SINPAk15dense':
        model_function = lambda x, load_model_param: SINPA(x, load_model_param, k=15/5, sparse=False, random=False)

    elif model_name == 'SINPAk5denseRandom':
        model_function = lambda x, load_model_param: SINPA(x, load_model_param, k=5/5, sparse=False, random=True)
    elif model_name == 'SINPAk5sparseRandom':
        model_function = lambda x, load_model_param: SINPA(x, load_model_param, k=5/5, sparse=True, random=True)

    elif model_name == 'SINPAk1denseRandom':
        model_function = lambda x, load_model_param: SINPA(x, load_model_param, k=1/5, sparse=False, random=True)
    elif model_name == 'SINPAk3denseRandom':
        model_function = lambda x, load_model_param: SINPA(x, load_model_param, k=3/5, sparse=False, random=True)
    elif model_name == 'SINPAk7denseRandom':
        model_function = lambda x, load_model_param: SINPA(x, load_model_param, k=7/5, sparse=False, random=True)
    elif model_name == 'SINPAk10denseRandom':
        model_function = lambda x, load_model_param: SINPA(x, load_model_param, k=10/5, sparse=False, random=True)
    elif model_name == 'SINPAk15denseRandom':
        model_function = lambda x, load_model_param: SINPA(x, load_model_param, k=15/5, sparse=False, random=True)
        
    else:
        raise ValueError('Invalid model name.')
    
    model_info = inspect.getsource(model_function)

    if model_name[:3] == 'MA3':
        Y_output, model_param_dict = model_function(X)
    else:
        Y_output, model_param_dict = model_function(X, load_model_param)

    if output_info and output_model_param:
        return Y_output, model_info, model_param_dict
    elif output_info:
        return Y_output, model_info
    elif output_model_param:
        return Y_output, model_param_dict
    else:
        return Y_output

def savedata(train_X, train_Y, val_X, val_Y, test_X, test_Y, model_info, save_addr, W=None, model_param_dict=None):
    save_addr = check_folder_syntaxandexistence(save_addr)
    os.makedirs(save_addr)

    tr_size = train_X.shape[0]
    vl_size = val_X.shape[0]
    test_size = test_X.shape[0]
    assert train_X.shape[1] == val_X.shape[1] == test_X.shape[1]
    dim_x = train_X.shape[1]
    dim_y = train_Y.shape[1] if len(train_Y.shape) > 1 else 1

    # log data information
    data_info_text = f'tr_size={tr_size:d}, vl_size={vl_size:d}, test_size={test_size:d}, \ndim_y={dim_y:d}, dim_x={dim_x:d}'
    if W is not None:
        data_info_text += f',\nW: node_dim={W.shape[0]:d}, arc_dim={W.shape[1]:d}'
    data_info_text += f'\nData model: \n{model_info}'
    write_to_file(save_addr + 'data_info.log', data_info_text)

    # save arrays
    data_dict = {
        'train_X': train_X,
        'train_Y': train_Y,
        'val_X': val_X,
        'val_Y': val_Y,
        'test_X': test_X,
        'test_Y': test_Y,
    }
    if W is not None:
        data_dict['W'] = W
    np.savez(save_addr + 'data.npz', **data_dict)
    if model_param_dict is not None:
        np.savez(save_addr + 'data_model.npz', **model_param_dict)


def data_generator(tr_size, vl_size, test_size, problem_setting, data_model_name, loop_dict, data_suffix='', arc_dim=None, save_data=False, **kwargs_list_dict):
    """
    Generate data for the newsvendor problem or the product placement problem.
    ----------
    problem_setting: 'nv', 'pp', 'nv2prod'
    """
    if problem_setting == 'nv':
        loop_dict['ydim'] = 1
    
    loopParams_dict, numTotal = load_loop_params(loop_dict, 'xdim', 'ydim')
    for idx_loop in range(numTotal):
        xDim, yDim = pick_loop_params(idx_loop, numTotal, loopParams_dict)
        assert isinstance(yDim, int)
        kwargs_dict = {}
        for key, value in kwargs_list_dict.items():
            kwargs_dict[key] = value[idx_loop]
        
        X = np.random.uniform(-1, 1, (tr_size + vl_size + test_size, xDim))

        Y, model_info, model_param_dict = true_data_model(X, yDim, model_name=data_model_name, output_info=True, output_model_param=True, **kwargs_dict)
        W = None
        if problem_setting == 'pp':
            W = generateW(yDim, arc_dim)
            model_info += f'\nW: \n{W}'

        if save_data:
            data_suffix = f'_{data_suffix}' if data_suffix != '' else ''
            save_addr = f'./data/{problem_setting}/{data_model_name}{data_suffix}/ydim{yDim}arcdim{arc_dim}/xdim{xDim}/data_{unique_stamp()}'
            if arc_dim is None:
                save_addr = f'./data/{problem_setting}/{data_model_name}{data_suffix}/ydim{yDim}/xdim{xDim}/data_{unique_stamp()}'
            savedata(X[:tr_size], Y[:tr_size], X[tr_size:tr_size+vl_size], Y[tr_size:tr_size+vl_size], X[tr_size+vl_size:], Y[tr_size+vl_size:], model_info, save_addr, W=W, model_param_dict=model_param_dict)