import numpy as np
import gurobipy as grb


# Functions for the newsvendor problem
def build_nv_grb_model(Y):
    n = Y.shape[0]
    m = grb.Model('nv_saa')
    m.Params.LogToConsole = 0  # do not output the log info
    z = m.addMVar(1, lb=0, name='z')
    vB = m.addMVar(n, lb=0, name='arti-var for penalty cost') # auxiliary variables
    vH = m.addMVar(n, lb=0, name='arti-var for holding cost')
    
    for i in range(n):
        m.addConstr(vB[i] >= Y[i] - z[0])
        m.addConstr(vH[i] >= z[0] - Y[i])
    return m, z, vB, vH

def opt_nv_grb_model(m, z, vB, vH, cb, ch, weight_arr):
    m.setObjective((weight_arr * cb) @ vB + (weight_arr * ch) @ vH)
    m.optimize()
    return z.X[0], m.ObjVal

def solve_nv_saa(weight_arr, Y, cb, ch):
    m, z, vB, vH = build_nv_grb_model(Y)
    z_solution, _ = opt_nv_grb_model(m, z, vB, vH, cb, ch, weight_arr)
    return z_solution

def nv_cost(cb, ch, z, Y, average=True, weights=None):
    """
    Compute the cost of the newsvendor problem.
    z could be an array that has two dims and the second dim aligned with Y
    """
    cost = ch*np.maximum(z - Y, 0) + cb*np.maximum(Y - z, 0)
    if average:
        weights = np.ones(Y.shape[0]) / Y.shape[0] if weights is None else weights
        return np.sum(weights * cost)
    else:
        return cost


# Functions for the 2-product constrained newsvendor problem
def build_nv2prod_grb_model(Y, capacity):
    assert Y.shape[1] == 2
    n = Y.shape[0]
    m = grb.Model('nv2prod_saa')
    m.Params.LogToConsole = 0  # do not output the log info
    z1 = m.addMVar(1, lb=0, name='z1')
    z2 = m.addMVar(1, lb=0, name='z2')
    vB1 = m.addMVar(n, lb=0, name='arti-var for penalty cost 1') # auxiliary variables
    vH1 = m.addMVar(n, lb=0, name='arti-var for holding cost 1')
    vB2 = m.addMVar(n, lb=0, name='arti-var for penalty cost 2')
    vH2 = m.addMVar(n, lb=0, name='arti-var for holding cost 2')

    for i in range(n):
        m.addConstr(vB1[i] >= Y[i][0] - z1[0])
        m.addConstr(vH1[i] >= z1[0] - Y[i][0])
        m.addConstr(vB2[i] >= Y[i][1] - z2[0])
        m.addConstr(vH2[i] >= z2[0] - Y[i][1])
    
    m.addConstr(z1 + z2 <= capacity)

    return m, (z1, z2), (vB1, vB2), (vH1, vH2)

def opt_nv2prod_grb_model(m, z_tup, vB_tup, vH_tup, cb_tup, ch_tup, weight_arr):
    z1, z2 = z_tup
    vB1, vB2 = vB_tup
    vH1, vH2 = vH_tup
    cb1, cb2 = cb_tup
    ch1, ch2 = ch_tup
    m.setObjective((weight_arr * cb1) @ vB1 + (weight_arr * ch1) @ vH1 + (weight_arr * cb2) @ vB2 + (weight_arr * ch2) @ vH2)
    m.optimize()
    return np.array([z1.X[0], z2.X[0]]), m.ObjVal

def solve_nv2prod_saa(weight_arr, Y, cb1, cb2, ch1, ch2, capacity):
    m, z_tup, vB_tup, vH_tup = build_nv2prod_grb_model(Y, capacity)
    z_solution, _ = opt_nv2prod_grb_model(m, z_tup, vB_tup, vH_tup, (cb1, cb2), (ch1, ch2), weight_arr)
    return z_solution

# Functions for the newsvendor problem with a nonconvex capacity cost
def capacost_function_oneprod(z, capacost_array=None, average=True, weights=None):
    """
    Nonconvex capacity cost given order decision z and parameters of the capacity cost.
    """
    if capacost_array is not None:
        z_array = np.hstack((z.reshape((-1, 1)), np.ones((z.shape[0], 1))))
        res = - np.max(capacost_array.dot(z_array.T), axis=0)
        if not average:
            return res
        else:
            weights = np.ones(z.shape[0]) / z.shape[0] if weights is None else weights
            return np.sum(weights * res)
    else:
        return 0

def solve_ncvx_nv_saa(weight_arr, Y, cb, ch, capacity_arr):
    assert capacity_arr is not None
    z_range = np.linspace(0, np.max(Y), 10000)
    capacity_cost_range = capacost_function_oneprod(z_range, capacity_arr) # (z_dim, )
    nv_cost_range = np.average(nv_cost(cb, ch, z_range.reshape((-1, 1)) * np.ones_like(Y).reshape((1, -1)), Y, average=False), axis=1, weights=weight_arr) # (z_dim, )
    total_cost_range = nv_cost_range + capacity_cost_range
    return z_range[np.argmin(total_cost_range)]

def feasible_2prodnv_cost(cb_list, ch_list, capacity, z_arr, Y_arr, weight_arr=None, constr_arr=None):
    """
    For the constrained newsvendor problem with the capacity constraint z.sum <= capacity, project the decision z_arr to the feasible region if overflow and compute the frequency of feasibility of the original decision.
    If the constraint is concave (`constr_arr` is not None), then compute the costs of feasible decisions.
    """
    if constr_arr is None:
        assert z_arr.shape[1] == Y_arr.shape[1] == len(cb_list) == len(ch_list)
        z_arr = z_arr * (z_arr >= 0)
        z_overflow = np.sum(z_arr, axis=1) - capacity
        z_overflow = z_overflow * (z_overflow >= 0)
        real_z_arr = z_arr - (z_overflow/z_arr.shape[1]).reshape((-1, 1)) # projection if overflow
        cost = np.sum([nv_cost(cb_list[idx], ch_list[idx], real_z_arr[:, idx], Y_arr[:, idx], weights=weight_arr) for idx in range(z_arr.shape[1])])
        weight_arr = np.ones(z_arr.shape[0]) / z_arr.shape[0] if weight_arr is None else weight_arr
        return cost, np.sum(weight_arr * (z_overflow <= 0))
    else:
        assert z_arr.shape[1] == Y_arr.shape[1] == len(cb_list) == len(ch_list) == constr_arr.shape[1]
        capacity_cost = np.sum([capacost_function_oneprod(z_arr[:, idx], capacost_array=constr_arr, average=False) for idx in range(z_arr.shape[1])], axis=0)
        feasible_indices = (capacity_cost - capacity - 1e-8) <= 0

        if np.sum(feasible_indices)>0:
            weight_arr = np.ones(z_arr.shape[0]) / z_arr.shape[0] if weight_arr is None else weight_arr
            cost = np.sum([nv_cost(cb_list[idx], ch_list[idx], z_arr[feasible_indices, idx], Y_arr[feasible_indices, idx], weights=weight_arr[feasible_indices]) for idx in range(z_arr.shape[1])])
            return cost, np.sum(weight_arr[feasible_indices])
        else:
            return 1e10, 0