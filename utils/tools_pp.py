import numpy as np
import gurobipy as grb

def build_pp_grb_model(Y, g, b, h, W):
    node_dim, arc_dim = b.shape[0], g.shape[0]
    n = Y.shape[0]
    m = grb.Model('pp_saa')
    m.Params.LogToConsole = 0  # do not output the log info
    z = m.addMVar(node_dim, lb=0, name='z')
    f = m.addMVar((arc_dim, n), lb=0, name='f')
    T = m.addMVar((n, 1), name='T')

    for i in range(n):
        m.addConstr(T[i] >= Y[i, :] @ b + (g + b @ W) @ f[:, i] - b @ z)
        m.addConstr(T[i] >= - Y[i, :] @ h + (g - h @ W) @ f[:, i] + h @ z)
        # m.addConstr(W @ f[:, i] <= z)
    
    return m, z, T

def opt_pp_grb_model(m, z, T, c, weight_arr):
    m.setObjective(c @ z + weight_arr @ T[:, 0])
    m.optimize()

    return z.X, m.ObjVal

def solve_pp_saa(weight_arr, Y, c, g, b, h, W):
    m, z, T = build_pp_grb_model(Y, g, b, h, W)
    z_solution, _ = opt_pp_grb_model(m, z, T, c, weight_arr)
    return z_solution

def pp_cost(c, g, b, h, W, z, Y, weights=None):
    """
    Compute the cost of the product placement problem.
    """
    n, arc_dim = Y.shape[0], W.shape[1]
    # z = z * (z >= 0)
    m = grb.Model('pp_subproblem')
    m.Params.LogToConsole = 0  # do not output the log info
    f = m.addMVar((arc_dim, n), lb=0, name='f')
    T = m.addMVar((n, 1), name='T')
    weights = np.ones(n) / n if weights is None else weights
    
    for i in range(n):
        m.addConstr(T[i] >= Y[i, :] @ b + (g + b @ W) @ f[:, i] - b @ z[i, :])
        m.addConstr(T[i] >= - Y[i, :] @ h + (g - h @ W) @ f[:, i] + h @ z[i, :])
        # m.addConstr(W @ f[:, i] <= z[i, :])

    m.setObjective(weights @ (z @ c) + weights @ T[:, 0])
    m.optimize()
    return m.objVal

def generateW(dim_node, dim_arc):
    """
    Generate a matrix W that maps the arc weights to the node weights. The first dim_node arcs connect a circular graph, and the remaining dim_arc-dim_node arcs are randomly connected.
    ----------
    dim_node : int, the number of nodes in the graph.
    dim_arc : int, the number of arcs in the graph.
    ----------
    return : np.array, the node-arc matrix W with shape (dim_node, dim_arc).
    """
    assert dim_arc <= dim_node * (dim_node - 1) and dim_arc >= dim_node
    
    W = np.zeros((dim_node, dim_arc))
    for i in range(dim_node):
        W[i, i%dim_node] = 1
        W[(i+1)%dim_node, i] = -1
    
    remaining_idx_arr = np.sort(np.random.choice(np.arange(dim_node*(dim_node-2)), dim_arc-dim_node, replace=False))    
    counter = dim_node
    for remaining_idx in remaining_idx_arr:
        i = remaining_idx // (dim_node-2)
        j = remaining_idx % (dim_node-2) # the remaining connections do not include i->i or i->i+1
        W[i, counter] = 1
        W[(i+j+2)%dim_node, counter] = -1
        counter += 1

    return W