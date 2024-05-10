import numpy as np
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
COOLING = 0.9
LOWER_LIMIT = 1
UPPER_LIMIT = 3

def cooling(t):
    return COOLING * t

def Simulated_Annealing(f, lims, n, range_lim = True):
    dim, lb, ub = 5, np.array([0,0,0,0,0]), np.array([1,1,1,1,1])
    range_orig = np.array([lb, ub])
    best_fit_hist = np.ones(lims) * np.inf
    global_best = {'Fitness': np.inf,'Sol': None}
    t_initial = 0.05
    t_final = 1e-6

    for k in tqdm(range(1, lims + 1)):
        xo, zo, global_best = init(n, range_orig, dim, f)
        best_fit_hist[k-1] = global_best['Fitness'] 
        xn = np.zeros_like(xo)
        zn = np.zeros_like(zo)
        t = t_initial
        finished = False

        while not finished:
            for j in range(n):
                if range_lim:
                    xn[j, :] = xo[j, :] + np.random.randn(dim) * 5 * t
                else:
                    xn[j, :] = xo[j, :] + np.random.randn(dim) * 0.01
                xn[j, :] = find_range(xn[j, :], lb, ub)
                zn[j] = f.output(xn[j, :])
                delta = zn[j] - zo[j]

                if range_lim:
                    prob = np.exp(delta / t)
                else:
                    prob = np.exp(delta / t * 20)

                if delta > 0:
                    xo[j, :] = xn[j, :]
                    zo[j] = zn[j]
                elif delta < 0 and prob > np.random.rand():
                    xo[j, :] = xn[j, :]
                    zo[j] = zn[j]

                if zo[j] > global_best['Fitness']:
                    global_best['Fitness'] = zo[j]
                    global_best['Sol'] = xo[j, :]
            t = cooling(t)
            if t < t_final:
                finished = True

            best_fit_hist[k-1] = global_best['Fitness']

        if range_lim:
            range_orig, lb, ub = range_limitation(k,lims, lb, ub, global_best)
        
    best_fit = global_best['Fitness']
    best_sol = global_best['Sol']

    return best_fit, best_fit_hist, best_sol

def range_limitation(k, lims, lb, ub, global_best):
    if k < lims / 2:
        limit_range = LOWER_LIMIT + k * (UPPER_LIMIT - LOWER_LIMIT) / (lims / 2)
    else:
        limit_range = UPPER_LIMIT
    
    lb = lb / limit_range
    ub = ub / limit_range

    lim = (ub - lb) / 2
    return np.array([global_best['Sol'] - lim, global_best['Sol'] + lim]), lb, ub


def init(n, range_orig, dim, f):
    xo = np.random.rand(n, dim) * (range_orig[1] - range_orig[0]) + range_orig[0]
    zo = np.zeros(n)
    for j in range(n):
        xo[j, :] = find_range(xo[j, :], range_orig[0], range_orig[1])
        zo[j] = f.output(xo[j, :])
    best_idx = np.argmin(zo)   
    global_best = {
        'Sol': xo[best_idx, :],
        'Fitness': zo[best_idx]
    }

    return xo, zo, global_best

def find_range(xo, lb, ub):
    for i in range(len(xo)):
        xo[i] = min(max(xo[i],lb[i]),ub[i])
    return xo

