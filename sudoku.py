#!/usr/bin/env python3
import numpy as np
from numpy.lib.arraysetops import unique
import itertools as itt
import random

SIDE = 9
SIDE_ROOT = 3

def solve(s, max_solutions=1):
    idcs_x, idcs_y = np.where(s == 0)
    num_empty = len(idcs_x)
    if num_empty == 0:
        return [s]
    props = [np.random.choice(SIDE, SIDE, False) + 1 for _ in range(num_empty)]
    search_idcs = [0 for _ in props]
    i = 0
    sols = []
    while i >= 0:
        x, y = idcs_x[i], idcs_y[i]
        prop = props[i]
        si = search_idcs[i]
        while si < len(prop):
            p = prop[si]
            s[x, y] = p
            if check_consistent(s, (x, y)):
                search_idcs[i] = si + 1
                if np.all(s):
                    sols.append(np.copy(s))
                    if len(sols) >= max_solutions and max_solutions > 0:
                        return sols
                else:
                    i += 1
                break
            si += 1
        if si >= len(prop):
            search_idcs[i] = 0
            s[x, y] = 0
            i -= 1
    return sols


def make_empty(s, empty=1):
    s_shape = s.shape
    s = np.copy(s)
    s = s.ravel()
    s[np.random.choice(len(s), empty, False)] = 0
    s = s.reshape(s_shape)
    return s

def generate(empty=0):
    s = np.zeros((SIDE, SIDE), np.int32)
    s = solve(s)[0]
    if empty > 0:
        s = make_empty(s, empty)
    return s

def check_consistent_unit(u):
    u = u.ravel()
    uu, c = unique(u, return_counts=True)
    if uu[0] == 0:
        c = c[1:]
    res = np.all(c == 1)
    return res

def random_move(s):
    inds_x, inds_y = np.where(s == 0)
    if len(inds_x) == 0:
        raise Exception('No move possible')
    i = np.random.choice(len(inds_x))
    x, y = inds_x[i], inds_y[i]
    s = np.copy(s)
    s[x, y] = np.random.randint(1, SIDE+1)
    return s, (x, y)

def check_all_filled(s):
    return np.all(s)

def check_solved(s):
    return check_consistent(s) and check_all_filled(s)

def check_consistent(s, idx=None):
    if idx is not None:
        if not check_consistent_unit(s[idx[0], :]):
            return False
        if not check_consistent_unit(s[:, idx[1]]):
            return False
        i, j = idx[0] // SIDE_ROOT, idx[1] // SIDE_ROOT
        if not check_consistent_unit(s[i*SIDE_ROOT:(i+1)*SIDE_ROOT,j*SIDE_ROOT:(j+1)*SIDE_ROOT]):
            return False
    else:
        for i in range(SIDE):
            if not check_consistent_unit(s[i, :]):
                return False
            if not check_consistent_unit(s[:, i]):
                return False
        for i in range(SIDE_ROOT):
            for j in range(SIDE_ROOT):
                if not check_consistent_unit(s[i*SIDE_ROOT:(i+1)*SIDE_ROOT,j*SIDE_ROOT:(j+1)*SIDE_ROOT]):
                    return False
    return True
    


if __name__ == '__main__':
    for _ in range(100):
        for i in range(60):
            s = np.mean([len(solve(generate(i), -1)) for _ in range(100)])
            print(i, s)
