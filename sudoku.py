#!/usr/bin/env python3
import numpy as np
from numpy.lib.arraysetops import unique
import itertools as itt
import random

def generate():
    s = np.zeros((9, 9), np.int)
    stack = [s]
    idcs = list(itt.product(range(9), range(9)))
    props = [np.random.choice(9, 9, False) + 1 for _ in range(9*9)]
    search_idcs = [0 for _ in props]
    i = 0
    while i < len(idcs):
        idx = idcs[i]
        sn = np.copy(stack[-1])
        prop = props[i]
        si = search_idcs[i]
        while si < len(prop):
            p = prop[si]
            sn[idx] = p
            if check_consistent(sn, idx):
                stack.append(sn)
                search_idcs[i] = si + 1
                i += 1
                break
            si += 1
        if si >= len(prop):
            search_idcs[i] = 0
            stack.pop()
            i -= 1
    return stack[-1]

def check_consistent_unit(u):
    u = u.ravel()
    uu, c = unique(u, return_counts=True)
    if uu[0] == 0:
        c = c[1:]
    res = np.all(c == 1)
    return res

def check_consistent(s, idx=None):
    if idx is not None:
        if not check_consistent_unit(s[idx[0], :]):
            return False
        if not check_consistent_unit(s[:, idx[1]]):
            return False
        i, j = idx[0] // 3, idx[1] // 3
        if not check_consistent_unit(s[i*3:(i+1)*3,j*3:(j+1)*3]):
            return False
    else:
        for i in range(9):
            if not check_consistent_unit(s[i, :]):
                return False
            if not check_consistent_unit(s[:, i]):
                return False
        for i in range(3):
            for j in range(3):
                if not check_consistent_unit(s[i*3:(i+1)*3,j*3:(j+1)*3]):
                    return False
    return True
    


if __name__ == '__main__':
    for _ in range(100):
        s = generate()
        c = check_consistent(s)
        print(s, c)
