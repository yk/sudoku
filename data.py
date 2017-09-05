#!/usr/bin/env python3
import numpy as np
import sudoku
from concurrent.futures import ProcessPoolExecutor
import os


def generate(i):
    return sudoku.generate()


def load(num_files=1000):
    games = []
    for i in range(num_files):
        fn = (os.path.join(os.path.expanduser('~'), 'data', 'sudoku', 'games_{}.npy'.format(i)))
        gs = np.load(fn)
        games.extend(gs)
    return np.asarray(games)



def main():
    games_per_file = 1000
    num_files = 1000

    base_path = os.path.join(os.path.expanduser('~'), 'data', 'sudoku')
    os.makedirs(base_path, exist_ok=True)

    tpe = ProcessPoolExecutor()

    for fi in range(num_files):
        fn = os.path.join(base_path, 'games_{}.npy'.format(fi))
        print(fn)
        if not os.path.exists(fn):
            res = list(tpe.map(generate, range(games_per_file)))
            res = np.asarray(res)
            np.save(fn, res)


if __name__ == '__main__':
    main()
