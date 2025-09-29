import numpy as np

board = np.array([np.array(range(1, 10)) for _ in range(9)])


def get_fitness(board: np.array) -> int:
    return -np.sum([len(set(board[:, i])) != 9 for i in range(9)])


for row in board:
    print(row)

print(get_fitness(board))
