import numpy as np
import pandas as pd
import plotly.express as px


def get_fitness(board: np.array) -> int:
    return -np.sum([len(set(board[:, i])) != 9 for i in range(9)])


def swap(board: np.array) -> np.array:
    row = np.random.randint(0, 9)

    idx1 = np.random.randint(0, 9)
    idx2 = np.random.randint(0, 9)
    num1 = board[row, idx1]
    num2 = board[row, idx2]

    board[row, idx1] = num2
    board[row, idx2] = num1

    return board


def main(board: np.array) -> tuple[np.array, np.array]:
    iter = 0
    max_iter = 30000
    fitnesses = np.array([])
    current_fitness = get_fitness(board)
    while current_fitness < 0 and iter < max_iter:
        new_board = board.copy()
        new_board = swap(new_board)
        new_fitness = get_fitness(new_board)

        if new_fitness >= current_fitness:
            board = new_board
            current_fitness = new_fitness
            print(f"Iteration {iter}: neue Fitness: {current_fitness}")

        iter += 1
        fitnesses = np.append(fitnesses, current_fitness)

    return board, fitnesses


if __name__ == "__main__":
    board = np.array([np.array(range(1, 10)) for _ in range(9)])

    board, fitnesses = main(board)

    for row in board:
        print(row)

    print(get_fitness(board))

    fitness_df = pd.DataFrame(fitnesses, columns=["fitness"])
    fig = px.line(
        fitness_df, y="fitness", labels={"index": "Iteration", "fitness": "Fitness"}
    )
    fig.show()
