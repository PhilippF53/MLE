import numpy as np
import pandas as pd
import plotly.express as px


def get_fitness(board: np.array) -> int:
    col_errors = np.sum([len(set(board[:, i])) != 9 for i in range(9)])
    box_errors = 0
    for row_start in range(0, 9, 3):
        for col_start in range(0, 9, 3):
            box = board[row_start : row_start + 3, col_start : col_start + 3].flatten()
            if len(set(box)) != 9:
                box_errors += 1

    return -(col_errors + box_errors)


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
    max_iter = 40000
    fitnesses = np.array([])
    current_fitness = get_fitness(board)
    temperature = 1.0
    cooling_rate = 0.99
    while current_fitness < 0 and iter < max_iter:
        new_board = board.copy()
        new_board = swap(new_board)
        new_fitness = get_fitness(new_board)

        if new_fitness > current_fitness:
            board = new_board
            current_fitness = new_fitness
        else:
            p = np.exp((new_fitness - current_fitness) / temperature)
            if p > np.random.rand():
                board = new_board
                current_fitness = new_fitness
        temperature *= cooling_rate

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
