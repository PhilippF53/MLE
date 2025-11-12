import numpy as np
import pygame
import random
import math
import json
from collections import deque, defaultdict


# Initialize pygame
pygame.init()

# Define constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400
CELL_SIZE = 40

# Define colors
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

# Labyrinth as a string
labyrinth = ["##########", "#........#", "#.##..##.#", "#........#", "##########"]
WALL = "#"
COOKIE = "."

# Get labyrinth dimensions
ROWS = len(labyrinth)
COLS = len(labyrinth[0])

# Initialize game screen
screen = pygame.display.set_mode((COLS * CELL_SIZE, ROWS * CELL_SIZE))
pygame.display.set_caption("Micro-Pacman")


# Train Parameter
epochs = 10000
train_steps = 400
alpha = 0.05
gamma = 0.8
epsilon = 1.00
epsilon_min = 0.05
epsilon_decay = 0.999

actions = ["UP", "DOWN", "LEFT", "RIGHT", "NONE"]


# Pacman class
class Pacman:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.count = 0

    def move(self, dx, dy):
        new_x, new_y = self.x + dx, self.y + dy
        if labyrinth[new_y][new_x] != "#":
            self.x = new_x
            self.y = new_y

    def draw(self):
        radius = CELL_SIZE // 2 - 4
        start_angle = math.pi / 6
        end_angle = -math.pi / 6
        pygame.draw.circle(
            screen,
            YELLOW,
            (self.x * CELL_SIZE + CELL_SIZE // 2, self.y * CELL_SIZE + CELL_SIZE // 2),
            CELL_SIZE // 2 - 4,
        )
        # Calculate the points for the mouth
        start_pos = (
            self.x * CELL_SIZE
            + CELL_SIZE // 2
            + int(radius * 1.3 * math.cos(start_angle)),
            self.y * CELL_SIZE
            + CELL_SIZE // 2
            - int(radius * 1.3 * math.sin(start_angle)),
        )
        end_pos = (
            self.x * CELL_SIZE
            + CELL_SIZE // 2
            + int(radius * 1.3 * math.cos(end_angle)),
            self.y * CELL_SIZE
            + CELL_SIZE // 2
            - int(radius * 1.3 * math.sin(end_angle)),
        )
        self.count += 1
        if self.count % 2 == 0:
            # Draw the mouth by filling a polygon
            pygame.draw.polygon(
                screen,
                BLACK,
                [
                    (
                        self.x * CELL_SIZE + CELL_SIZE // 2,
                        self.y * CELL_SIZE + CELL_SIZE // 2,
                    ),
                    start_pos,
                    end_pos,
                ],
            )


# Ghost class with pixel art
class Ghost:
    # Define the pixel art for the ghost using strings
    ghost_pixels = [" #### ", "######", "## # #", "######", "######", "# # # "]

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def move_towards_pacman(self, pacman):
        if self.x < pacman.x and labyrinth[self.y][self.x + 1] != "#":
            self.x += 1
        elif self.x > pacman.x and labyrinth[self.y][self.x - 1] != "#":
            self.x -= 1
        elif self.y < pacman.y and labyrinth[self.y + 1][self.x] != "#":
            self.y += 1
        elif self.y > pacman.y and labyrinth[self.y - 1][self.x] != "#":
            self.y -= 1

    def draw(self):
        pixel_size = CELL_SIZE // len(
            self.ghost_pixels
        )  # Size of each pixel in the ghost art
        for row_idx, row in enumerate(self.ghost_pixels):
            for col_idx, pixel in enumerate(row):
                if pixel == "#":
                    pixel_x = self.x * CELL_SIZE + col_idx * pixel_size
                    pixel_y = self.y * CELL_SIZE + row_idx * pixel_size
                    pygame.draw.rect(
                        screen, RED, (pixel_x, pixel_y, pixel_size, pixel_size)
                    )


# Draw walls and cookies
def draw_labyrinth():
    for y, row in enumerate(labyrinth):
        for x, cell in enumerate(row):
            if cell == "#":
                pygame.draw.rect(
                    screen, BLUE, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                )
            elif cell == ".":
                pygame.draw.circle(
                    screen,
                    WHITE,
                    (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2),
                    5,
                )


def is_wall_at(x: int, y: int, labyrinth: list) -> bool:
    if x < 0 or x >= COLS or y < 0 or y >= ROWS:
        return True
    return labyrinth[y][x] == WALL


def get_state(pacman: Pacman, ghost: Ghost, labyrinth: list):
    pacman_x, pacman_y = pacman.x, pacman.y
    ghost_x, ghost_y = ghost.x, ghost.y
    distance_cookie_x, distance_cookie_y = get_nearest_cookie(
        pacman_x, pacman_y, labyrinth
    )

    dir_ghost_x, dir_ghost_y = int(np.sign(ghost_x - pacman_x)), int(
        np.sign(ghost_y - pacman_y)
    )
    distance_ghost_x = min(3, abs(ghost_x - pacman_x))
    distance_ghost_y = min(3, abs(ghost_y - pacman_y))

    return (
        pacman_x,
        pacman_y,
        dir_ghost_x,
        dir_ghost_y,
        distance_ghost_x,
        distance_ghost_y,
        distance_cookie_x,
        distance_cookie_y,
    )


def get_nearest_cookie(pacman_x, pacman_y, labyrinth):
    cookies = [
        (x, y)
        for y, row in enumerate(labyrinth)
        for x, cell in enumerate(row)
        if cell == "."
    ]
    if cookies:
        nearest = min(
            cookies, key=lambda c: abs(c[0] - pacman_x) + abs(c[1] - pacman_y)
        )
        cookie_dx = int(np.sign(nearest[0] - pacman_x))
        cookie_dy = int(np.sign(nearest[1] - pacman_y))
    else:
        cookie_dx, cookie_dy = 0, 0

    return cookie_dx, cookie_dy


def epsilon_greedy(state):
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        q_values = q_table[state]
        return actions[int(np.argmax(q_values))]


def take_action(action_taken, labyrinth, pacman: Pacman, ghost: Ghost):
    if pacman.x == ghost.x and pacman.y == ghost.y:
        return get_state(pacman, ghost, labyrinth), -100

    move_x, move_y = pacman.x, pacman.y
    if action_taken == "UP":
        move_y -= 1
    elif action_taken == "DOWN":
        move_y += 1
    elif action_taken == "LEFT":
        move_x -= 1
    elif action_taken == "RIGHT":
        move_x += 1

    if is_wall_at(move_x, move_y, labyrinth):
        return get_state(pacman, ghost, labyrinth), -10  # lighter penalty

    if move_x == ghost.x and move_y == ghost.y:
        return get_state(pacman, ghost, labyrinth), -100

    pacman.x, pacman.y = move_x, move_y

    if labyrinth[move_y][move_x] == COOKIE:
        labyrinth[move_y] = (
            labyrinth[move_y][:move_x] + " " + labyrinth[move_y][move_x + 1 :]
        )
        if any(COOKIE in row for row in labyrinth):
            return get_state(pacman, ghost, labyrinth), +10  # strong positive
        else:
            return get_state(pacman, ghost, labyrinth), +100  # finishing bonus

    # Small living penalty to encourage efficiency
    return get_state(pacman, ghost, labyrinth), -1


def train(q_table):
    global epsilon
    print("started training")
    for epoch in range(epochs):
        print(f"Epoche {epoch}/{epochs}") if epoch % 1000 == 0 else None
        labyrinth = [
            "##########",
            "#........#",
            "#.##..##.#",
            "#........#",
            "##########",
        ]
        pacman = Pacman(1, 1)
        ghost = Ghost(COLS - 2, ROWS - 2)
        state = get_state(pacman, ghost, labyrinth)

        for step in range(train_steps):
            action = epsilon_greedy(state)
            ghost.move_towards_pacman(pacman)
            new_state, reward = take_action(action, labyrinth, pacman, ghost)

            old_value = q_table[state][actions.index(action)]
            next_max = np.max(q_table[new_state])
            new_value = old_value + alpha * ((reward + gamma * next_max) - old_value)
            q_table[state][actions.index(action)] = new_value

            state = new_state

            if ghost.x == pacman.x and ghost.y == pacman.y:
                # print(f"caught at {step}")
                break

            if labyrinth[pacman.y][pacman.x] == ".":
                labyrinth[pacman.y] = (
                    labyrinth[pacman.y][: pacman.x]
                    + " "
                    + labyrinth[pacman.y][pacman.x + 1 :]
                )

            if not any(COOKIE in row for row in labyrinth):
                # print("gotem")
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print("training finished")
    return q_table


def load_qtable():
    with open("./assets/q_table.json", "r") as f:
        loaded = json.load(f)
        return {tuple(map(int, k.split(","))): np.array(v) for k, v in loaded.items()}


def dump_qtable(q_table: dict):
    json_compatible = {",".join(map(str, k)): v.tolist() for k, v in q_table.items()}

    with open("./assets/q_table.json", "w") as f:
        json.dump(json_compatible, f, indent=2)


# Main game function
def main(q_table):
    clock = pygame.time.Clock()

    # Initialize Pacman and Ghost positions
    pacman = Pacman(1, 1)
    ghost = Ghost(COLS - 2, ROWS - 2)
    state = get_state(pacman, ghost, labyrinth)

    # Game loop
    running = True
    iter = 0
    while running:
        screen.fill(BLACK)
        iter = iter + 1
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Handle Pacman movement
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            pacman.move(-1, 0)
        if keys[pygame.K_RIGHT]:
            pacman.move(1, 0)
        if keys[pygame.K_UP]:
            pacman.move(0, -1)
        if keys[pygame.K_DOWN]:
            pacman.move(0, 1)

        if iter % 3 == 0:
            action = epsilon_greedy(state)

            # Ghost moves towards Pacman
            ghost.move_towards_pacman(pacman)

            new_state, reward = take_action(action, labyrinth, pacman, ghost)

            old_value = q_table[state][actions.index(action)]
            next_max = np.max(q_table[new_state])
            new_value = old_value + alpha * ((reward + gamma * next_max) - old_value)
            q_table[state][actions.index(action)] = new_value

            state = new_state

        # Check for collisions (game over if ghost catches pacman)
        if pacman.x == ghost.x and pacman.y == ghost.y:
            print("Game Over! The ghost caught Pacman.")
            running = False
            return False

        # Eat cookies
        if labyrinth[pacman.y][pacman.x] == ".":
            labyrinth[pacman.y] = (
                labyrinth[pacman.y][: pacman.x]
                + " "
                + labyrinth[pacman.y][pacman.x + 1 :]
            )

        # Check if all cookies are eaten (game over)
        if all("." not in row for row in labyrinth):
            print("You Win! Pacman ate all the cookies.")
            running = False
            return True

        # Draw the labyrinth, pacman, and ghost
        draw_labyrinth()
        pacman.draw()
        ghost.draw()

        # Update display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(5)

    pygame.quit()


if __name__ == "__main__":
    try:
        q_table = load_qtable()
        print("Loaded existing Q-table.")
    except FileNotFoundError:
        print("No existing Q-table found. Starting fresh.")
        q_table = defaultdict(
            lambda: np.random.uniform(low=-0.05, high=0.05, size=len(actions))
        )

    if not isinstance(q_table, defaultdict):
        q_table = defaultdict(
            lambda: np.random.uniform(low=-0.05, high=0.05, size=len(actions)), q_table
        )

    q_table = train(q_table)
    dump_qtable(q_table)
    epsilon = 0
    successes = 0
    fails = 0
    for i in range(10):
        success = main(q_table)
        if success:
            successes += 1
        else:
            fails += 1

    print(f"Insgesamt {successes} mal erfolgreich, {fails} mal fehlerhaft")

"""
Write a pacman game using pygame. The pacman should be a yellow circle, the ghost is a red square. The labyrinth is written as a string as follows:
"##########
#........#
#.##..##.#
#........#
##########"

The "." are cookies the pacman can eat (as Graphics small white circles). The "#" are walls (as graphics blue squares on a black background ).  The ghost always tries to catch the pacman. Pacman as well as the ghost go one step in each game loop iteration. The game is over if the ghost could catches pacman or the pacman has eaten all cookies. Start your answer with "Shure, here is the full pacman code.


Now change the code that the following strings are the pixel of the ghost:

" ####
######
## # #
######
######
# # # " 

"""
