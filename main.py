import pygame
import random
import math
import json
from collections import defaultdict, deque
import numpy as np

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

WALL = "#"
COOKIE = "."

# Labyrinth as a string
labyrinth = [
    "####################",
    "# .......  ........#",
    "#.##..##.#.### ###.#",
    "#.......   ... ... #",
    "#.##..##.#.### ###.#",
    "#.......  .........#",
    "####################",
]

# Get labyrinth dimensions
ROWS = len(labyrinth)
COLS = len(labyrinth[0])

# Initialize game screen
screen = pygame.display.set_mode((COLS * CELL_SIZE, ROWS * CELL_SIZE))
pygame.display.set_caption("Micro-Pacman")

actions = ["UP", "DOWN", "LEFT", "RIGHT", "NONE"]
q_table = defaultdict(lambda: np.random.uniform(low=0, high=0.05, size=len(actions)))


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
        if self.x < pacman.x and labyrinth[self.y][self.x + 1] != WALL:
            self.x += 1
        elif self.x > pacman.x and labyrinth[self.y][self.x - 1] != WALL:
            self.x -= 1
        elif self.y < pacman.y and labyrinth[self.y + 1][self.x] != WALL:
            self.y += 1
        elif self.y > pacman.y and labyrinth[self.y - 1][self.x] != WALL:
            self.y -= 1

    def draw(self):
        pixel_size = CELL_SIZE // len(
            self.ghost_pixels
        )  # Size of each pixel in the ghost art
        for row_idx, row in enumerate(self.ghost_pixels):
            for col_idx, pixel in enumerate(row):
                if pixel == WALL:
                    pixel_x = self.x * CELL_SIZE + col_idx * pixel_size
                    pixel_y = self.y * CELL_SIZE + row_idx * pixel_size
                    pygame.draw.rect(
                        screen, RED, (pixel_x, pixel_y, pixel_size, pixel_size)
                    )


# Draw walls and cookies
def draw_labyrinth():
    for y, row in enumerate(labyrinth):
        for x, cell in enumerate(row):
            if cell == WALL:
                pygame.draw.rect(
                    screen, BLUE, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                )
            elif cell == COOKIE:
                pygame.draw.circle(
                    screen,
                    WHITE,
                    (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2),
                    5,
                )


def find_nearest_cookie(start, labyrinth):
    visited = set([start])
    queue = deque([(start, 0)])

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        (x, y), dist = queue.popleft()

        if labyrinth[y][x] == COOKIE:
            return x, y

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if (
                0 <= nx < COLS
                and 0 <= ny < ROWS
                and labyrinth[ny][nx] != WALL
                and (nx, ny) not in visited
            ):
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))

    return None, None


def sign(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


def compute_state(pacman: Pacman, ghost: Ghost, labyrinth: list):
    """
    This function computes the state for q-learning.

    Possible state combinations:

        wall_up ∈ {0, 1}
        wall_down ∈ {0, 1}
        wall_left ∈ {0, 1}
        wall_right ∈ {0, 1}
        cookie_dir_x ∈ {-1, 0, 1}
        cookie_dir_y ∈ {-1, 0, 1}
        ghost_dir_x ∈ {-1, 0, 1}
        ghost_dir_y ∈ {-1, 0, 1}
        distance_to_ghost ∈ [0, COLS + ROWS]

    Therefore the size of the state space is: 2**4 * 3**4 * (ROWS + COLS) = 1296
    """

    nearest_cookie_x, nearest_cookie_y = find_nearest_cookie(
        (pacman.x, pacman.y), labyrinth
    )

    wall_up = int(labyrinth[pacman.y - 1][pacman.x] == WALL)
    wall_down = int(labyrinth[pacman.y + 1][pacman.x] == WALL)
    wall_left = int(labyrinth[pacman.y][pacman.x - 1] == WALL)
    wall_right = int(labyrinth[pacman.y][pacman.x + 1] == WALL)

    cookie_dir_x = sign(nearest_cookie_x - pacman.x)
    cookie_dir_y = sign(nearest_cookie_y - pacman.y)

    ghost_dir_x = sign(ghost.x - pacman.x)
    ghost_dir_y = sign(ghost.y - pacman.y)
    distance_to_ghost = abs(ghost.x - pacman.x) + abs(ghost.y - pacman.y)

    return (
        wall_up,
        wall_down,
        wall_left,
        wall_right,
        cookie_dir_x,
        cookie_dir_y,
        ghost_dir_x,
        ghost_dir_y,
        distance_to_ghost,
    )


def epsilon_greedy(q_table, state, actions, epsilon):
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        q_values = q_table[state]
        return actions[np.argmax(q_values)]


def take_action(action, labyrinth, pacman: Pacman, ghost: Ghost):
    if pacman.x == ghost.x and pacman.y == ghost.y:
        return compute_state(pacman, ghost, labyrinth), -100

    new_x, new_y = pacman.x, pacman.y

    match action:
        case "UP":
            new_y -= 1
        case "DOWN":
            new_y += 1
        case "LEFT":
            new_x -= 1
        case "RIGHT":
            new_x += 1

    if labyrinth[new_y][new_x] == WALL:
        return compute_state(pacman, ghost, labyrinth), -50

    if new_x == ghost.x and new_y == ghost.y:
        return compute_state(pacman, ghost, labyrinth), -350

    pacman.x, pacman.y = new_x, new_y

    if labyrinth[new_y][new_x] == COOKIE:
        if all(COOKIE not in row for row in labyrinth):
            return compute_state(pacman, ghost, labyrinth), 1000
        else:
            return compute_state(pacman, ghost, labyrinth), 10

    return compute_state(pacman, ghost, labyrinth), -5


def train(episodes):

    epsilon = 1.00
    epsilon_min = 0.01
    epsilon_decay = 0.995
    max_steps = 400

    for i in range(episodes):
        if i % 1000 == 0:
            print(f"Episode {i}/{episodes}")

        labyrinth = [
            "####################",
            "# .......  ........#",
            "#.##..##.#.### ###.#",
            "#.......   ... ... #",
            "#.##..##.#.### ###.#",
            "#.......  .........#",
            "####################",
        ]

        pacman = Pacman(1, 1)
        ghost = Ghost(COLS - 2, ROWS - 2)

        state = compute_state(pacman, ghost, labyrinth)

        for i in range(max_steps):
            ghost.move_towards_pacman(pacman)

            action = epsilon_greedy(q_table, state, actions, epsilon)

            new_state, reward = take_action(action, labyrinth, pacman, ghost)

            q_table[state][actions.index(action)] += 0.1 * (
                reward
                + 0.9 * np.max(q_table[new_state])
                - q_table[state][actions.index(action)]
            )

            state = new_state

            if pacman.x == ghost.x and pacman.y == ghost.y:
                break

            # Eat cookies
            if labyrinth[pacman.y][pacman.x] == ".":
                labyrinth[pacman.y] = (
                    labyrinth[pacman.y][: pacman.x]
                    + " "
                    + labyrinth[pacman.y][pacman.x + 1 :]
                )

            # Check if all cookies are eaten (game over)
            if all("." not in row for row in labyrinth):
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)


def persist_q_table(q_table: dict):
    json_compatible = {",".join(map(str, k)): v.tolist() for k, v in q_table.items()}

    with open("q_table.json", "w") as f:
        json.dump(json_compatible, f, indent=2)


def load_q_table() -> dict:
    with open("q_table.json", "r") as f:
        loaded = json.load(f)
        return {tuple(map(int, k.split(","))): np.array(v) for k, v in loaded.items()}


# Main game function
def main():
    clock = pygame.time.Clock()

    # Initialize Pacman and Ghost positions
    pacman = Pacman(1, 1)
    ghost = Ghost(COLS - 2, ROWS - 2)

    state = compute_state(pacman, ghost, labyrinth)

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
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    pacman.move(-1, 0)
                if event.key == pygame.K_RIGHT:
                    pacman.move(1, 0)
                if event.key == pygame.K_UP:
                    pacman.move(0, -1)
                if event.key == pygame.K_DOWN:
                    pacman.move(0, 1)

        if iter % 3 == 0:
            # Ghost moves towards Pacman
            ghost.move_towards_pacman(pacman)

            action = epsilon_greedy(q_table, state, actions, 0)
            new_state, reward = take_action(action, labyrinth, pacman, ghost)

            q_table[state][actions.index(action)] += 0.1 * (
                reward
                + 0.9 * np.max(q_table[new_state])
                - q_table[state][actions.index(action)]
            )

            state = new_state

        # Check for collisions (game over if ghost catches pacman)
        if pacman.x == ghost.x and pacman.y == ghost.y:
            print("Game Over! The ghost caught Pacman.")
            running = False

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
    train(10000)
    main()
