import pygame
import random
import math

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

# Get labyrinth dimensions
ROWS = len(labyrinth)
COLS = len(labyrinth[0])

# Initialize game screen
screen = pygame.display.set_mode((COLS * CELL_SIZE, ROWS * CELL_SIZE))
pygame.display.set_caption("Micro-Pacman")


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


# Main game function
def main():
    clock = pygame.time.Clock()

    # Initialize Pacman and Ghost positions
    pacman = Pacman(1, 1)
    ghost = Ghost(COLS - 2, ROWS - 2)

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
            # Ghost moves towards Pacman
            ghost.move_towards_pacman(pacman)

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
    main()


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
