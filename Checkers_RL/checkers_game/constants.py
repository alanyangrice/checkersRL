import pygame

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 128, 0)

# Constants
WIDTH, HEIGHT = 600, 600
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS

# Fonts
pygame.font.init()
font = pygame.font.SysFont('Arial', 24)

# Helper functions
def board_number_to_position(num):
    mapping = [
        (0, 1), (0, 3), (0, 5), (0, 7),
        (1, 0), (1, 2), (1, 4), (1, 6),
        (2, 1), (2, 3), (2, 5), (2, 7),
        (3, 0), (3, 2), (3, 4), (3, 6),
        (4, 1), (4, 3), (4, 5), (4, 7),
        (5, 0), (5, 2), (5, 4), (5, 6),
        (6, 1), (6, 3), (6, 5), (6, 7),
        (7, 0), (7, 2), (7, 4), (7, 6)
    ]
    return mapping[num - 1]

def position_to_board_number(row, col):
    mapping = [
        (0, 1), (0, 3), (0, 5), (0, 7),
        (1, 0), (1, 2), (1, 4), (1, 6),
        (2, 1), (2, 3), (2, 5), (2, 7),
        (3, 0), (3, 2), (3, 4), (3, 6),
        (4, 1), (4, 3), (4, 5), (4, 7),
        (5, 0), (5, 2), (5, 4), (5, 6),
        (6, 1), (6, 3), (6, 5), (6, 7),
        (7, 0), (7, 2), (7, 4), (7, 6)
    ]
    return mapping.index((row, col)) + 1