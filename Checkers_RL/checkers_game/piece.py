import pygame
from checkers_game.constants import SQUARE_SIZE, WHITE

# Piece class to represent red and blue pieces
class Piece:
    PADDING = 15
    OUTLINE = 5

    def __init__(self, row, col, color):
        self.row = row
        self.col = col
        self.color = color
        self.king = False
        self.direction = -1 if color == (255, 0, 0) else 1  # RED is (255, 0, 0)

    def __str__(self):
        if self.color == (0, 0, 255) and self.king:
            return "B̂"  # Blue King
        elif self.color == (0, 0, 255) and not self.king:
            return "B"  # Blue piece
        elif self.color == (255, 0, 0) and self.king:
            return "R̂"  # Red King
        elif self.color == (255, 0, 0) and not self.king:
            return "R"  # Red piece
        else:
            return "?"  # Unknown piece

    def make_king(self):
        self.king = True

    def draw(self, screen):
        radius = SQUARE_SIZE // 2 - self.PADDING
        pygame.draw.circle(screen, self.color, (self.col * SQUARE_SIZE + SQUARE_SIZE // 2, self.row * SQUARE_SIZE + SQUARE_SIZE // 2), radius)
        if self.king:
            pygame.draw.circle(screen, WHITE, (self.col * SQUARE_SIZE + SQUARE_SIZE // 2, self.row * SQUARE_SIZE + SQUARE_SIZE // 2), radius - self.OUTLINE)

    def move(self, row, col):
        self.row = row
        self.col = col