import pygame
from checkers_game.constants import SQUARE_SIZE, WHITE, BLACK

# Piece class to represent red and blue pieces
class Piece:
    PADDING = 15
    OUTLINE = 5

    def __init__(self, row, col, color):
        self.row = row
        self.col = col
        self.color = color
        self.king = False
        self.direction = -1 if color == (255, 0, 0) else 1  # RED moves up, BLUE moves down

    def __str__(self):
        if self.color == (0, 0, 255) and self.king:
            return "B\u0302"  # Blue King
        elif self.color == (0, 0, 255) and not self.king:
            return "B"  # Blue piece
        elif self.color == (255, 0, 0) and self.king:
            return "R\u0302"  # Red King
        elif self.color == (255, 0, 0) and not self.king:
            return "R"  # Red piece
        else:
            return "?"  # Unknown piece

    def make_king(self):
        self.king = True

    def draw(self, screen):
        center_x = self.col * SQUARE_SIZE + SQUARE_SIZE // 2
        center_y = self.row * SQUARE_SIZE + SQUARE_SIZE // 2
        radius = SQUARE_SIZE // 2 - self.PADDING

        pygame.draw.circle(screen, self.color, (center_x, center_y), radius)

        if self.king:
            # Draw a crown "K" on the king piece
            king_font = pygame.font.SysFont('Arial', 20, bold=True)
            text_color = WHITE if self.color != WHITE else BLACK
            text = king_font.render("K", True, text_color)
            text_rect = text.get_rect(center=(center_x, center_y))
            screen.blit(text, text_rect)

    def move(self, row, col):
        self.row = row
        self.col = col
