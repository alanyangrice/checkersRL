import sys
import pygame
from checkers_game.constants import WIDTH, HEIGHT, BLUE, RED, WHITE, BLACK, large_font, medium_font
from checkers_game.game import Game


def draw_end_screen(screen, winner_text):
    """Draw a game-over overlay with the result and a Play Again button."""
    # Semi-transparent overlay
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    screen.blit(overlay, (0, 0))

    # Winner text
    text = large_font.render(winner_text, True, WHITE)
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 40))
    screen.blit(text, text_rect)

    # Play Again button
    button_width, button_height = 200, 50
    button_rect = pygame.Rect(
        WIDTH // 2 - button_width // 2,
        HEIGHT // 2 + 30,
        button_width,
        button_height
    )
    pygame.draw.rect(screen, (0, 150, 0), button_rect, border_radius=8)
    pygame.draw.rect(screen, WHITE, button_rect, 2, border_radius=8)

    btn_text = medium_font.render("Play Again", True, WHITE)
    btn_text_rect = btn_text.get_rect(center=button_rect.center)
    screen.blit(btn_text, btn_text_rect)

    pygame.display.update()
    return button_rect


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Checkers Game")

    game = Game()
    game.update_board(screen)

    run = True
    game_over = False
    play_again_rect = None

    while run:
        if game_over:
            # Handle end-screen events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if play_again_rect and play_again_rect.collidepoint(event.pos):
                        # Reset the game
                        game = Game()
                        game.update_board(screen)
                        game_over = False
                        play_again_rect = None
        else:
            turn_complete = game.player_action(screen)

            if turn_complete:
                game.update_board(screen)

                winner = game.check_winner()
                if winner:
                    if winner == "Tie":
                        winner_text = "It's a Draw!"
                    else:
                        color_name = "Blue" if winner == BLUE else "Red"
                        winner_text = f"{color_name} Wins!"

                    play_again_rect = draw_end_screen(screen, winner_text)
                    game_over = True
                else:
                    game.switch_turn()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
