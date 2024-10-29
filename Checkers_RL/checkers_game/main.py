import sys
sys.path.append(r"/Users/alanyang/Downloads/checkersRL/Checkers_RL")

import pygame
from checkers_game.constants import WIDTH, HEIGHT
from checkers_game.game import Game  # Import the Game class

# Initialize pygame
pygame.init()

# Set up the game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Checkers Game")

def main():
    # Initialize the Game
    game = Game()
    run = True

    game.update_board(screen)

    while run:
        # Process player actions and update only if a turn is complete
        turn_complete = game.player_action(screen)
        
        if turn_complete:
            print("Turn completed")
            game.switch_turn()

            #print("Updating board")
            game.update_board(screen)

            # Check for a win or tie condition after the turn ends
            winner = game.check_winner()
            if winner:
                if winner == "Tie":
                    print("The game is a draw!")
                else:
                    print(f"{'Blue' if winner == (0, 0, 255) else 'Red'} wins!")
                run = False  # End game

if __name__ == "__main__":
    main()
