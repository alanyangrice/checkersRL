import sys
sys.path.append(r"c:/Users/Alan Yang/Desktop/checkersRL/Checkers_RL")

import pygame
import sys
from checkers_game.board import Board
from checkers_game.constants import WIDTH, HEIGHT, BLUE, RED

# Initialize pygame
pygame.init()

# Load the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.display.set_caption("Checkers Game")

def main():
    board = Board()
    board_states = {}  # Track if there is a tie
    moves = []
    move_num = 0
    run = True

    while run:
        board.draw(screen)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                sys.exit()

        board.get_all_possible_moves()

        user_input = input(f"Enter your move ({'Red' if board.turn == (255, 0, 0) else 'Blue'}), e.g., '11-15' for move, '11x18' or '11x18x22' for capture: ")

        capture = board.is_capture_possible() # Must capture if possible

        try:
            move_successful = False  # Flag to track if the move was successful
            if "-" in user_input and not capture:  # Normal move
                move_successful = board.move_piece(list(map(int, user_input.split('-'))))
                if not move_successful:
                    print("Invalid move.")
            elif "-" in user_input and capture:
                print("Need to capture.")
            elif "x" in user_input:  # Capture move
                capture_moves = list(map(int, user_input.split('x')))
                move_successful = board.capture_piece(capture_moves)
                if not move_successful:
                    print("Invalid capture sequence.")
            else:
                print("Invalid input format.")

            # Update board state after a successful move or capture
            if move_successful:
                board.switch_turn() # Switch turn only after a successful move
                move_num += 1
                moves.append(user_input)
                board_hash = board.get_board_hash()
                if board_hash in board_states:
                    board_states[board_hash] += 1
                else:
                    board_states[board_hash] = 1
                print(f"Move Number: {move_num}, Moves made: {moves}")

                # Check for tie by repitition
                for count in board_states.values():
                    if count >= 3:
                        print("The game is a draw!")
                        run = False
                        break

        except Exception as e:
            print("Invalid input. Please enter your move in the correct format.")
            print(f"Error: {e}")
    
    print(f"Game moves: {moves}") # Print moves played in entire game

if __name__ == "__main__":
    main()