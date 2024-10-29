# Initialize the move dictionaries and the counter
move_to_index = {}
index_to_move = {}
next_index = 0

# Function to convert action to index, dynamically updating dictionaries
def get_action_index(action_str):
    global next_index
    if action_str not in move_to_index:
        move_to_index[action_str] = next_index
        index_to_move[next_index] = action_str
        next_index += 1
    return move_to_index[action_str]

# Function to decode action indices back to strings
def decode_action_index(action_index):
    return index_to_move.get(action_index, "Unknown Action")