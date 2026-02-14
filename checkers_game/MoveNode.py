class MoveNode:
    def __init__(self, position, curr_board):
        self.position = position  # The current position of the piece (Number 1 to 32)
        self.board = curr_board  # The current board state
        self.children = []  # List of child MoveNodes representing further moves

    def add_child(self, child_node):
        """Add a child node to the current node."""
        self.children.append(child_node)
    
    def get_leaf_sequences(self, node, current_path=None, delimiter='-'):
        """Retrieve leaf node sequences based on the specified delimited (- for move, x for capture)."""
        if current_path is None:
            current_path = [node.position]  # Start the path with the current node value

        # If the node has no children, it's a leaf node
        if not node.children:
            yield delimiter.join(map(str, current_path)), node.board  # Yield the path as a string
        else:
            for child in node.children:
                # Continue the path with the child's value
                yield from self.get_leaf_sequences(child, current_path + [child.position], delimiter)