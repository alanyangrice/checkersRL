class MoveNode:
    def __init__(self, position, curr_board):
        self.position = position  # The current position of the piece (Number 1 to 32)
        self.board = curr_board # The current board state
        self.children = []  # List of child MoveNodes representing further moves

    def add_child(self, child_node):
        """Add a child node to the current node."""
        self.children.append(child_node)

    def __repr__(self):
        return f"MoveNode(position={self.position})"