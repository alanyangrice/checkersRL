from MoveNode import MoveNode

class MoveTree:
    def __init__(self, root_value, curr_board):
        self.root = MoveNode(root_value, curr_board)  # The root node of the tree
        self.board = curr_board # Deep copy of current board state

    def add_node(self, parent_node, child_value, board):
        """Add a child node with the specified value to the given parent node."""
        child_node = MoveNode(child_value, board)
        parent_node.add_child(child_node)

    def get_leaf_sequences(self, node, current_path=None, delimiter='-'):
        """Retrieve leaf node sequences based on the specified delimited (- for move, x for capture)."""
        if current_path is None:
            current_path = [node.position]  # Start the path with the current node value

        # If the node has no children, it's a leaf node
        if not node.children:
            yield delimiter.join(map(str, current_path))  # Yield the path as a string
        else:
            for child in node.children:
                # Continue the path with the child's value
                yield from self.get_leaf_sequences(child, current_path + [child.position], delimiter)


"""
# Example Usage
tree = MoveTree(1)
tree.add_node(tree.root, 2)
tree.add_node(tree.root, 3)
tree.add_node(tree.root.children[0], 4)
tree.add_node(tree.root.children[0], 5)
tree.add_node(tree.root.children[1], 6)
tree.add_node(tree.root.children[0].children[1], 7)

# Get leaf sequences using 'x' for capture
print("Leaf Sequences with 'x':")
for sequence in tree.get_leaf_sequences(tree.root, delimiter='x'):
    print(sequence)

# Get leaf sequences using '-' for normal moves
print("\nLeaf Sequences with '-':")
for sequence in tree.get_leaf_sequences(tree.root, delimiter='-'):
    print(sequence)
"""