class TreeNode:
    """Simple binary tree node."""

    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


def level_order_search(root, target):
    """Return True if target is found using level order traversal."""
    if root is None:
        return False

    from collections import deque

    queue = deque([root])
    while queue:
        node = queue.popleft()
        if node.value == target:
            return True
        if node.left is not None:
            queue.append(node.left)
        if node.right is not None:
            queue.append(node.right)
    return False


if __name__ == "__main__":
    # Construct a small example tree
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)

    # Search for a present value
    print("Searching for 4:", level_order_search(root, 4))
    # Search for an absent value
    print("Searching for 6:", level_order_search(root, 6))
