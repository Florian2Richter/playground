import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.traverse_order_search import TreeNode, level_order_search


def build_tree():
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    return root


def test_level_order_found():
    root = build_tree()
    assert level_order_search(root, 4) is True


def test_level_order_not_found():
    root = build_tree()
    assert level_order_search(root, 6) is False
