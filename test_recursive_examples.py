#!/usr/bin/env python3
"""
Test Examples for Recursive Code Parser
Contains various recursive function examples to test the parser capabilities
"""

def factorial(n):
    """Linear recursion example"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)


def fibonacci_naive(n):
    """Binary recursion example with overlapping subproblems"""
    if n <= 1:
        return n
    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)


def fibonacci_tail(n, a=0, b=1):
    """Tail recursion example"""
    if n == 0:
        return a
    return fibonacci_tail(n - 1, b, a + b)


def merge_sort(arr):
    """Divide and conquer recursion example"""
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)


def merge(left, right):
    """Helper function for merge sort"""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def binary_tree_height(node):
    """Tree traversal recursion example"""
    if node is None:
        return 0
    
    left_height = binary_tree_height(node.left)
    right_height = binary_tree_height(node.right)
    
    return 1 + max(left_height, right_height)


def is_even(n):
    """Mutual recursion example - function 1"""
    if n == 0:
        return True
    return is_odd(n - 1)


def is_odd(n):
    """Mutual recursion example - function 2"""
    if n == 0:
        return False
    return is_even(n - 1)


def ackermann(m, n):
    """Nested recursion example (computationally intensive)"""
    if m == 0:
        return n + 1
    elif n == 0:
        return ackermann(m - 1, 1)
    else:
        return ackermann(m - 1, ackermann(m, n - 1))


def generate_permutations(arr, start=0):
    """Backtracking recursion example"""
    if start == len(arr):
        return [arr[:]]
    
    result = []
    for i in range(start, len(arr)):
        # Swap
        arr[start], arr[i] = arr[i], arr[start]
        
        # Recursive call
        result.extend(generate_permutations(arr, start + 1))
        
        # Backtrack
        arr[start], arr[i] = arr[i], arr[start]
    
    return result


def power(base, exp):
    """Optimized recursive exponentiation"""
    if exp == 0:
        return 1
    elif exp == 1:
        return base
    elif exp % 2 == 0:
        half_power = power(base, exp // 2)
        return half_power * half_power
    else:
        return base * power(base, exp - 1)


def count_paths(grid, row, col, memo=None):
    """Dynamic programming recursion with memoization"""
    if memo is None:
        memo = {}
    
    if (row, col) in memo:
        return memo[(row, col)]
    
    if row == 0 or col == 0:
        return 1
    
    paths = count_paths(grid, row - 1, col, memo) + count_paths(grid, row, col - 1, memo)
    memo[(row, col)] = paths
    return paths


def tower_of_hanoi(n, source, destination, auxiliary):
    """Classic recursive problem"""
    if n == 1:
        print(f"Move disk 1 from {source} to {destination}")
        return
    
    tower_of_hanoi(n - 1, source, auxiliary, destination)
    print(f"Move disk {n} from {source} to {destination}")
    tower_of_hanoi(n - 1, auxiliary, destination, source)


class TreeNode:
    """Helper class for tree examples"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def depth_first_search(graph, node, visited=None):
    """Graph traversal recursion"""
    if visited is None:
        visited = set()
    
    visited.add(node)
    result = [node]
    
    for neighbor in graph.get(node, []):
        if neighbor not in visited:
            result.extend(depth_first_search(graph, neighbor, visited))
    
    return result


if __name__ == "__main__":
    # Test examples
    print("Testing recursive functions:")
    
    print(f"factorial(5) = {factorial(5)}")
    print(f"fibonacci_naive(7) = {fibonacci_naive(7)}")
    print(f"fibonacci_tail(7) = {fibonacci_tail(7)}")
    
    arr = [3, 1, 4, 1, 5, 9, 2, 6]
    print(f"merge_sort({arr}) = {merge_sort(arr)}")
    
    print(f"is_even(4) = {is_even(4)}")
    print(f"is_odd(4) = {is_odd(4)}")
    
    print(f"power(2, 8) = {power(2, 8)}")
    print(f"count_paths grid(2,2) = {count_paths(None, 2, 2)}")
    
    perms = generate_permutations([1, 2, 3])
    print(f"permutations([1,2,3]) count = {len(perms)}")
    
    # Tree example
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    print(f"binary_tree_height = {binary_tree_height(root)}")
    
    # Graph example
    graph = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }
    print(f"DFS from A = {depth_first_search(graph, 'A')}")