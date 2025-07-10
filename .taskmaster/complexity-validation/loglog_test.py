#!/usr/bin/env python3
import sys
import math
import time

def loglog_space_algorithm(n):
    """O(log n * log log n) space algorithm simulation"""
    if n <= 2:
        return n
    
    log_n = max(1, int(math.log(n)))
    log_log_n = max(1, int(math.log(log_n)) if log_n > 1 else 1)
    
    # Allocate O(log n * log log n) space
    stack = list(range(log_n))
    memoization = {}
    
    for i in range(log_log_n):
        for j in range(log_n):
            key = (i, j)
            memoization[key] = i * j
    
    # Simulate processing
    time.sleep(0.5)  # Give time for memory measurement
    
    result = sum(stack) + sum(memoization.values())
    return result

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    print(f"Running loglog algorithm with n={n}")
    result = loglog_space_algorithm(n)
    print(f"Result: {result}")
