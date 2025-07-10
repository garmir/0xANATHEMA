#!/usr/bin/env python3
import sys
import math
import time

def sqrt_space_algorithm(n):
    """O(√n) space algorithm simulation"""
    chunk_size = max(1, int(math.sqrt(n)))
    chunks = []
    
    # Allocate sqrt(n) chunks, each with sqrt(n) elements
    for i in range(chunk_size):
        chunk = list(range(chunk_size))
        chunks.append(chunk)
    
    # Simulate processing
    time.sleep(0.5)  # Give time for memory measurement
    
    result = 0
    for chunk in chunks:
        result += sum(chunk)
    
    return result

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    print(f"Running sqrt algorithm with n={n}")
    result = sqrt_space_algorithm(n)
    print(f"Result: {result}")
