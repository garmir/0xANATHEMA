#!/usr/bin/env python3
import sys
import math
import csv

def analyze_growth_pattern(csv_file, complexity_type):
    n_values = []
    memory_values = []
    
    # Read data
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_values.append(int(row['n']))
            memory_values.append(float(row['peak_memory_mb']))
    
    if len(n_values) < 3:
        return False, 0.0, "Insufficient data points"
    
    # Calculate theoretical growth
    theoretical_values = []
    for n in n_values:
        if complexity_type == "sqrt":
            theoretical_values.append(math.sqrt(n))
        elif complexity_type == "loglog":
            if n <= 2:
                theoretical_values.append(1)
            else:
                log_n = math.log(n)
                log_log_n = math.log(log_n) if log_n > 1 else 1
                theoretical_values.append(log_n * log_log_n)
    
    # Normalize both series to compare growth patterns
    def normalize(values):
        if not values or min(values) == max(values):
            return values
        min_val, max_val = min(values), max(values)
        return [(v - min_val) / (max_val - min_val) for v in values]
    
    norm_memory = normalize(memory_values)
    norm_theoretical = normalize(theoretical_values)
    
    # Calculate correlation coefficient
    if len(norm_memory) == len(norm_theoretical):
        n_points = len(norm_memory)
        sum_xy = sum(x * y for x, y in zip(norm_memory, norm_theoretical))
        sum_x = sum(norm_memory)
        sum_y = sum(norm_theoretical)
        sum_x2 = sum(x * x for x in norm_memory)
        sum_y2 = sum(y * y for y in norm_theoretical)
        
        numerator = n_points * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n_points * sum_x2 - sum_x**2) * (n_points * sum_y2 - sum_y**2))
        
        if denominator != 0:
            correlation = numerator / denominator
        else:
            correlation = 0.0
    else:
        correlation = 0.0
    
    # Growth pattern validation (correlation > 0.7 indicates good match)
    is_valid = correlation > 0.7
    
    # Calculate growth ratios for additional validation
    memory_ratios = []
    theoretical_ratios = []
    
    for i in range(1, len(memory_values)):
        if memory_values[i-1] > 0:
            memory_ratios.append(memory_values[i] / memory_values[i-1])
        if theoretical_values[i-1] > 0:
            theoretical_ratios.append(theoretical_values[i] / theoretical_values[i-1])
    
    # Print detailed analysis
    print(f"Growth Pattern Analysis for {complexity_type}:")
    print(f"  Data points: {n_values}")
    print(f"  Memory values: {[f'{v:.2f}' for v in memory_values]}")
    print(f"  Theoretical values: {[f'{v:.2f}' for v in theoretical_values]}")
    print(f"  Correlation coefficient: {correlation:.3f}")
    print(f"  Growth pattern valid: {is_valid}")
    
    return is_valid, correlation, "Growth pattern analysis complete"

if __name__ == "__main__":
    csv_file = sys.argv[1]
    complexity_type = sys.argv[2]
    is_valid, correlation, message = analyze_growth_pattern(csv_file, complexity_type)
    print(f"RESULT: {is_valid},{correlation:.3f},{message}")
