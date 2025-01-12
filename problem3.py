import numpy as np
from itertools import product

# Function to compute R = S^T S
def compute_R(S):
    return S.T @ S

# Function to check normality for a given matrix S
def is_normal(S):
    K = S.shape[1]
    R = compute_R(S)

    # Generate all a, b such that a, b \in {0, 1}^K and satisfy the conditions
    for a in product([0, 1], repeat=K):
        a = np.array(a)
        if np.all(a == 0):  # Skip a = 0
            continue
        b = 1 - a  # b = 1 - a to satisfy a + b = 1
        if np.all(b == 0):  # Skip b = 0
            continue

        # Check the condition a^T R b < 0
        if not (a @ R @ b < 0):
            return False  # If any condition fails, S is not normal

    return True  # S is normal if all conditions are satisfied

# Main function to count normal matrices
def count_normal_matrices(N, K):
    # Generate all possible S matrices with entries in {-1/sqrt(N), 1/sqrt(N)}
    values = [-1 / np.sqrt(N), 1 / np.sqrt(N)]
    total_matrices = 0
    normal_count = 0

    for S_flat in product(values, repeat=N * K):
        S = np.array(S_flat).reshape(N, K)
        total_matrices += 1

        if is_normal(S):
            normal_count += 1

    return normal_count, total_matrices

# Example usage
N = 4   # Number of rows
K = 3  # Number of columns

normal_count, total_matrices = count_normal_matrices(N, K)
print(f"Number of normal matrices: {normal_count}")
print(f"Total matrices checked: {total_matrices}")
