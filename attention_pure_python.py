# Scaled Dot-Product Attention using pure Python lists
# --------------------------------------------------
# Implements scaled dot-product attention from scratch using only Python lists.
# Demonstrates unmasked, causal masked, and padding masked attention, and visualizes
# attention weights as heatmaps using matplotlib.

import math
from typing import List


def transpose_short (matrix):
    return [list(row) for row in zip(*matrix)]


def transpose_long (matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    result = []

    # loop over columns of original
    for c in range(cols):
        new_row = []
        # loop over rows of original
        for r in range(rows):
            new_row.append(matrix[r][c])
        result.append(new_row)

    return result


M = [[1, 2, 3], [4, 5, 6]]
print(f'transpose_short(M):{transpose_short(M):}')

M = [[1, 2], [3, 4], [5, 6]]
print(f'transpose_long(M): {transpose_long(M):}')


def matmul(A, B):

    # Basic shape checks
    if not A or not B:
        raise ValueError("A and B must be non-empty matrices")

    m = len(A)
    n = len(A[0])
    # Validate A is rectangular
    for r in A:
        if len(r) != n:
            raise ValueError("All rows of A must have same length")

    # Validate B is rectangular and inner dimension matches n
    if any(len(row) == 0 for row in B):
        raise ValueError("Rows of B must be non-empty")

    if len(B) != n:
        raise AssertionError(f"Inner dimensions must match: A has {n} columns but B has {len(B)} rows")

    p = len(B[0])
    for row in B:
        if len(row) != p:
            raise ValueError("All rows of B must have same length")

    # Initialize result matrix with zeros: m x p
    C = [[0.0 for _ in range(p)] for _ in range(m)]

    # Triple loop: for each i, j, accumulate sum over k
    for i in range(m):
        for k in range(n):
            a_ik = A[i][k]
            # micro-optimization: pull row of B for faster inner loop access
            Bk = B[k]
            for j in range(p):
                C[i][j] += a_ik * Bk[j]

    return C


A = [[1,2,3],[4,5,6]]
B = [[7,8],[9,10],[11,12]]
print(f'matmul(A, B):{matmul(A,B)}')


def softmax_short(vector):
    exps = [math.exp(x) for x in vector]
    s = sum(exps)
    return [e / s for e in exps]


def softmax(vec: List[float]) -> List[float]:
    if not vec:
        raise ValueError("softmax: input vector must be non-empty")

    # subtract max for numerical stability
    m = max(vec)
    exps = [math.exp(x - m) for x in vec]

    # normalize
    s = sum(exps)
    if s == 0.0:
        # extremely unlikely after subtracting max unless underflowed;
        # fall back to a uniform distribution to avoid division by zero.
        n = len(vec)
        return [1.0 / n] * n

    return [e / s for e in exps]


values_normal = [1.0, 2.0, 3.0]
print(f'softmax(values_normal): {softmax(values_normal)}')

# print(f'softmax_short(values_normal): {softmax_short(values_normal)}')



values_large = [1000.0, 1001.0]
# Without subtracting max you'd try exp(1001) which overflows.
print(f'softmax(values_large): {softmax(values_large)}')


def pretty_print_matrix(m, name="Matrix"):
    print(f"{name} (shape {len(m)} x {len(m[0])}):")
    for row in m:
        print("  [" + ", ".join(f"{x: .4f}" for x in row) + "]")
    print()

Q = [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
]

K = [
    [1.0, 0.5],
    [0.5, 1.0],
    [1.0, -1.0]
]

dk = len(Q[0])


KT = transpose_short(K)
pretty_print_matrix(Q, "Q")
pretty_print_matrix(K, "K")
pretty_print_matrix(KT, "K^T")

scores = matmul(Q, KT)
scale = math.sqrt(dk)
scores_scaled = [[elem / scale for elem in row] for row in scores]

pretty_print_matrix(scores, "Raw scores = Q K^T (pre-scale)")
pretty_print_matrix(scores_scaled, f"Scaled scores = Q K^T / sqrt({dk})")