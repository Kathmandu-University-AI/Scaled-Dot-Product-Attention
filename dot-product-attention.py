# Scaled Dot-Product Attention using pure Python lists
# --------------------------------------------------
# Implements scaled dot-product attention from scratch using only Python lists.
# Demonstrates unmasked, causal masked, and padding masked attention, and visualizes
# attention weights as heatmaps using matplotlib.

import math
from typing import List
import matplotlib.pyplot as plt

NEG_INF = -1e9  # sentinel for masked positions

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


def scalar_divide_matrix(M, scalar):
    return [[elem / scalar for elem in row] for row in M]


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

# ---------- Scaled dot-product attention (pure-python lists) ----------
def scaled_dot_product_attention(Q, K, V, mask=None, dk=None):
    # Q: seq x dk, K: seq x dk, V: seq x dv (dv can be dk here)
    if dk is None:
        dk = len(K[0])
    # scores = Q K^T
    KT = transpose_short(K)  # dk x seq
    scores = matmul(Q, KT)  # seq x seq
    scale = math.sqrt(dk)
    scores = scalar_divide_matrix(scores, scale)
    # apply mask: mask should be same shape as scores, with True to mask out
    if mask is not None:
        # mask True => set to very negative to zero out after softmax
        NEG_INF = -1e9
        masked = []
        for i in range(len(scores)):
            row = []
            for j in range(len(scores[0])):
                if mask[i][j]:
                    row.append(NEG_INF)
                else:
                    row.append(scores[i][j])
            masked.append(row)
        scores = masked
    # softmax row-wise to get attention weights
    attn_weights = [softmax(row) for row in scores]
    # output = attn_weights @ V
    output = matmul(attn_weights, V)
    return output, attn_weights, scores


def pretty_print_matrix(m, name="Matrix"):
    print(f"{name} (shape {len(m)} x {len(m[0])}):")
    for row in m:
        print("  [" + ", ".join(f"{x: .4f}" for x in row) + "]")
    print()

# ---------- example matrices (seq_len=3, d_k=2) ----------

Q = [
    [1.0, 0.0],  # query for token 0
    [0.0, 1.0],  # query for token 1
    [1.0, 1.0],  # query for token 2
]

K = [
    [1.0, 0.5],  # key for token 0
    [0.5, 1.0],  # key for token 1
    [1.0, -1.0], # key for token 2
]

V = [
    [1.0, 0.0],  # value for token 0
    [0.0, 1.0],  # value for token 1
    [1.0, 1.0],  # value for token 2
]

seq_len = len(Q)
dk = len(Q[0])

out_unmasked, attn_unmasked, scores_unmasked = scaled_dot_product_attention(Q, K, V, mask=None, dk=dk)

#---------- Causal mask (prevent attending to future tokens) ----------
causal_mask = [[(j > i) for j in range(seq_len)] for i in range(seq_len)]
out_causal, attn_causal, scores_causal = scaled_dot_product_attention(Q, K, V, mask=causal_mask, dk=dk)

# ---------- Padding mask example ----------
# Suppose token 1 is padding (index 1). We should prevent attending to padding positions (column masked).

padding_index = 1
padding_mask = [[(j == padding_index) for j in range(seq_len)] for i in range(seq_len)]
out_padding, attn_padding, scores_padding = scaled_dot_product_attention(Q, K, V, mask=padding_mask, dk=dk)


KT = transpose_short(K)
pretty_print_matrix(Q, "Q")
pretty_print_matrix(K, "K")
pretty_print_matrix(KT, "K^T")

scores = matmul(Q, KT)
scale = math.sqrt(dk)
scores_scaled = [[elem / scale for elem in row] for row in scores]

pretty_print_matrix(scores, "Raw scores = Q K^T (pre-scale)")
pretty_print_matrix(scores_scaled, f"Scaled scores = Q K^T / sqrt({dk})")

print("Q:")
pretty_print_matrix(Q)
print("\nK:")
pretty_print_matrix(K)
print("\nV:")
pretty_print_matrix(V)

print("\nScores (unmasked, pre-softmax):")
pretty_print_matrix(scores_unmasked)
print("\nAttention weights (unmasked):")
pretty_print_matrix(attn_unmasked)
print("\nOutput (unmasked):")
pretty_print_matrix(out_unmasked)

print("\nScores (causal masked, pre-softmax):")
pretty_print_matrix(scores_causal)
print("\nAttention weights (causal):")
pretty_print_matrix(attn_causal)
print("\nOutput (causal masked):")
pretty_print_matrix(out_causal)

print("\nScores (padding masked, pre-softmax):")
pretty_print_matrix(scores_padding)
print("\nAttention weights (padding):")
pretty_print_matrix(attn_padding)
print("\nOutput (padding masked):")
pretty_print_matrix(out_padding)

# visualize attention weights as heatmaps: Unmasked | Causal | Padding
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].set_title("Unmasked"); axes[0].imshow(attn_unmasked); axes[0].set_xlabel("Key positions (j)"); axes[0].set_ylabel("Query positions (i)")
axes[1].set_title("Causal Masked"); axes[1].imshow(attn_causal); axes[1].set_xlabel("Key positions (j)"); axes[1].set_ylabel("Query positions (i)")
axes[2].set_title("Padding Masked (pos 1)"); axes[2].imshow(attn_padding); axes[2].set_xlabel("Key positions (j)"); axes[2].set_ylabel("Query positions (i)")
plt.tight_layout()
plt.show()