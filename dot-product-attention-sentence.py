import math

# Static embeddings (d_model = 4)
embeddings = {
    "i":      [0.1, 0.2, 0.3, 0.4],
    "love":   [0.5, 0.1, 0.3, 0.7],
    "nepalese": [0.6, 0.4, 0.2, 0.1],
    "food":   [0.3, 0.8, 0.5, 0.2]
}

sentence = ["i", "love", "nepalese", "food"]
X = [embeddings[w] for w in sentence]   # shape (4,4)
print(X)

def positional_encoding(seq_len, d_model):
    PE = [[0]*d_model for _ in range(seq_len)]
    for pos in range(seq_len):
        for i in range(d_model):
            angle = pos / (10000 ** ((2 * i) / d_model))
            if i % 2 == 0:  # even
                PE[pos][i] = math.sin(angle)
            else:           # odd
                PE[pos][i] = math.cos(angle)
    return PE

PE = positional_encoding(len(sentence), 4)

# Add PE to embeddings
X_pe = [[X[r][c] + PE[r][c] for c in range(4)] for r in range(len(sentence))]
print(X_pe)

# Random small projections (4->2)
WQ = [[0.2, 0.1],[0.0,0.3],[0.1,0.2],[0.3,0.0]]
WK = [[0.1,0.3],[0.2,0.0],[0.0,0.2],[0.1,0.1]]
WV = [[0.3,0.0],[0.0,0.1],[0.2,0.2],[0.1,0.3]]

def transpose(matrix):
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

def matmul(A, B):
    return [[sum(A[i][k]*B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]

def softmax(x):
    m = max(x)
    exps = [math.exp(i - m) for i in x]
    s = sum(exps)
    return [e/s for e in exps]

Q = matmul(X_pe, WQ)  # (4,2)
K = matmul(X_pe, WK)  # (4,2)
V = matmul(X_pe, WV)  # (4,2)

d_k = len(K[0])
K_T = transpose(K)

# Raw scaled scores
scores = matmul(Q, K_T)
scores = [[s / math.sqrt(d_k) for s in row] for row in scores]

# Softmax row-wise
attention_weights = [softmax(row) for row in scores]

# Weighted sum of V
output = matmul(attention_weights, V)

print("Attention Weights:")
for row in attention_weights:
    print([round(x,3) for x in row])

print("\nOutput (weighted V):")
for row in output:
    print([round(x,3) for x in row])