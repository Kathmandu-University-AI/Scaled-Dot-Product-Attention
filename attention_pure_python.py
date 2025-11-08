# Scaled Dot-Product Attention using pure Python lists
# --------------------------------------------------
# Implements scaled dot-product attention from scratch using only Python lists.
# Demonstrates unmasked, causal masked, and padding masked attention, and visualizes
# attention weights as heatmaps using matplotlib.


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
print(transpose_short(M))

M = [[1, 2], [3, 4], [5, 6]]
print(transpose_long(M))
