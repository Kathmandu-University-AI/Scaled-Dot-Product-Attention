# Scaled Dot-Product Attention using Pure Python Lists

This repository demonstrates **Scaled Dot-Product Attention** implemented entirely from scratch using only Python lists. It’s an educational example suitable for teaching or explaining attention mechanisms step-by-step.

## Features

- Implements `softmax`, `matmul`, and `transpose` using Python lists.
- Computes attention scores `QKᵀ / √dₖ` and applies softmax to obtain attention weights.
- Multiplies by `V` to get the attention output.
- Includes **causal mask** (for autoregressive decoding) and **padding mask** (for ignoring padded tokens).
- Visualizes attention weights using Matplotlib.

## Mathematical Recap

Given queries **Q**, keys **K**, and values **V**:

1. **Scores:** `S = QKᵀ`
2. **Scale:** `S' = S / sqrt(dₖ)`
3. **Mask:** Optionally apply a mask to block positions (causal or padding)
4. **Weights:** `A = softmax(S')`
5. **Output:** `O = AV`

## Example Setup

```python
Q = [[1., 0.],
     [0., 1.],
     [1., 1.]]

K = [[1., 0.5],
     [0.5, 1.],
     [1., -1.]]

V = [[1., 0.],
     [0., 1.],
     [1., 1.]]
```

Sequence length = 3, key dimension = 2.

## Visualizations

The script produces three heatmaps:
- **Unmasked attention**
- **Causal masked attention**
- **Padding masked attention**

These help visualize how masking changes attention distributions.

## Educational Use

This example is designed for clarity, not speed. It’s ideal for:
- Classroom demonstrations
- Step-by-step debugging
- Comparing with NumPy or PyTorch implementations

## License

[MIT License](https://github.com/sparshrestha/Scaled-Dot-Product-Attention/blob/main/LICENSE)
