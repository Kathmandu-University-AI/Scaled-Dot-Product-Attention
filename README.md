# Scaled Dot-Product Attention using Pure Python Lists

This repository demonstrates **Scaled Dot-Product Attention** implemented entirely from scratch using only Python lists. It’s an educational example suitable for teaching or explaining attention mechanisms step-by-step.

## Features

- Implements `softmax`, `matmul`, and `transpose` using Python lists.
- Computes attention scores `QKᵀ / √dₖ` and applies softmax to obtain attention weights.
- Multiplies by `V` to get the attention output.
- Includes **causal mask** (for autoregressive decoding) and **padding mask** (for ignoring padded tokens).
- Visualizes attention weights using Matplotlib.
- Use very small matrices (seq_len=3, d_k=2) for maximum transparency.

## Mathematical Recap

Given queries **Q**, keys **K**, and values **V**:

1. **Scores:** `S = QKᵀ`
2. **Scale:** `S' = S / sqrt(dₖ)`
3. **Mask:(optional)** Apply a mask to block positions (causal or padding)
4. **Weights:** `A = softmax(S')`
5. **Output:** `O = AV`

## How to Run
Clone the repo:
```
git clone https://github.com/sparshrestha/Scaled-Dot-Product-Attention.git
```
```
cd Scaled-Dot-Product-Attention
```
Create a virtual environment (recommended):

macOS, Linux
```
python3 -m venv env
```
Windows
```
.\env\Scripts\activate and for linux source env/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```

Run the script:
```
python dot-product-attention.py
```
```
python dot-product-attention-sentence.py
```


## Example Setup
This mirrors the computation used inside transformer attention layers, but implemented in the simplest possible form.

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

Here the `sequence length is 3` and the `key/query dimensionality is 2`.

These tiny values make it easy to manually verify dot products, scaling, masking, and softmax behavior.

## Visualizations

The script generate heatmaps for:
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
