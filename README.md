# playground

Contains miscellaneous code snippets for my own study & experimentation.

## Quick Start

```bash
pip install -r requirements.txt
```

## PyTorch mini example

After setting up the environment following [Quick Start](#quick-start) above, run:

```bash
python toy_example.py
```

This will:

- Create a simple neural network
- Generate toy classification data
- Train the model for 100 epochs
- Display training progress and final accuracy
- Show a plot of the training loss

## Transformer implementation

A complete transformer architecture implementation with encoder-decoder structure, following the
[_Attention Is All You Need_ paper](https://arxiv.org/abs/1706.03762) and the
[TensorFlow tutorial](https://www.tensorflow.org/text/tutorials/transformer).

Run the tests to verify all components work correctly:

```bash
python transformer_test.py
```

## What's Included

- **requirements.txt**: Essential PyTorch dependencies
- **toy_example.py**: Simple neural network example that demonstrates:
  - Model creation with `nn.Module`
  - Data generation and preprocessing
  - Training loop with loss calculation
  - Model evaluation
  - Training visualization
- **transformer.py**: Encoder-decoder transformer architecture implementation
- **visualize_positional_encoding.py**: Visualization script for positional encoding patterns

## Project Structure

```
playground/
├── requirements.txt                   # Python dependencies
├── toy_example.py                     # Simple PyTorch example
├── transformer.py                     # Transformer architecture implementation
├── transformer_test.py                # Unit tests for transformer
├── visualize_positional_encoding.py   # Positional encoding visualization
└── README.md                          # This file
```

## Troubleshooting

- **CUDA not available**: The example will automatically use CPU
- **Import errors**: Make sure you've installed the requirements
