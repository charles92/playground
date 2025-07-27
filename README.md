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

## What's Included

- **requirements.txt**: Essential PyTorch dependencies
- **toy_example.py**: Simple neural network example that demonstrates:
  - Model creation with `nn.Module`
  - Data generation and preprocessing
  - Training loop with loss calculation
  - Model evaluation
  - Training visualization

## Project Structure

```
pytorch-playground/
├── requirements.txt   # Python dependencies
├── toy_example.py     # Simple PyTorch example
└── README.md          # This file
```

## Troubleshooting

- **CUDA not available**: The example will automatically use CPU
- **Import errors**: Make sure you've installed the requirements
- **Memory issues**: Reduce the number of samples in `generate_toy_data()`
