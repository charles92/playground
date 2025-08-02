"""Implements the basic transformer architecture.

Mostly just follows the TensorFlow tutorial: https://www.tensorflow.org/text/tutorials/transformer

Paper: Vaswani et al., 2017. Attention is all you need. https://arxiv.org/abs/1706.03762

Common parameters:
- `d_model`: primary model dimension, which is the length of the token embedding vectors.
- `d_ff`: dimension of the hidden layer in the feed-forward network.
- `d_out`: output vocabulary size. The final embedding vector of length `d_model` is projected to 
  logits of length `d_out`.
- `num_heads`: number of attention heads in each multi-head attention (MHA) module.
- `num_layers`: number of MHA layers in the encoder and decoder respectively.

In the original paper, the following parameters are used:

```
d_model = 512
d_ff = 2048
d_out = ~30k (depending on the language)
num_heads = 8
num_layers = 6
```
"""

from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchtext import datasets, transforms


class PositionalEncoding(nn.Module):
    """Positional encoding for the transformer model."""

    def __init__(self, max_seq_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        # Encoding matrix is (L, d_model) where L is the sequence length.
        positions = torch.arange(max_seq_len)[:, None]  # (L, 1)
        emb_dim = torch.arange(0, d_model, 2)[None, :] / d_model  # (1, d_model // 2)

        angular_rates = 1 / (10000**emb_dim)  # (1, d_model // 2)
        angles = positions * angular_rates  # (L, d_model // 2)

        self.pe = torch.zeros(max_seq_len, d_model, requires_grad=False)
        self.pe[:, 0::2] = torch.sin(angles)
        self.pe[:, 1::2] = torch.cos(angles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle both batched (B, L, d_model) and unbatched (L, d_model) cases
        seq_len = x.size(-2)
        return x + self.pe[:seq_len, :]


class BaseAttention(nn.Module):
    """Base attention module with multi-head attention and layer normalization."""

    def __init__(
        self, d_model: int, num_heads: int, dropout_rate: float = 0.1, **kwargs
    ):
        super().__init__(**kwargs)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)


class SelfAttention(BaseAttention):
    """Self-attention mechanism that attends to the same sequence."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.mha(query=x, key=x, value=x, need_weights=False)
        x += attn_out
        x = self.norm(x)
        return x


class CrossAttention(BaseAttention):
    """Cross-attention mechanism that attends to a different context sequence."""

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.mha(query=x, key=ctx, value=ctx, need_weights=False)
        x += attn_out
        x = self.norm(x)
        return x


class CausalSelfAttention(BaseAttention):
    """Causal self-attention mechanism with masking for autoregressive generation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: also consider padding mask.
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1
        )

        attn_out, _ = self.mha(
            query=x,
            key=x,
            value=x,
            need_weights=False,
            attn_mask=causal_mask,
            is_causal=True,
        )
        x += attn_out
        x = self.norm(x)
        return x


class FeedForward(nn.Module):
    """Feed-forward network with residual connection and layer normalization."""

    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.seq = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_rate),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc_out = self.seq(x)
        x = x + fc_out
        x = self.norm(x)
        return x


class EncoderLayer(nn.Module):
    """Single layer of the transformer encoder with self-attention and feed-forward."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attn = SelfAttention(d_model, num_heads, dropout_rate, **kwargs)
        self.ff = FeedForward(d_model, d_ff, dropout_rate, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.ff(x)
        return x


class Encoder(nn.Module):
    """Transformer encoder with multiple encoder layers."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = [
            EncoderLayer(d_model, d_ff, num_heads, dropout_rate, **kwargs)
            for _ in range(num_layers)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.emb(x)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderLayer(nn.Module):
    """Single layer of the transformer decoder with self- & cross-attention and feed-forward."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.self_attn = CausalSelfAttention(d_model, num_heads, dropout_rate, **kwargs)
        self.cross_attn = CrossAttention(d_model, num_heads, dropout_rate, **kwargs)
        self.ff = FeedForward(d_model, d_ff, dropout_rate, **kwargs)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        x = self.self_attn(x)
        x = self.cross_attn(x, ctx)
        x = self.ff(x)
        return x


class Decoder(nn.Module):
    """Transformer decoder with multiple decoder layers."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = [
            DecoderLayer(d_model, d_ff, num_heads, dropout_rate, **kwargs)
            for _ in range(num_layers)
        ]

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        x = self.emb(x)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, ctx)
        return x


class Transformer(nn.Module):
    """Complete transformer model with encoder-decoder architecture."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        d_out: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = Encoder(
            vocab_size,
            d_model,
            d_ff,
            num_heads,
            num_layers,
            max_seq_len,
            dropout_rate,
            **kwargs,
        )
        self.decoder = Decoder(
            vocab_size,
            d_model,
            d_ff,
            num_heads,
            num_layers,
            max_seq_len,
            dropout_rate,
            **kwargs,
        )
        self.linear = nn.Linear(d_model, d_out)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        ctx = self.encoder(ctx)
        x = self.decoder(x, ctx)
        x = self.linear(x)
        return x


class TokenDataset(Dataset):
    """Custom dataset wrapper for Multi30k data."""

    def __init__(self, data_iter, tokenizer: Callable[[str], list[str]] | None = None):
        # Convert iterator to list. We could have also tried an IterableDataset, which would
        # avoid this conversion (it only needs to support fetch next).
        self.data = list(data_iter)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]

        if self.tokenizer:
            # Tokenize the text if tokenizer is provided
            src_tokens = [int(tid) for tid in self.tokenizer(src)]
            tgt_tokens = [int(tid) for tid in self.tokenizer(tgt)]
            return src_tokens, tgt_tokens
        else:
            # Return raw text
            return src, tgt


def collate_fn(batch):
    """Custom collate function to handle variable length sequences."""
    src_batch, tgt_batch = zip(*batch)

    # For now, return as-is. In a real implementation, you'd want to:
    # 1. Pad sequences to the same length
    # 2. Convert to tensors
    # 3. Create attention masks
    return src_batch, tgt_batch


def load_dataset(batch_size=32, shuffle=True) -> tuple[DataLoader, DataLoader, int]:
    """Load the Multi30k dataset with German-English language pair.

    Returns:
        train_loader: DataLoader for the training set.
        valid_loader: DataLoader for the validation set.
        vocab_size: The size of the vocabulary.
    """

    # Load the Multi30k dataset with German-English language pair.
    print("Loading Multi30k dataset...")
    train_iter, valid_iter = datasets.Multi30k(
        root=".data", split=("train", "valid"), language_pair=("de", "en")
    )

    # Load the tokenizer.
    print("Loading BERT tokenizer...")
    tokenizer = transforms.BERTTokenizer(
        vocab_path="https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"
    )
    # TODO: get the vocabulary size from the tokenizer.
    _TOKEN_VOCAB_SIZE = 30522

    # Create dataset objects.
    train_dataset = TokenDataset(train_iter, tokenizer)
    valid_dataset = TokenDataset(valid_iter, tokenizer)

    # Create DataLoader objects.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn if tokenizer else None,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn if tokenizer else None,
    )

    print("Dataset loaded successfully!")
    print(f"Training set: {len(train_dataset)} examples")
    print(f"Validation set: {len(valid_dataset)} examples")

    # Print a few examples from the training set
    print("\nSample training examples:")
    for i in range(min(5, len(train_dataset))):
        src, tgt = train_dataset[i]
        print(f"Example {i+1}:")
        if tokenizer:
            print(f"  German tokens: {src}")
            print(f"  English tokens: {tgt}")
        else:
            print(f"  German: {src}")
            print(f"  English: {tgt}")
        print()

    return train_loader, valid_loader, _TOKEN_VOCAB_SIZE


def main() -> None:
    # Uncomment the following lines when torchtext is properly installed
    train_loader, valid_loader, vocab_size = load_dataset(batch_size=16)

    # Test the DataLoader
    print("\nTesting DataLoader:")
    for batch_idx, (src_batch, tgt_batch) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Batch size: {len(src_batch)}")
        print(f"  Sample German: {src_batch[0]}")
        print(f"  Sample English: {tgt_batch[0]}")
        if batch_idx >= 2:  # Only show first 3 batches
            break


if __name__ == "__main__":
    main()
