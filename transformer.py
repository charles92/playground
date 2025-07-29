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

import torch
import torch.nn as nn


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
        **kwargs
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
        d_model: int,
        d_ff: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = [
            EncoderLayer(d_model, d_ff, num_heads, dropout_rate, **kwargs)
            for _ in range(num_layers)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        **kwargs
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
        d_model: int,
        d_ff: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = [
            DecoderLayer(d_model, d_ff, num_heads, dropout_rate, **kwargs)
            for _ in range(num_layers)
        ]

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, ctx)
        return x


class Transformer(nn.Module):
    """Complete transformer model with encoder-decoder architecture."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        d_out: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = Encoder(
            d_model, d_ff, num_heads, num_layers, dropout_rate, **kwargs
        )
        self.decoder = Decoder(
            d_model, d_ff, num_heads, num_layers, dropout_rate, **kwargs
        )
        self.linear = nn.Linear(d_model, d_out)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        ctx = self.encoder(ctx)
        x = self.decoder(x, ctx)
        x = self.linear(x)
        return x
