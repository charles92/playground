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

import datasets
import torch
import torch.nn as nn
import transformers
from torch.utils import data


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
        return x + self.pe[:seq_len, :].to(x.device)


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

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            mask: Padding mask of shape (batch_size, seq_len), where 1 indicates a valid token and
                0 indicates padding.
        """
        if mask is not None:
            # MHA's key_padding_mask expects True to mark padding (invalid) tokens.
            mask = ~mask.bool()
        attn_out, _ = self.mha(
            query=x, key=x, value=x, key_padding_mask=mask, need_weights=False
        )
        # Note: in-place addition "x += attn_out" is NOT equivalent, and produces unexpected
        # results. See https://github.com/pytorch/pytorch/issues/8212.
        x = x + attn_out
        x = self.norm(x)
        return x


class CrossAttention(BaseAttention):
    """Cross-attention mechanism that attends to a different context sequence."""

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        ctx_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            ctx: Context tensor of shape (batch_size, ctx_len, d_model).
            ctx_mask: Padding mask of shape (batch_size, ctx_len), where 1 indicates a valid token
                and 0 indicates padding.
        """
        if ctx_mask is not None:
            # MHA's key_padding_mask expects True to mark padding (invalid) tokens.
            ctx_mask = ~ctx_mask.bool()
        attn_out, _ = self.mha(
            query=x, key=ctx, value=ctx, key_padding_mask=ctx_mask, need_weights=False
        )
        # Note: in-place addition "x += attn_out" is NOT equivalent, and produces unexpected
        # results. See https://github.com/pytorch/pytorch/issues/8212.
        x = x + attn_out
        x = self.norm(x)
        return x


class CausalSelfAttention(BaseAttention):
    """Causal self-attention mechanism with masking for autoregressive generation."""

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            mask: Padding mask of shape (batch_size, seq_len), where 1 indicates a valid token and
                0 indicates padding.
        """
        if mask is not None:
            # MHA's key_padding_mask expects True to mark padding (invalid) tokens.
            mask = ~mask.bool()

        # Causal attention mask.
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1
        )

        attn_out, _ = self.mha(
            query=x,
            key=x,
            value=x,
            key_padding_mask=mask,
            need_weights=False,
            attn_mask=causal_mask,
            is_causal=True,
        )
        # Note: in-place addition "x += attn_out" is NOT equivalent, and produces unexpected
        # results. See https://github.com/pytorch/pytorch/issues/8212.
        x = x + attn_out
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

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.attn(x, mask)
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
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, d_ff, num_heads, dropout_rate, **kwargs)
            for _ in range(num_layers)
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.emb(x)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
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

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        mask: torch.Tensor | None = None,
        ctx_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.self_attn(x, mask)
        x = self.cross_attn(x, ctx, ctx_mask)
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
        self.layers = nn.ModuleList(
            DecoderLayer(d_model, d_ff, num_heads, dropout_rate, **kwargs)
            for _ in range(num_layers)
        )

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        mask: torch.Tensor | None = None,
        ctx_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.emb(x)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, ctx, mask, ctx_mask)
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

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        mask: torch.Tensor | None = None,
        ctx_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ctx = self.encoder(ctx, ctx_mask)
        x = self.decoder(x, ctx, mask, ctx_mask)
        x = self.linear(x)
        return x


def load_dataset(
    batch_size=32, shuffle=True
) -> tuple[data.DataLoader, data.DataLoader, int]:
    """Load the Multi30k dataset with German-English language pair.

    Returns:
        train_loader: DataLoader for the training set.
        valid_loader: DataLoader for the validation set.
        vocab_size: The size of the vocabulary.
    """

    # Load the Multi30k dataset with German-English language pair.
    print("Loading Multi30k dataset...")
    train_ds = datasets.load_dataset("bentrevett/multi30k", split="train")
    valid_ds = datasets.load_dataset("bentrevett/multi30k", split="validation")

    print("Dataset loaded successfully!")
    print(f"Training set: {len(train_ds)} examples")
    print(f"Validation set: {len(valid_ds)} examples")

    # Print a few examples from the training set
    print("\nSample training examples:")
    for i in range(min(5, len(train_ds))):
        src = train_ds[i]["de"]
        tgt = train_ds[i]["en"]
        # print(train_ds[i])
        print(f"Example {i+1}:")
        print(f"  German text: {src}")
        print(f"  English text: {tgt}")
        print()

    # Load the tokenizer.
    print("Loading BERT tokenizer...")
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

    def _string_to_tokens(
        batch: list[dict[str, str]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Converts batch of source-target pairs into batches of tokens and attention masks.

        Returns:
            - Source token IDs.
            - Target token IDs.
            - Source padding masks, where 1 indicates a valid token and 0 indicates padding.
            - Target padding masks, where 1 indicates a valid token and 0 indicates padding.
        """
        src_batch = [pair["de"] for pair in batch]
        tgt_batch = [pair["en"] for pair in batch]
        src_tokens = tokenizer(
            text=src_batch,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_attention_mask=True,
            return_tensors="pt",
        )
        tgt_tokens = tokenizer(
            text=tgt_batch,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return (
            src_tokens["input_ids"],
            tgt_tokens["input_ids"],
            src_tokens["attention_mask"],
            tgt_tokens["attention_mask"],
        )

    # Create DataLoader objects.
    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_string_to_tokens,
    )

    valid_loader = data.DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_string_to_tokens,
    )

    return train_loader, valid_loader, len(tokenizer)


def main() -> None:
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.device_count() > 0:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load dataset
    train_loader, valid_loader, vocab_size = load_dataset(batch_size=16)
    print(f"Vocabulary size: {vocab_size}")

    # Initialize model
    model = Transformer(
        vocab_size=vocab_size,
        d_model=256,
        d_ff=1024,
        d_out=vocab_size,
        num_heads=4,
        num_layers=4,
        max_seq_len=64,
        dropout_rate=0.1,
    ).to(device)

    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is padding token

    # Training loop
    num_epochs = 2
    print(f"\nStarting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (src, tgt, src_mask, tgt_mask) in enumerate(train_loader):
            # Move data to device
            src = src.to(device)
            tgt = tgt.to(device)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)

            # Forward pass
            # For training, we use teacher forcing - input is tgt[:-1], target is tgt[1:]
            tgt_input = tgt[:, :-1]  # Remove last token
            tgt_target = tgt[:, 1:]  # Remove first token (BOS)
            tgt_input_mask = tgt_mask[:, :-1]

            optimizer.zero_grad()
            output = model(tgt_input, src, tgt_input_mask, src_mask)

            # Reshape output and target for loss calculation
            output = output.view(-1, vocab_size)
            target = tgt_target.reshape(-1)

            # Calculate loss
            loss = criterion(output, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Calculate average loss for the epoch
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for src, tgt, src_mask, tgt_mask in valid_loader:
                src = src.to(device)
                tgt = tgt.to(device)
                src_mask = src_mask.to(device)
                tgt_mask = tgt_mask.to(device)

                tgt_input = tgt[:, :-1]
                tgt_target = tgt[:, 1:]
                tgt_input_mask = tgt_mask[:, :-1]

                output = model(tgt_input, src, tgt_input_mask, src_mask)
                output = output.view(-1, vocab_size)
                target = tgt_target.reshape(-1)

                loss = criterion(output, target)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        print(f"Validation loss: {avg_val_loss:.4f}")
        print("-" * 50)

    print("Training completed!")


if __name__ == "__main__":
    main()
