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

# TODO: Learned positional encoding.


class MultiHeadAttention(nn.Module):
    """Hand-crafted multi-head attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        d_key: int | None = None,
        d_value: int | None = None,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads

        # Use default Q, K, V dimensions if not specified.
        d_head = d_model // num_heads
        self.d_qk = d_key or d_head
        self.d_v = d_value or d_head

        # Linear projections for query, key, and value. The projection matrices already take into
        # account the multi-head concatenation. Hence the `num_heads` factor.
        self.q_proj = nn.Linear(d_model, self.d_qk * num_heads)
        self.k_proj = nn.Linear(d_model, self.d_qk * num_heads)
        self.v_proj = nn.Linear(d_model, self.d_v * num_heads)

        # Linear projection for the output.
        self.o_proj = nn.Linear(self.d_v * num_heads, d_model)

        # Output dropout.
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query: Query tensor of shape (batch_size, q_len, d_model).
            key: Key tensor of shape (batch_size, kv_len, d_model).
            value: Value tensor of shape (batch_size, kv_len, d_model).
            key_padding_mask: Validity mask of shape (batch_size, kv_len) indicating which entries
              in the key are invalid (e.g., padding). For boolean masks, true indicates that
              position will be ignored for the purpose of attention. For float masks, the mask
              values are added directly to the pre-softmax attention scores.
            attn_mask: Attention mask of shape (batch_size, q_len, kv_len). Both boolean and float
              masks are supported, similar to `key_padding_mask`.

        Returns:
            Output tensor of shape (batch_size, q_len, d_model).
        """
        batch_size, q_len, _ = query.size()
        _, kv_len, _ = key.size()

        q = self.q_proj(query)  # (B, q_len, d_qk * num_heads)
        k = self.k_proj(key)  # (B, kv_len, d_qk * num_heads)
        v = self.v_proj(value)  # (B, kv_len, d_v * num_heads)

        # Transpose q & k such that the first two dimensions are (B, num_heads), and the last two
        # dimensions are dot-producted.
        q = q.view(batch_size, q_len, self.num_heads, self.d_qk).transpose(-2, -3)
        k = k.view(batch_size, kv_len, self.num_heads, self.d_qk).transpose(-2, -3)

        # (B, num_heads, q_len, kv_len)
        attn_score = (q @ k.transpose(-1, -2)) * torch.rsqrt(
            torch.tensor(self.d_qk, dtype=torch.float32, device=query.device)
        )

        # Process attention masks.
        if attn_mask is None:
            attn_mask = torch.zeros(batch_size, q_len, kv_len, device=query.device)
        elif attn_mask.dtype == torch.bool:
            attn_mask = torch.where(attn_mask, -torch.inf, 0.0)
        if key_padding_mask is not None:
            if key_padding_mask.dtype == torch.bool:
                key_padding_mask = torch.where(key_padding_mask, -torch.inf, 0.0)
            key_padding_mask = key_padding_mask.unsqueeze(-2)  # (B, 1, kv_len)
            attn_mask = attn_mask + key_padding_mask

        # Compute final attention scores.
        attn_mask = attn_mask.unsqueeze(-3)  # (B, 1, q_len, kv_len)
        attn_score = torch.softmax(
            attn_score + attn_mask, dim=-1
        )  # (B, num_heads, q_len, kv_len)

        # Transpose v to (B, num_heads, kv_len, d_v).
        v = v.view(batch_size, kv_len, self.num_heads, self.d_v).transpose(-2, -3)

        # Apply attention scores to the value vectors, concatenate the heads, and project to the
        # output dimension.
        output = attn_score @ v  # (B, num_heads, q_len, d_v)
        output = output.transpose(-2, -3).reshape(
            batch_size, q_len, self.d_v * self.num_heads
        )
        output = self.o_proj(output)
        output = self.dropout(output)

        return output


class SinusoidalPositionalEncoding(nn.Module):
    """Original sinusoidal positional encoding for the transformer model."""

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


class TrainablePositionalEncoding(nn.Module):
    """Learned positional encoding for the transformer model."""

    def __init__(self, max_seq_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.pe = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(-2)
        positions = torch.arange(seq_len, device=x.device)
        if x.dim() > 2:
            positions = positions[None, :]  # (1, seq_len)
        return x + self.pe(positions)


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
        self.self_attn = MultiHeadAttention(
            d_model, num_heads, dropout_rate=dropout_rate, **kwargs
        )
        self.norm = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout_rate, **kwargs)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            mask: Validity mask of shape (batch_size, seq_len), where 0 means invalid.

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Convert validity mask (0 means invalid - ignored) to padding mask (true means padding -
        # ignored).
        if mask is not None:
            mask = ~mask.bool()

        # Global self-attention.
        attn_out = self.self_attn(query=x, key=x, value=x, key_padding_mask=mask)
        x = self.norm(x + attn_out)

        # Feed-forward.
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
        self.pe = TrainablePositionalEncoding(max_seq_len, d_model)
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
        self.self_attn = MultiHeadAttention(
            d_model, num_heads, dropout_rate=dropout_rate, **kwargs
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(
            d_model, num_heads, dropout_rate=dropout_rate, **kwargs
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout_rate, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        mask: torch.Tensor | None = None,
        ctx_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            ctx: Context tensor of shape (batch_size, ctx_seq_len, d_model).
            mask: Validity mask of shape (batch_size, seq_len), where 0 means invalid.
            ctx_mask: Validity mask of shape (batch_size, ctx_seq_len), where 0 means invalid.

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Convert validity mask (0 means invalid - ignored) to padding mask (true means padding -
        # ignored).
        if mask is not None:
            mask = ~mask.bool()
        if ctx_mask is not None:
            ctx_mask = ~ctx_mask.bool()

        # Causal self-attention.
        seq_len = x.size(1)
        causal_mask = torch.ones(
            seq_len, seq_len, dtype=torch.bool, device=x.device, requires_grad=False
        ).triu(diagonal=1)

        attn_out = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=mask,
            attn_mask=causal_mask,
        )
        x = self.norm1(x + attn_out)

        # Cross-attention.
        attn_out = self.cross_attn(
            query=x,
            key=ctx,
            value=ctx,
            key_padding_mask=ctx_mask,
        )
        x = self.norm2(x + attn_out)

        # Feed-forward.
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
        self.pe = TrainablePositionalEncoding(max_seq_len, d_model)
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
            logits = model(tgt_input, src, tgt_input_mask, src_mask)

            # Reshape output and target for loss calculation
            logits = logits.view(-1, vocab_size)  # (B * L, vocab_size)
            labels = tgt_target.reshape(-1)  # (B * L)

            # Calculate loss
            loss = criterion(logits, labels)

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

                logits = model(tgt_input, src, tgt_input_mask, src_mask)
                logits = logits.view(-1, vocab_size)
                labels = tgt_target.reshape(-1)

                loss = criterion(logits, labels)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        print(f"Validation loss: {avg_val_loss:.4f}")
        print("-" * 50)

    print("Training completed!")


if __name__ == "__main__":
    main()
