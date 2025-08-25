import pytest
import torch

from transformer import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    FeedForward,
    MultiHeadAttention,
    SinusoidalPositionalEncoding,
    TrainablePositionalEncoding,
    Transformer,
)


@pytest.mark.parametrize("q_len", [10, 20])
@pytest.mark.parametrize("kv_len", [10, 20])
@pytest.mark.parametrize("num_heads", [1, 4])
def test_mha_output_shape(q_len, kv_len, num_heads):
    batch_size, d_model = 4, 64
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    mha.eval()

    q = torch.randn(batch_size, q_len, d_model)
    k = torch.randn(batch_size, kv_len, d_model)
    v = torch.randn(batch_size, kv_len, d_model)

    output = mha(q, k, v)

    assert output.shape == (batch_size, q_len, d_model)


@pytest.mark.parametrize("d_key", [16, 32])
@pytest.mark.parametrize("d_value", [16, 32])
def test_mha_custom_qkv_dimensions_output_shape(d_key, d_value):
    batch_size, d_model = 4, 64
    q_len, kv_len = 10, 20

    mha = MultiHeadAttention(d_model=d_model, num_heads=8, d_key=d_key, d_value=d_value)
    mha.eval()

    # Check that custom dimensions are set correctly
    assert mha.d_qk == d_key
    assert mha.d_v == d_value

    q = torch.randn(batch_size, q_len, d_model)
    k = torch.randn(batch_size, kv_len, d_model)
    v = torch.randn(batch_size, kv_len, d_model)

    output = mha(q, k, v)

    # Check output shape
    assert output.shape == (batch_size, q_len, d_model)


@pytest.mark.parametrize(
    "mask",
    [
        torch.tensor([[False, False, True, True]] * 2, dtype=torch.bool),
        torch.tensor([[0.0, 0.0, -torch.inf, -torch.inf]] * 2, dtype=torch.float32),
    ],
)
def test_mha_padding_mask(mask):
    batch_size, seq_len, d_model = 2, 4, 64
    num_heads = 8

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    mha.eval()

    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)

    y = mha(q, k, v, key_padding_mask=mask)

    # Tweaking the masked part of the key shouldn't affect the output.
    masked_k = k.clone()
    masked_k[:, 2:, :] = 0.0
    y_masked_k = mha(q, masked_k, v, key_padding_mask=mask)
    assert torch.allclose(y, y_masked_k)

    # Tweaking the masked part of the value shouldn't affect the output.
    masked_v = v.clone()
    masked_v[:, 2:, :] = 0.0
    y_masked_v = mha(q, k, masked_v, key_padding_mask=mask)
    assert torch.allclose(y, y_masked_v)


@pytest.mark.parametrize(
    "mask",
    [
        torch.ones(10, 10, dtype=torch.bool).triu(diagonal=1),
        torch.full((10, 10), -torch.inf, dtype=torch.float32).triu(diagonal=1),
    ],
)
def test_mha_bool_causal_attention_mask(mask):
    batch_size, seq_len, d_model = 4, 10, 64
    num_heads = 8

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    mha.eval()

    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)

    y = mha(q, k, v, attn_mask=mask)

    # Tweaking the second half of the key shouldn't affect the first half of the output.
    masked_k = k.clone()
    masked_k[:, -3:, :] = 0.0
    y_masked_k = mha(q, masked_k, v, attn_mask=mask)
    assert torch.allclose(y[:, :-3, :], y_masked_k[:, :-3, :])
    assert not torch.allclose(y[:, -3:, :], y_masked_k[:, -3:, :])

    # Tweaking the second half of the value shouldn't affect the first half of the output.
    masked_v = v.clone()
    masked_v[:, -3:, :] = 0.0
    y_masked_v = mha(q, k, masked_v, attn_mask=mask)
    assert torch.allclose(y[:, :-3, :], y_masked_v[:, :-3, :])
    assert not torch.allclose(y[:, -3:, :], y_masked_v[:, -3:, :])


def test_mha_gradients():
    batch_size, seq_len, d_model = 4, 10, 64

    mha = MultiHeadAttention(d_model=d_model, num_heads=8)

    query = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    key = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    value = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    y = mha(query, key, value)

    # Compute loss and gradients
    loss = y.sum()
    loss.backward()

    # Check gradient shapes
    assert query.grad.shape == query.shape
    assert key.grad.shape == key.shape
    assert value.grad.shape == value.shape


@pytest.mark.parametrize(
    "ctor", [SinusoidalPositionalEncoding, TrainablePositionalEncoding]
)
@pytest.mark.parametrize("seq_length", [1, 4, 8])
def test_positional_encoding_unbatched(ctor, seq_length):
    layer = ctor(8, 16)
    layer.eval()

    x = torch.randn(seq_length, 16)
    y = layer(x)
    assert y.shape == x.shape
    assert torch.norm(y - x) > 0


@pytest.mark.parametrize(
    "ctor", [SinusoidalPositionalEncoding, TrainablePositionalEncoding]
)
@pytest.mark.parametrize("seq_length", [1, 4, 8])
def test_positional_encoding_batched(ctor, seq_length):
    layer = ctor(8, 16)
    layer.eval()

    x = torch.randn(4, seq_length, 16)
    y = layer(x)
    assert y.shape == x.shape

    encoding = y - x
    print(encoding)
    assert torch.norm(encoding) > 0
    assert torch.allclose(encoding[0, ...], encoding[1, ...], atol=1e-6)


def test_feed_forward_output_shape():
    x = torch.rand((16, 102, 512))  # (batch, length, d_model)
    ff = FeedForward(
        d_model=512,
        d_ff=2048,
    )
    output = ff(x)
    assert output.shape == x.shape


def test_encoder_layer_output_shape():
    x = torch.rand((16, 143, 512))  # (batch, length, d_model)
    layer = EncoderLayer(
        d_model=512,
        d_ff=2048,
        num_heads=8,
    )
    output = layer(x)
    assert output.shape == x.shape


def test_encoder_output_shape():
    x = torch.randint(0, 1000, (16, 143))  # (batch, length)
    encoder = Encoder(
        vocab_size=1000,
        d_model=512,
        d_ff=2048,
        num_heads=8,
        num_layers=6,
        max_seq_len=256,
    )
    output = encoder(x)
    assert output.shape == (16, 143, 512)


def test_decoder_layer_output_shape():
    x = torch.rand((16, 102, 512))  # (batch, length, d_model)
    ctx = torch.rand((16, 143, 512))  # (batch, length, d_model)
    layer = DecoderLayer(
        d_model=512,
        d_ff=2048,
        num_heads=8,
    )
    output = layer(x, ctx)
    assert output.shape == x.shape


def test_decoder_output_shape():
    x = torch.randint(0, 1000, (16, 102))  # (batch, length)
    ctx = torch.rand((16, 143, 512))  # (batch, length, d_model)
    decoder = Decoder(
        vocab_size=1000,
        d_model=512,
        d_ff=2048,
        num_heads=8,
        num_layers=6,
        max_seq_len=256,
    )
    output = decoder(x, ctx)
    assert output.shape == (16, 102, 512)


def test_transformer_output_shape():
    x = torch.randint(0, 1000, (16, 102))  # (batch, length)
    ctx = torch.randint(0, 1000, (16, 143))  # (batch, length)
    transformer = Transformer(
        vocab_size=1000,
        d_model=512,
        d_ff=2048,
        d_out=1000,
        num_heads=8,
        num_layers=6,
        max_seq_len=256,
    )
    output = transformer(x, ctx)
    assert output.shape == (16, 102, 1000)


def test_transformer_output_shape_no_grad():
    x = torch.randint(0, 1000, (16, 102))  # (batch, length)
    ctx = torch.randint(0, 1000, (16, 143))  # (batch, length)
    transformer = Transformer(
        vocab_size=1000,
        d_model=512,
        d_ff=2048,
        d_out=1000,
        num_heads=8,
        num_layers=6,
        max_seq_len=256,
    )

    with torch.no_grad():
        output = transformer(x, ctx)
        assert output.shape == (16, 102, 1000)


@pytest.fixture
def padding_test_data():
    """Fixture providing test data for padding mask tests."""
    x_tokens = torch.randint(0, 1000, (16, 102))  # (batch, length)
    x = torch.rand((16, 102, 512))  # (batch, length, d_model)
    mask = torch.ones((16, 102), dtype=torch.int)
    mask[:, 80:] = 0  # Last 22 tokens are padding

    ctx_tokens = torch.randint(0, 1000, (16, 143))  # (batch, ctx_length)
    ctx = torch.rand((16, 143, 512))  # (batch, ctx_length, d_model)
    ctx_mask = torch.ones((16, 143), dtype=torch.int)
    ctx_mask[:, 120:] = 0  # Last 23 context tokens are padding

    return {
        "x_tokens": x_tokens,
        "x": x,
        "mask": mask,
        "ctx_tokens": ctx_tokens,
        "ctx": ctx,
        "ctx_mask": ctx_mask,
    }


def test_encoder_layer_with_padding_output_shape(padding_test_data):
    data = padding_test_data
    layer = EncoderLayer(
        d_model=512,
        d_ff=2048,
        num_heads=8,
    )
    output = layer(data["x"], data["mask"])
    assert output.shape == data["x"].shape


def test_encoder_with_padding_output_shape(padding_test_data):
    data = padding_test_data
    encoder = Encoder(
        vocab_size=1000,
        d_model=512,
        d_ff=2048,
        num_heads=8,
        num_layers=6,
        max_seq_len=256,
    )
    output = encoder(data["x_tokens"], data["mask"])
    assert output.shape == (16, 102, 512)


def test_decoder_layer_with_padding_output_shape(padding_test_data):
    data = padding_test_data
    layer = DecoderLayer(
        d_model=512,
        d_ff=2048,
        num_heads=8,
    )
    output = layer(data["x"], data["ctx"], data["mask"], data["ctx_mask"])
    assert output.shape == data["x"].shape


def test_decoder_with_padding_output_shape(padding_test_data):
    data = padding_test_data
    decoder = Decoder(
        vocab_size=1000,
        d_model=512,
        d_ff=2048,
        num_heads=8,
        num_layers=6,
        max_seq_len=256,
    )
    output = decoder(data["x_tokens"], data["ctx"], data["mask"], data["ctx_mask"])
    assert output.shape == (16, 102, 512)


def test_transformer_with_padding_output_shape(padding_test_data):
    data = padding_test_data
    transformer = Transformer(
        vocab_size=1000,
        d_model=512,
        d_ff=2048,
        d_out=1000,
        num_heads=8,
        num_layers=6,
        max_seq_len=256,
    )
    output = transformer(
        data["x_tokens"], data["ctx_tokens"], data["mask"], data["ctx_mask"]
    )
    assert output.shape == (16, 102, 1000)


def test_encoder_layer_padding_mask_works():
    # Construct x1, x2 such that they are the same in the first few positions.
    x1 = torch.rand((2, 8, 4))  # (batch, length, d_model)
    x1[:, 6:, :] = 0.0
    x2 = x1.clone()
    x2[:, 6:, :] = 1.0

    # The remaining positions are marked as padding.
    mask = torch.ones((2, 8), dtype=torch.int)
    mask[:, 6:] = 0

    layer = EncoderLayer(
        d_model=4,
        d_ff=16,
        num_heads=2,
    )
    # Disable dropout, etc.
    layer.eval()

    y1 = layer(x1, mask)
    y2 = layer(x2, mask)
    # The output for non-padded positions should be the same
    torch.testing.assert_close(y1[:, :6, :], y2[:, :6, :])


def test_decoder_layer_x_padding_mask_works():
    # Construct x1, x2 such that they are the same in the first few positions.
    x1 = torch.rand((2, 8, 4))  # (batch, length, d_model)
    x1[:, 6:, :] = 0.0
    x2 = x1.clone()
    x2[:, 6:, :] = 1.0

    # The remaining positions are marked as padding.
    mask = torch.ones((2, 8), dtype=torch.int)
    mask[:, 6:] = 0

    ctx = torch.rand((2, 8, 4))  # (batch, ctx_length, d_model)

    layer = DecoderLayer(
        d_model=4,
        d_ff=16,
        num_heads=2,
    )
    # Disable dropout, etc.
    layer.eval()

    y1 = layer(x1, ctx, mask=mask)
    y2 = layer(x2, ctx, mask=mask)
    # The output for non-padded positions should be the same.
    torch.testing.assert_close(y1[:, :6, :], y2[:, :6, :])


def test_decoder_layer_ctx_padding_mask_works():
    x = torch.rand((2, 8, 4))  # (batch, length, d_model)

    # Construct ctx1, ctx2 such that they are the same in the first few positions.
    ctx1 = torch.rand((2, 8, 4))  # (batch, ctx_length, d_model)
    ctx1[:, 6:, :] = 0.0
    ctx2 = ctx1.clone()
    ctx2[:, 6:, :] = 1.0

    # Mark the remaining positions as padding.
    ctx_mask = torch.ones((2, 8), dtype=torch.int)
    ctx_mask[:, 6:] = 0

    layer = DecoderLayer(
        d_model=4,
        d_ff=16,
        num_heads=2,
    )
    # Disable dropout, etc.
    layer.eval()

    y1 = layer(x, ctx1, ctx_mask=ctx_mask)
    y2 = layer(x, ctx2, ctx_mask=ctx_mask)
    # The output should be the same since context padding shouldn't affect non-padded positions.
    torch.testing.assert_close(y1, y2)


def test_decoder_layer_causal_mask_works():
    # Construct x1, x2 such that they are the same in the first few positions.
    x1 = torch.rand((2, 8, 4))  # (batch, length, d_model)
    x1[:, 4:, :] = 1.0
    x2 = x1.clone()
    x2[:, 4:, :] = 0.0

    ctx = torch.rand((2, 8, 4))  # (batch, ctx_length, d_model)

    layer = DecoderLayer(
        d_model=4,
        d_ff=16,
        num_heads=2,
    )
    # Disable dropout, etc.
    layer.eval()

    # Test that causal masking works: changing later positions shouldn't affect earlier outputs
    y1 = layer(x1, ctx)
    y2 = layer(x2, ctx)

    # The first 4 positions should be identical since causal masking prevents later positions
    # from affecting earlier ones.
    torch.testing.assert_close(y1[:, :4, :], y2[:, :4, :])
