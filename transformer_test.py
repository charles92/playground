import pytest
import torch

from transformer import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    FeedForward,
    PositionalEncoding,
    Transformer,
)


def test_positional_encoding_unbatched():
    seq_length, d_model = 100, 512
    x = torch.randn(seq_length, d_model)
    y = PositionalEncoding(seq_length, d_model)(x)
    assert y.shape == (seq_length, d_model)
    assert torch.norm(y - x) > 0


def test_positional_encoding_batched():
    batch_size, seq_length, d_model = 4, 100, 512
    x = torch.randn(batch_size, seq_length, d_model)
    y = PositionalEncoding(seq_length, d_model)(x)
    assert y.shape == (batch_size, seq_length, d_model)
    assert torch.norm(y - x) > 0


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
