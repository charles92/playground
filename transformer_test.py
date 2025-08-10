import unittest

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


class PositionalEncodingTest(unittest.TestCase):

    def test_positional_encoding_unbatched(self):
        seq_length, d_model = 100, 512
        x = torch.randn(seq_length, d_model)
        y = PositionalEncoding(seq_length, d_model)(x)
        self.assertEqual(y.shape, (seq_length, d_model))
        self.assertGreater(torch.norm(y - x), 0)

    def test_positional_encoding_batched(self):
        batch_size, seq_length, d_model = 4, 100, 512
        x = torch.randn(batch_size, seq_length, d_model)
        y = PositionalEncoding(seq_length, d_model)(x)
        self.assertEqual(y.shape, (batch_size, seq_length, d_model))
        self.assertGreater(torch.norm(y - x), 0)


class OutputShapeTest(unittest.TestCase):

    def test_feed_forward(self):
        x = torch.rand((16, 102, 512))  # (batch, length, d_model)
        ff = FeedForward(
            d_model=512,
            d_ff=2048,
        )
        output = ff(x)
        self.assertEqual(output.shape, x.shape)

    def test_encoder_layer(self):
        x = torch.rand((16, 143, 512))  # (batch, length, d_model)
        layer = EncoderLayer(
            d_model=512,
            d_ff=2048,
            num_heads=8,
        )
        output = layer(x)
        self.assertEqual(output.shape, x.shape)

    def test_encoder(self):
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
        self.assertEqual(output.shape, (16, 143, 512))

    def test_decoder_layer(self):
        x = torch.rand((16, 102, 512))  # (batch, length, d_model)
        ctx = torch.rand((16, 143, 512))  # (batch, length, d_model)
        layer = DecoderLayer(
            d_model=512,
            d_ff=2048,
            num_heads=8,
        )
        output = layer(x, ctx)
        self.assertEqual(output.shape, x.shape)

    def test_decoder(self):
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
        self.assertEqual(output.shape, (16, 102, 512))

    def test_transformer(self):
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
        self.assertEqual(output.shape, (16, 102, 1000))

    def test_transformer_no_grad(self):
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
            self.assertEqual(output.shape, (16, 102, 1000))


class OutputShapeWithPaddingMaskTest(unittest.TestCase):

    def setUp(self):
        self.x_tokens = torch.randint(0, 1000, (16, 102))  # (batch, length)
        self.x = torch.rand((16, 102, 512))  # (batch, length, d_model)
        self.mask = torch.ones((16, 102), dtype=torch.int)
        self.mask[:, 80:] = 0  # Last 22 tokens are padding

        self.ctx_tokens = torch.randint(0, 1000, (16, 143))  # (batch, ctx_length)
        self.ctx = torch.rand((16, 143, 512))  # (batch, ctx_length, d_model)
        self.ctx_mask = torch.ones((16, 143), dtype=torch.int)
        self.ctx_mask[:, 120:] = 0  # Last 23 context tokens are padding

    def test_encoder_layer(self):
        layer = EncoderLayer(
            d_model=512,
            d_ff=2048,
            num_heads=8,
        )
        output = layer(self.x, self.mask)
        self.assertEqual(output.shape, self.x.shape)

    def test_encoder(self):
        encoder = Encoder(
            vocab_size=1000,
            d_model=512,
            d_ff=2048,
            num_heads=8,
            num_layers=6,
            max_seq_len=256,
        )
        output = encoder(self.x_tokens, self.mask)
        self.assertEqual(output.shape, (16, 102, 512))

    def test_decoder_layer(self):
        layer = DecoderLayer(
            d_model=512,
            d_ff=2048,
            num_heads=8,
        )
        output = layer(self.x, self.ctx, self.mask, self.ctx_mask)
        self.assertEqual(output.shape, self.x.shape)

    def test_decoder(self):
        decoder = Decoder(
            vocab_size=1000,
            d_model=512,
            d_ff=2048,
            num_heads=8,
            num_layers=6,
            max_seq_len=256,
        )
        output = decoder(self.x_tokens, self.ctx, self.mask, self.ctx_mask)
        self.assertEqual(output.shape, (16, 102, 512))

    def test_transformer(self):
        transformer = Transformer(
            vocab_size=1000,
            d_model=512,
            d_ff=2048,
            d_out=1000,
            num_heads=8,
            num_layers=6,
            max_seq_len=256,
        )
        output = transformer(self.x_tokens, self.ctx_tokens, self.mask, self.ctx_mask)
        self.assertEqual(output.shape, (16, 102, 1000))


class AttentionMaskTest(unittest.TestCase):

    def test_encoder_layer_padding_mask(self):
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

    def test_decoder_layer_x_padding_mask(self):
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

    def test_decoder_layer_ctx_padding_mask(self):
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

    def test_decoder_layer_causal_mask(self):
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


if __name__ == "__main__":
    unittest.main()
