import unittest

import torch

from transformer import (
    CausalSelfAttention,
    CrossAttention,
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    FeedForward,
    PositionalEncoding,
    SelfAttention,
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

    def test_self_attention(self):
        x = torch.rand((16, 102, 512))  # (batch, length, d_model)
        attn = SelfAttention(
            d_model=512,
            num_heads=8,
        )
        output = attn(x)
        self.assertEqual(output.shape, x.shape)

    def test_cross_attention(self):
        x = torch.rand((16, 102, 512))  # (batch, length, d_model)
        ctx = torch.rand((16, 143, 512))  # (batch, length, d_model)
        attn = CrossAttention(
            d_model=512,
            num_heads=8,
        )
        output = attn(x, ctx)
        self.assertEqual(output.shape, x.shape)

    def test_causal_self_attention(self):
        x = torch.rand((16, 102, 512))  # (batch, length, d_model)
        attn = CausalSelfAttention(
            d_model=512,
            num_heads=8,
        )
        output = attn(x)
        self.assertEqual(output.shape, x.shape)

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


if __name__ == "__main__":
    unittest.main()
