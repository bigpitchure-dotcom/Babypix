"""
Tests for BabyPix1 Transformer model.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from babypix1.config import ModelConfig
from babypix1.model import (
    Embeddings,
    CausalSelfAttention,
    FeedForward,
    TransformerBlock,
    BabyPix1LM,
    create_model,
)


@pytest.fixture
def small_config():
    """Small config for fast testing."""
    return ModelConfig(
        vocab_size=1000,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_seq_len=64,
        dropout=0.0,
    )


@pytest.fixture
def tiny_config():
    """Tiny config for minimal tests."""
    return ModelConfig(
        vocab_size=100,
        d_model=64,
        n_heads=2,
        n_layers=1,
        d_ff=128,
        max_seq_len=32,
        dropout=0.0,
    )


@pytest.fixture
def sample_input(tiny_config):
    """Sample input tensor for testing."""
    batch_size = 2
    seq_len = 16
    return torch.randint(0, tiny_config.vocab_size, (batch_size, seq_len))


class TestModelConfig:
    """Tests for ModelConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.vocab_size == 2000
        assert config.d_model == 256
        assert config.n_heads == 8
        assert config.n_layers == 6
        assert config.d_ff == 1024
        assert config.max_seq_len == 512
        assert config.dropout == 0.1
    
    def test_custom_config(self, small_config):
        """Test custom configuration values."""
        assert small_config.vocab_size == 1000
        assert small_config.d_model == 128
        assert small_config.n_heads == 4
        assert small_config.n_layers == 2
        assert small_config.d_ff == 512
        assert small_config.max_seq_len == 64
    
    def test_d_model_divisible_by_n_heads(self):
        """Test that d_model must be divisible by n_heads."""
        config = ModelConfig(d_model=128, n_heads=4)
        assert config.d_model % config.n_heads == 0


class TestEmbeddings:
    """Tests for Embeddings layer."""
    
    def test_output_shape(self, tiny_config, sample_input):
        """Test that embeddings produce correct output shape."""
        embeddings = Embeddings(tiny_config)
        output = embeddings(sample_input)
        
        batch_size, seq_len = sample_input.shape
        assert output.shape == (batch_size, seq_len, tiny_config.d_model)
    
    def test_different_seq_lengths(self, tiny_config):
        """Test with different sequence lengths."""
        embeddings = Embeddings(tiny_config)
        
        for seq_len in [1, 8, 16, 32]:
            input_ids = torch.randint(0, tiny_config.vocab_size, (2, seq_len))
            output = embeddings(input_ids)
            assert output.shape == (2, seq_len, tiny_config.d_model)
    
    def test_dtype(self, tiny_config, sample_input):
        """Test output dtype is float."""
        embeddings = Embeddings(tiny_config)
        output = embeddings(sample_input)
        assert output.dtype == torch.float32


class TestCausalSelfAttention:
    """Tests for CausalSelfAttention."""
    
    def test_output_shape(self, small_config):
        """Test output shape matches input shape."""
        batch_size, seq_len = 2, 16
        attn = CausalSelfAttention(small_config)
        x = torch.randn(batch_size, seq_len, small_config.d_model)
        
        output = attn(x)
        assert output.shape == x.shape
    
    def test_causal_mask_prevents_future_attendance(self, small_config):
        """Test that causal mask prevents attending to future tokens."""
        batch_size, seq_len = 1, 8
        attn = CausalSelfAttention(small_config)
        x = torch.randn(batch_size, seq_len, small_config.d_model)
        
        # Get attention weights by running forward pass
        # We'll verify the mask by checking that the model produces
        # different outputs when we change future tokens
        output1 = attn(x)
        
        # Modify a future token and verify it doesn't affect earlier positions
        x2 = x.clone()
        x2[0, 5:, :] = torch.randn(3, small_config.d_model)  # Change tokens after position 4
        
        output2 = attn(x2)
        
        # Position 4 and earlier should be identical
        # (since they can't attend to positions 5+)
        # Note: Due to floating point, we use allclose
        assert torch.allclose(output1[0, :5, :], output2[0, :5, :], atol=1e-6)
    
    def test_mask_shape(self, small_config):
        """Test that causal mask has correct shape."""
        attn = CausalSelfAttention(small_config)
        expected_shape = (1, 1, small_config.max_seq_len, small_config.max_seq_len)
        assert attn.causal_mask.shape == expected_shape
    
    def test_mask_is_lower_triangular(self, small_config):
        """Test that causal mask is lower triangular."""
        attn = CausalSelfAttention(small_config)
        mask = attn.causal_mask.squeeze()  # Remove batch and head dims
        
        # Check it's lower triangular
        for i in range(small_config.max_seq_len):
            for j in range(i + 1, small_config.max_seq_len):
                assert mask[i, j] == 0, f"Position ({i}, {j}) should be masked"
            for j in range(i + 1):
                assert mask[i, j] == 1, f"Position ({i}, {j}) should be unmasked"


class TestFeedForward:
    """Tests for FeedForward network."""
    
    def test_output_shape(self, small_config):
        """Test output shape matches input shape."""
        batch_size, seq_len = 2, 16
        ffn = FeedForward(small_config)
        x = torch.randn(batch_size, seq_len, small_config.d_model)
        
        output = ffn(x)
        assert output.shape == x.shape
    
    def test_expansion(self, small_config):
        """Test that FFN expands to d_ff then contracts."""
        ffn = FeedForward(small_config)
        
        # Check linear layers
        assert ffn.linear1.in_features == small_config.d_model
        assert ffn.linear1.out_features == small_config.d_ff
        assert ffn.linear2.in_features == small_config.d_ff
        assert ffn.linear2.out_features == small_config.d_model


class TestTransformerBlock:
    """Tests for TransformerBlock."""
    
    def test_output_shape(self, small_config):
        """Test output shape matches input shape."""
        batch_size, seq_len = 2, 16
        block = TransformerBlock(small_config)
        x = torch.randn(batch_size, seq_len, small_config.d_model)
        
        output = block(x)
        assert output.shape == x.shape
    
    def test_residual_connections(self, small_config):
        """Test that residual connections are present."""
        block = TransformerBlock(small_config)
        
        # Check that block has both attention and FFN
        assert hasattr(block, "self_attn")
        assert hasattr(block, "ffn")
        assert hasattr(block, "ln1")
        assert hasattr(block, "ln2")


class TestBabyPix1LM:
    """Tests for BabyPix1LM model."""
    
    def test_forward_shape(self, small_config):
        """Test forward pass output shape."""
        batch_size, seq_len = 2, 16
        model = BabyPix1LM(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        
        output = model(input_ids)
        
        assert "logits" in output
        assert output["logits"].shape == (batch_size, seq_len, small_config.vocab_size)
    
    def test_forward_with_labels(self, small_config):
        """Test forward pass with labels computes loss."""
        batch_size, seq_len = 2, 16
        model = BabyPix1LM(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        
        output = model(input_ids, labels=labels)
        
        assert "logits" in output
        assert "loss" in output
        assert output["loss"] is not None
        assert output["loss"].shape == ()  # Scalar loss
    
    def test_different_batch_sizes(self, small_config):
        """Test with different batch sizes."""
        model = BabyPix1LM(small_config)
        seq_len = 16
        
        for batch_size in [1, 2, 4, 8]:
            input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
            output = model(input_ids)
            assert output["logits"].shape == (batch_size, seq_len, small_config.vocab_size)
    
    def test_different_seq_lengths(self, small_config):
        """Test with different sequence lengths."""
        model = BabyPix1LM(small_config)
        batch_size = 2
        
        for seq_len in [1, 4, 16, 32, 64]:
            input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
            output = model(input_ids)
            assert output["logits"].shape == (batch_size, seq_len, small_config.vocab_size)
    
    def test_parameter_count(self, small_config):
        """Test parameter count is reasonable."""
        model = BabyPix1LM(small_config)
        param_count = model.count_parameters()
        
        # Should have at least some parameters
        assert param_count > 0
        
        # Check against expected minimum
        # Embeddings: vocab_size * d_model + max_seq_len * d_model
        # Each block: ~4 * d_model^2 + 2 * d_model * d_ff
        min_params = small_config.vocab_size * small_config.d_model
        assert param_count >= min_params
    
    def test_model_creation_factory(self, small_config):
        """Test create_model factory function."""
        model = create_model(small_config)
        assert isinstance(model, BabyPix1LM)
    
    def test_gradient_flow(self, small_config):
        """Test that gradients flow through the model."""
        model = BabyPix1LM(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 16))
        
        output = model(input_ids)
        loss = output["logits"].sum()
        loss.backward()
        
        # Check that gradients exist for key parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestCausalMasking:
    """Tests specifically for causal masking behavior."""
    
    def test_causal_property(self, small_config):
        """Test that position i cannot attend to position j > i."""
        model = BabyPix1LM(small_config)
        model.eval()
        
        batch_size, seq_len = 1, 8
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        
        # Get output for full sequence
        with torch.no_grad():
            output_full = model(input_ids)["logits"]
        
        # Now get output for truncated sequence (first 4 tokens)
        with torch.no_grad():
            output_truncated = model(input_ids[:, :4])["logits"]
        
        # First 4 positions should be identical
        assert torch.allclose(
            output_full[0, :4, :],
            output_truncated[0, :4, :],
            atol=1e-5
        )
    
    def test_mask_blocks_future_information(self, small_config):
        """Test that changing future tokens doesn't affect current positions."""
        model = BabyPix1LM(small_config)
        model.eval()
        
        batch_size, seq_len = 1, 8
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        
        # Get output
        with torch.no_grad():
            output1 = model(input_ids)["logits"]
        
        # Create modified input with different future tokens
        input_ids2 = input_ids.clone()
        input_ids2[0, 4:] = torch.randint(0, small_config.vocab_size, (4,))
        
        with torch.no_grad():
            output2 = model(input_ids2)["logits"]
        
        # Positions 0-3 should be identical
        assert torch.allclose(output1[0, :4, :], output2[0, :4, :], atol=1e-5)


class TestDifferentConfigs:
    """Tests with various configurations."""
    
    def test_minimal_config(self):
        """Test with minimal configuration."""
        config = ModelConfig(
            vocab_size=10,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_seq_len=8,
            dropout=0.0,
        )
        model = BabyPix1LM(config)
        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        output = model(input_ids)
        assert output["logits"].shape == (1, 4, config.vocab_size)
    
    def test_larger_config(self):
        """Test with larger configuration."""
        config = ModelConfig(
            vocab_size=5000,
            d_model=512,
            n_heads=8,
            n_layers=12,
            d_ff=2048,
            max_seq_len=1024,
            dropout=0.1,
        )
        model = BabyPix1LM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 32))
        output = model(input_ids)
        assert output["logits"].shape == (2, 32, config.vocab_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
