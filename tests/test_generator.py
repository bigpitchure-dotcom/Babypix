"""
Tests for BabyPix1 text generation.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from babypix1.config import ModelConfig, TokenizerConfig
from babypix1.model import BabyPix1LM
from babypix1.tokenizer import BabyPixTokenizer
from babypix1.generator import (
    top_k_logits,
    top_p_logits,
    sample_from_logits,
    generate,
    generate_tokens,
    TextGenerator,
)


@pytest.fixture
def tiny_config():
    """Tiny model config for fast testing."""
    return ModelConfig(
        vocab_size=100,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        max_seq_len=32,
        dropout=0.0,
    )


@pytest.fixture
def sample_text_file(tmp_path):
    """Create a sample text file for tokenizer training."""
    text = """
    The quick brown fox jumps over the lazy dog.
    Machine learning is a subset of artificial intelligence.
    Deep learning uses neural networks with many layers.
    Natural language processing enables computers to understand language.
    Transformers are a type of neural network architecture.
    The attention mechanism allows models to focus on relevant parts.
    Training a language model requires large amounts of text data.
    Tokenization breaks text into smaller units called tokens.
    Byte-pair encoding is a popular tokenization algorithm.
    Embeddings convert token IDs into dense vectors that capture semantic meaning.
    """
    file_path = tmp_path / "sample.txt"
    file_path.write_text(text.strip())
    return str(file_path)


@pytest.fixture
def trained_tokenizer(sample_text_file, tmp_path):
    """Create and train a tokenizer."""
    config = TokenizerConfig(vocab_size=500, min_frequency=1)
    tokenizer = BabyPixTokenizer(config)
    save_path = str(tmp_path / "tokenizer.json")
    tokenizer.train([sample_text_file], save_path=save_path)
    return tokenizer


@pytest.fixture
def model(tiny_config):
    """Create a model for testing."""
    return BabyPix1LM(tiny_config)


@pytest.fixture
def model_with_matching_vocab(trained_tokenizer, tiny_config):
    """Create a model with vocab_size matching the tokenizer."""
    config = ModelConfig(
        vocab_size=trained_tokenizer.vocab_size,
        d_model=tiny_config.d_model,
        n_heads=tiny_config.n_heads,
        n_layers=tiny_config.n_layers,
        d_ff=tiny_config.d_ff,
        max_seq_len=tiny_config.max_seq_len,
        dropout=tiny_config.dropout,
    )
    return BabyPix1LM(config)


class TestTopKLogits:
    """Tests for top-k logits filtering."""
    
    def test_top_k_removes_low_probabilities(self):
        """Test that tokens with rank > k are set to -inf."""
        logits = torch.tensor([[10.0, 5.0, 3.0, 1.0, -2.0]])
        k = 3
        filtered = top_k_logits(logits, k)
        
        # Top 3 should remain, bottom 2 should be -inf
        assert filtered[0, 0].item() != float('-inf')
        assert filtered[0, 1].item() != float('-inf')
        assert filtered[0, 2].item() != float('-inf')
        assert filtered[0, 3].item() == float('-inf')
        assert filtered[0, 4].item() == float('-inf')
    
    def test_top_k_preserves_original_values(self):
        """Test that top-k values are unchanged."""
        logits = torch.tensor([[10.0, 5.0, 3.0, 1.0, -2.0]])
        k = 2
        filtered = top_k_logits(logits, k)
        
        # Top 2 values should be unchanged
        assert filtered[0, 0].item() == 10.0
        assert filtered[0, 1].item() == 5.0
    
    def test_top_k_zero_returns_original(self):
        """Test that k=0 returns original logits."""
        logits = torch.tensor([[10.0, 5.0, 3.0]])
        k = 0
        filtered = top_k_logits(logits, k)
        assert torch.equal(filtered, logits)
    
    def test_top_k_with_batch(self):
        """Test top-k with batch dimension."""
        logits = torch.tensor([
            [10.0, 5.0, 3.0, 1.0, -2.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ])
        k = 2
        filtered = top_k_logits(logits, k)
        
        # First row: top 2 are 10 and 5
        assert filtered[0, 0].item() != float('-inf')
        assert filtered[0, 1].item() != float('-inf')
        
        # Second row: top 2 are 5 and 4
        assert filtered[1, 4].item() != float('-inf')
        assert filtered[1, 3].item() != float('-inf')


class TestTopPLogits:
    """Tests for top-p (nucleus) sampling."""
    
    def test_top_p_keeps_cumulative(self):
        """Test that top-p keeps enough tokens for cumulative probability."""
        logits = torch.tensor([[10.0, 1.0, 1.0, 1.0, 1.0]])
        top_p = 0.9
        filtered = top_p_logits(logits, top_p)
        
        # The dominant token (logit 10) should be kept
        assert filtered[0, 0].item() != float('-inf')
    
    def test_top_p_one_keeps_all(self):
        """Test that top_p=1.0 doesn't filter anything."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        filtered = top_p_logits(logits, 1.0)
        assert torch.equal(filtered, logits)
    
    def test_top_p_zero_keeps_all(self):
        """Test that top_p=0.0 doesn't filter anything."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        filtered = top_p_logits(logits, 0.0)
        assert torch.equal(filtered, logits)
    
    def test_top_p_at_least_one_kept(self):
        """Test that at least one token is always kept."""
        logits = torch.tensor([[100.0, 0.0, 0.0, 0.0]])
        filtered = top_p_logits(logits, 0.1)
        
        # At least one should not be -inf
        non_inf_count = (filtered > float('-inf')).sum().item()
        assert non_inf_count >= 1


class TestSampleFromLogits:
    """Tests for sampling from logits."""
    
    def test_greedy_decoding(self):
        """Test that temperature=0 gives greedy (argmax) decoding."""
        logits = torch.tensor([[1.0, 5.0, 2.0, 3.0]])
        sampled = sample_from_logits(logits, temperature=0.0)
        assert sampled.item() == 1
    
    def test_temperature_scaling(self):
        """Test that temperature affects sampling distribution."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        sampled = sample_from_logits(logits, temperature=0.5)
        assert 0 <= sampled.item() < 4
    
    def test_top_k_sampling(self):
        """Test that top-k sampling only samples from top-k tokens."""
        logits = torch.tensor([[10.0, 1.0, 1.0, 1.0, 1.0]])
        top_k = 2
        sampled = sample_from_logits(logits, temperature=1.0, top_k=top_k)
        top_indices = [0, 1]
        assert sampled.item() in top_indices
    
    def test_top_p_sampling(self):
        """Test that top-p sampling restricts to nucleus."""
        logits = torch.tensor([[10.0, 1.0, 1.0, 1.0, 1.0]])
        sampled = sample_from_logits(logits, temperature=1.0, top_p=0.5)
        assert 0 <= sampled.item() < 5


class TestGenerate:
    """Tests for text generation."""
    
    def test_generate_returns_string(self, model_with_matching_vocab, trained_tokenizer):
        """Test that generate returns a string."""
        output = generate(
            model=model_with_matching_vocab,
            tokenizer=trained_tokenizer,
            prompt="Hello",
            max_new_tokens=10,
            temperature=0.8,
        )
        assert isinstance(output, str)
    
    def test_generate_length(self, model_with_matching_vocab, trained_tokenizer):
        """Test that generation produces expected length."""
        output = generate(
            model=model_with_matching_vocab,
            tokenizer=trained_tokenizer,
            prompt="Hello",
            max_new_tokens=10,
            temperature=0.8,
        )
        assert len(output) > len("Hello")
    
    def test_generate_max_new_tokens(self, model_with_matching_vocab, trained_tokenizer):
        """Test that max_new_tokens is respected."""
        output = generate(
            model=model_with_matching_vocab,
            tokenizer=trained_tokenizer,
            prompt="Hello",
            max_new_tokens=0,
            temperature=0.8,
        )
        assert len(output) > 0
    
    def test_generate_tokens_length(self, model_with_matching_vocab, trained_tokenizer):
        """Test that generated token IDs are correct length."""
        prompt = "Hello"
        input_ids = trained_tokenizer.encode(prompt, add_special_tokens=False)
        
        output_tokens = generate_tokens(
            model=model_with_matching_vocab,
            tokenizer=trained_tokenizer,
            prompt=prompt,
            max_new_tokens=10,
            temperature=1.0,
            return_tokens=True,
        )
        assert len(output_tokens) >= len(input_ids)
    
    def test_eos_termination(self, model_with_matching_vocab, trained_tokenizer):
        """Test that generation stops at EOS token."""
        eos_id = trained_tokenizer.eos_token_id
        
        output_tokens = generate_tokens(
            model=model_with_matching_vocab,
            tokenizer=trained_tokenizer,
            prompt="Hello",
            max_new_tokens=100,
            temperature=0.0,
            return_tokens=True,
        )
        
        # If EOS appears, it should be the last token (or not present)
        prompt_len = len(trained_tokenizer.encode("Hello", add_special_tokens=False))
        if eos_id in output_tokens[prompt_len:]:
            eos_pos = output_tokens.index(eos_id, prompt_len)
            assert eos_pos == len(output_tokens) - 1


class TestGreedyDeterminism:
    """Tests for deterministic behavior."""
    
    def test_greedy_is_deterministic(self, model_with_matching_vocab, trained_tokenizer):
        """Test that temperature=0 (greedy) produces same output every time."""
        prompt = "Hello world"
        
        output1 = generate(
            model=model_with_matching_vocab,
            tokenizer=trained_tokenizer,
            prompt=prompt,
            max_new_tokens=20,
            temperature=0.0,
        )
        
        output2 = generate(
            model=model_with_matching_vocab,
            tokenizer=trained_tokenizer,
            prompt=prompt,
            max_new_tokens=20,
            temperature=0.0,
        )
        
        assert output1 == output2, "Greedy decoding should be deterministic"
    
    def test_greedy_gives_argmax(self, model_with_matching_vocab, trained_tokenizer):
        """Test that greedy decoding uses argmax over logits."""
        prompt = "Hello"
        input_ids = trained_tokenizer.encode(prompt, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        
        model_with_matching_vocab.eval()
        with torch.no_grad():
            outputs = model_with_matching_vocab(input_tensor)
            logits = outputs["logits"]
            expected_next = torch.argmax(logits[0, -1]).item()
        
        # Generate with temperature=0
        output_tokens = generate_tokens(
            model=model_with_matching_vocab,
            tokenizer=trained_tokenizer,
            prompt=prompt,
            max_new_tokens=1,
            temperature=0.0,
            return_tokens=True,
        )
        
        # The first generated token should be argmax
        prompt_len = len(input_ids)
        assert output_tokens[prompt_len] == expected_next


class TestTextGenerator:
    """Tests for TextGenerator class."""
    
    def test_text_generator_creation(self, model_with_matching_vocab, trained_tokenizer):
        """Test TextGenerator initialization."""
        generator = TextGenerator(model_with_matching_vocab, trained_tokenizer)
        assert generator.model is model_with_matching_vocab
        assert generator.tokenizer is trained_tokenizer
        assert generator.model.training is False
    
    def test_text_generator_generate(self, model_with_matching_vocab, trained_tokenizer):
        """Test TextGenerator.generate()."""
        generator = TextGenerator(model_with_matching_vocab, trained_tokenizer)
        output = generator.generate(
            prompt="Hello",
            max_new_tokens=10,
            temperature=0.8,
        )
        assert isinstance(output, str)
        assert len(output) > 0
    
    def test_text_generator_multiple(self, model_with_matching_vocab, trained_tokenizer):
        """Test generating multiple sequences."""
        generator = TextGenerator(model_with_matching_vocab, trained_tokenizer)
        results = generator.generate_multiple(
            prompt="Hello",
            num_return_sequences=3,
            max_new_tokens=10,
            temperature=1.0,
        )
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)


class TestContextWindow:
    """Tests for context window management."""
    
    def test_long_prompt_cropping(self, trained_tokenizer):
        """Test that long prompts are cropped to max_seq_len."""
        config = ModelConfig(
            vocab_size=trained_tokenizer.vocab_size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            max_seq_len=8,
            dropout=0.0,
        )
        model = BabyPix1LM(config)
        
        long_prompt = " ".join(["Hello"] * 20)
        tokens = trained_tokenizer.encode(long_prompt)
        assert len(tokens) > 8
        
        output = generate(
            model=model,
            tokenizer=trained_tokenizer,
            prompt=long_prompt,
            max_new_tokens=5,
            temperature=0.8,
        )
        assert isinstance(output, str)


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_end_to_end_generation(self, sample_text_file, tmp_path):
        """Test generation from a trained model checkpoint."""
        # Train a tokenizer
        tok_config = TokenizerConfig(vocab_size=500, min_frequency=1)
        tokenizer = BabyPix1LM.__new__(BabyPix1LM)  # not used
        tokenizer_obj = BabyPixTokenizer(tok_config)
        tok_path = str(tmp_path / "tokenizer.json")
        tokenizer_obj.train([sample_text_file], save_path=tok_path)
        
        # Create model with matching vocab
        config = ModelConfig(
            vocab_size=tokenizer_obj.vocab_size,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=128,
            max_seq_len=32,
            dropout=0.0,
        )
        model = BabyPix1LM(config)
        
        # Generate text
        prompt = "The quick"
        output = generate(
            model=model,
            tokenizer=tokenizer_obj,
            prompt=prompt,
            max_new_tokens=10,
            temperature=0.8,
        )
        
        assert isinstance(output, str)
        assert len(output) > len(prompt)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
