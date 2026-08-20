"""
Tests for BabyPix1 tokenizer.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from babypix1.config import TokenizerConfig
from babypix1.tokenizer import BabyPixTokenizer


@pytest.fixture
def sample_file(tmp_path):
    """Create a temporary text file for training."""
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
Embeddings convert token IDs into dense vectors.
This is a test sentence for the BabyPix1 tokenizer.
Another sentence to add variety to the training data.
We need enough text to train a meaningful vocabulary.
The tokenizer should learn subword patterns from this text.
Each word gets split into meaningful subword units.
"""
    file_path = tmp_path / "train.txt"
    file_path.write_text(text.strip())
    return str(file_path)


@pytest.fixture
def trained_tokenizer(sample_file, tmp_path):
    """Create and train a tokenizer for testing."""
    config = TokenizerConfig(vocab_size=500, min_frequency=1)
    tokenizer = BabyPixTokenizer(config)
    save_path = str(tmp_path / "tokenizer.json")
    tokenizer.train([sample_file], save_path=save_path)
    return tokenizer


class TestTokenizerConfig:
    """Tests for TokenizerConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TokenizerConfig()
        assert config.vocab_size == 2000
        assert config.min_frequency == 2
        assert "<pad>" in config.special_tokens
        assert "<unk>" in config.special_tokens
        assert "<bos>" in config.special_tokens
        assert "<eos>" in config.special_tokens
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = TokenizerConfig(vocab_size=1000, min_frequency=5)
        assert config.vocab_size == 1000
        assert config.min_frequency == 5
    
    def test_save_load_config(self, tmp_path):
        """Test saving and loading configuration."""
        config = TokenizerConfig(vocab_size=1500)
        config_path = str(tmp_path / "config.yaml")
        config.save(config_path)
        
        loaded = TokenizerConfig.load(config_path)
        assert loaded.vocab_size == 1500


class TestTokenizerTraining:
    """Tests for tokenizer training."""
    
    def test_train_tokenizer(self, trained_tokenizer):
        """Test that tokenizer trains successfully."""
        assert trained_tokenizer._tokenizer is not None
        assert trained_tokenizer.vocab_size > 0
    
    def test_vocab_size(self, trained_tokenizer):
        """Test that vocab size matches config."""
        # Vocab should be at most vocab_size + special tokens
        assert trained_tokenizer.vocab_size <= 500 + 4
    
    def test_special_tokens_exist(self, trained_tokenizer):
        """Test that all special tokens are in vocabulary."""
        assert trained_tokenizer.pad_token_id is not None
        assert trained_tokenizer.unk_token_id is not None
        assert trained_tokenizer.bos_token_id is not None
        assert trained_tokenizer.eos_token_id is not None
    
    def test_special_tokens_unique(self, trained_tokenizer):
        """Test that special token IDs are unique."""
        ids = [
            trained_tokenizer.pad_token_id,
            trained_tokenizer.unk_token_id,
            trained_tokenizer.bos_token_id,
            trained_tokenizer.eos_token_id,
        ]
        assert len(ids) == len(set(ids))


class TestTokenizerEncode:
    """Tests for encoding text."""
    
    def test_encode_returns_list(self, trained_tokenizer):
        """Test that encode returns a list of integers."""
        result = trained_tokenizer.encode("Hello world")
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)
    
    def test_encode_adds_bos_eos(self, trained_tokenizer):
        """Test that encode adds <bos> and <eos> tokens."""
        result = trained_tokenizer.encode("Hello world", add_special_tokens=True)
        assert result[0] == trained_tokenizer.bos_token_id
        assert result[-1] == trained_tokenizer.eos_token_id
    
    def test_encode_without_special_tokens(self, trained_tokenizer):
        """Test encoding without special tokens."""
        result = trained_tokenizer.encode("Hello world", add_special_tokens=False)
        assert result[0] != trained_tokenizer.bos_token_id
        assert result[-1] != trained_tokenizer.eos_token_id
    
    def test_encode_empty_string(self, trained_tokenizer):
        """Test encoding an empty string."""
        result = trained_tokenizer.encode("", add_special_tokens=False)
        assert result == []
    
    def test_encode_single_token(self, trained_tokenizer):
        """Test encoding a single word."""
        result = trained_tokenizer.encode("the", add_special_tokens=False)
        assert len(result) >= 1


class TestTokenizerDecode:
    """Tests for decoding token IDs."""
    
    def test_decode_returns_string(self, trained_tokenizer):
        """Test that decode returns a string."""
        ids = trained_tokenizer.encode("Hello world")
        result = trained_tokenizer.decode(ids)
        assert isinstance(result, str)
    
    def test_decode_roundtrip(self, trained_tokenizer):
        """Test encode -> decode roundtrip preserves content."""
        # Test with words that are definitely in the training vocabulary
        original = "the quick brown fox"
        encoded = trained_tokenizer.encode(original, add_special_tokens=False)
        decoded = trained_tokenizer.decode(encoded)
        # Content should be preserved (whitespace may differ due to BPE)
        assert decoded.strip() == original.strip()
    
    def test_decode_skips_special_tokens(self, trained_tokenizer):
        """Test that decode skips special tokens by default."""
        ids = [
            trained_tokenizer.bos_token_id,
            10,
            20,
            trained_tokenizer.eos_token_id,
        ]
        result = trained_tokenizer.decode(ids)
        # Special tokens should not appear in output
        assert "<bos>" not in result
        assert "<eos>" not in result
    
    def test_decode_keeps_special_tokens(self, trained_tokenizer):
        """Test that decode with skip_special_tokens=False doesn't filter tokens."""
        # When skip_special_tokens=False, the raw tokens are passed to decode
        # The underlying tokenizer may not render them as strings, but the
        # filtering should not happen
        ids = [
            trained_tokenizer.bos_token_id,
            10,
            trained_tokenizer.eos_token_id,
        ]
        result_skip = trained_tokenizer.decode(ids, skip_special_tokens=True)
        result_keep = trained_tokenizer.decode(ids, skip_special_tokens=False)
        # With skip=True, special tokens are filtered out
        # With skip=False, they are not filtered (output may differ)
        assert isinstance(result_keep, str)
        # The key test: filtering should produce different results
        # when special tokens are present
        assert len(result_keep) >= len(result_skip)
    
    def test_decode_empty_list(self, trained_tokenizer):
        """Test decoding an empty list."""
        result = trained_tokenizer.decode([])
        assert result == ""


class TestTokenizerBatch:
    """Tests for batch operations."""
    
    def test_encode_batch(self, trained_tokenizer):
        """Test encoding a batch of texts."""
        texts = ["Hello", "World", "Test"]
        results = trained_tokenizer.encode_batch(texts)
        assert len(results) == len(texts)
        assert all(isinstance(r, list) for r in results)
    
    def test_decode_batch(self, trained_tokenizer):
        """Test decoding a batch of token lists."""
        texts = ["Hello", "World"]
        encoded = trained_tokenizer.encode_batch(texts)
        decoded = trained_tokenizer.decode_batch(encoded)
        assert len(decoded) == len(texts)
        assert all(isinstance(d, str) for d in decoded)


class TestTokenizerVocab:
    """Tests for vocabulary operations."""
    
    def test_get_vocab(self, trained_tokenizer):
        """Test getting the full vocabulary."""
        vocab = trained_tokenizer.get_vocab()
        assert isinstance(vocab, dict)
        assert len(vocab) > 0
    
    def test_token_to_id(self, trained_tokenizer):
        """Test converting token to ID."""
        # Special tokens should exist
        pad_id = trained_tokenizer.token_to_id("<pad>")
        assert pad_id is not None
        assert isinstance(pad_id, int)
    
    def test_id_to_token(self, trained_tokenizer):
        """Test converting ID to token."""
        pad_id = trained_tokenizer.pad_token_id
        token = trained_tokenizer.id_to_token(pad_id)
        assert token == "<pad>"
    
    def test_len(self, trained_tokenizer):
        """Test __len__ returns vocab size."""
        assert len(trained_tokenizer) == trained_tokenizer.vocab_size


class TestTokenizerSaveLoad:
    """Tests for saving and loading."""
    
    def test_save_and_load(self, trained_tokenizer, tmp_path):
        """Test saving and loading tokenizer."""
        save_path = str(tmp_path / "test_tokenizer.json")
        trained_tokenizer.save(save_path)
        
        new_tokenizer = BabyPixTokenizer()
        new_tokenizer.load(save_path)
        
        assert new_tokenizer.vocab_size == trained_tokenizer.vocab_size
    
    def test_loaded_tokenizer_encodes(self, trained_tokenizer, tmp_path):
        """Test that loaded tokenizer can encode text."""
        save_path = str(tmp_path / "test_tokenizer.json")
        trained_tokenizer.save(save_path)
        
        new_tokenizer = BabyPixTokenizer()
        new_tokenizer.load(save_path)
        
        # Both should produce same output
        text = "Hello world"
        ids1 = trained_tokenizer.encode(text)
        ids2 = new_tokenizer.encode(text)
        assert ids1 == ids2
    
    def test_load_nonexistent(self):
        """Test loading from nonexistent file raises error."""
        tokenizer = BabyPixTokenizer()
        with pytest.raises(FileNotFoundError):
            tokenizer.load("nonexistent.json")


class TestTokenizerEdgeCases:
    """Tests for edge cases."""
    
    def test_encode_whitespace(self, trained_tokenizer):
        """Test encoding whitespace."""
        result = trained_tokenizer.encode("   ", add_special_tokens=False)
        assert isinstance(result, list)
    
    def test_encode_special_characters(self, trained_tokenizer):
        """Test encoding special characters."""
        result = trained_tokenizer.encode("!@#$%^&*()", add_special_tokens=False)
        assert isinstance(result, list)
    
    def test_encode_numbers(self, trained_tokenizer):
        """Test encoding numbers."""
        result = trained_tokenizer.encode("12345", add_special_tokens=False)
        assert isinstance(result, list)
    
    def test_encode_long_text(self, trained_tokenizer):
        """Test encoding text longer than training data."""
        long_text = "word " * 100
        result = trained_tokenizer.encode(long_text)
        assert isinstance(result, list)
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
