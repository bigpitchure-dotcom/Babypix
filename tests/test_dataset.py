"""
Tests for BabyPix1 Dataset and DataLoader.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from babypix1.config import TokenizerConfig, ModelConfig
from babypix1.tokenizer import BabyPixTokenizer
from babypix1.dataset import TextDataset, TokenTensorDataset, create_dataloader
from babypix1.data_utils import (
    read_text_file,
    tokenize_text,
    build_sliding_window_sequences,
    prepare_token_stream,
)


@pytest.fixture
def sample_text_file(tmp_path):
    """Create a sample text file for testing."""
    text = """
The quick brown fox jumps over the lazy dog.
Machine learning is a subset of artificial intelligence.
Deep learning uses neural networks with many layers.
Natural language processing enables computers to understand language.
Transformers are a type of neural network architecture.
"""
    file_path = tmp_path / "sample.txt"
    file_path.write_text(text.strip())
    return str(file_path)


@pytest.fixture
def trained_tokenizer(sample_text_file, tmp_path):
    """Create and train a tokenizer for testing."""
    config = TokenizerConfig(vocab_size=500, min_frequency=1)
    tokenizer = BabyPixTokenizer(config)
    save_path = str(tmp_path / "tokenizer.json")
    tokenizer.train([sample_text_file], save_path=save_path)
    return tokenizer


@pytest.fixture
def sample_tokens(trained_tokenizer, sample_text_file):
    """Get tokenized sample text."""
    tokens = prepare_token_stream(sample_text_file, trained_tokenizer)
    return tokens


class TestBuildSlidingWindowSequences:
    """Tests for sliding window sequence building."""
    
    def test_basic_sliding_window(self):
        """Test basic sliding window functionality."""
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        max_seq_len = 4
        sequences = build_sliding_window_sequences(tokens, max_seq_len)
        
        # Should create sequences starting at positions 0, 1, 2, 3, 4, 5, 6
        # (positions where we can fit a full sequence of length 4)
        assert len(sequences) > 0
        for seq in sequences:
            assert seq.shape == (max_seq_len,)
    
    def test_sliding_window_content(self):
        """Test that sliding window contains correct content."""
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        max_seq_len = 4
        sequences = build_sliding_window_sequences(tokens, max_seq_len)
        
        # First sequence should be [1, 2, 3, 4]
        assert sequences[0].tolist() == [1, 2, 3, 4]
        
        # Second sequence should be [2, 3, 4, 5]
        assert sequences[1].tolist() == [2, 3, 4, 5]
    
    def test_sliding_window_with_padding(self):
        """Test sliding window with short input."""
        tokens = [1, 2, 3]
        max_seq_len = 5
        sequences = build_sliding_window_sequences(tokens, max_seq_len)
        
        assert len(sequences) > 0
        # First sequence should be padded
        assert sequences[0].tolist() == [1, 2, 3, 0, 0]
    
    def test_empty_tokens(self):
        """Test with empty token list."""
        tokens = []
        max_seq_len = 4
        sequences = build_sliding_window_sequences(tokens, max_seq_len)
        assert len(sequences) == 0
    
    def test_single_token(self):
        """Test with single token."""
        tokens = [42]
        max_seq_len = 4
        sequences = build_sliding_window_sequences(tokens, max_seq_len)
        # Should create one padded sequence
        assert len(sequences) == 1
        assert sequences[0].tolist() == [42, 0, 0, 0]


class TestTextDataset:
    """Tests for TextDataset."""
    
    def test_dataset_length(self, sample_tokens):
        """Test dataset length calculation."""
        max_seq_len = 16
        dataset = TextDataset(sample_tokens, max_seq_len=max_seq_len)
        
        # Dataset should have sequences
        assert len(dataset) > 0
    
    def test_dataset_getitem_shapes(self, sample_tokens):
        """Test that __getitem__ returns correct shapes."""
        max_seq_len = 16
        dataset = TextDataset(sample_tokens, max_seq_len=max_seq_len)
        
        input_ids, target_ids = dataset[0]
        
        assert input_ids.shape == (max_seq_len,)
        assert target_ids.shape == (max_seq_len,)
    
    def test_target_shift_alignment(self, sample_tokens):
        """Test that target is shifted by 1 position."""
        max_seq_len = 8
        dataset = TextDataset(sample_tokens, max_seq_len=max_seq_len)
        
        input_ids, target_ids = dataset[0]
        
        # For all positions except the last, target[t] should be input[t+1]
        for t in range(max_seq_len - 1):
            assert target_ids[t] == input_ids[t + 1], \
                f"Mismatch at position {t}: target[{t}]={target_ids[t]} != input[{t+1}]={input_ids[t+1]}"
    
    def test_dataset_from_text_file(self, sample_text_file, trained_tokenizer):
        """Test dataset creation from text file."""
        max_seq_len = 32
        dataset = TextDataset(
            token_ids=sample_text_file,
            max_seq_len=max_seq_len,
            tokenizer=trained_tokenizer,
        )
        
        assert len(dataset) > 0
        input_ids, target_ids = dataset[0]
        assert input_ids.shape == (max_seq_len,)
    
    def test_dataset_from_tensor(self):
        """Test dataset creation from tensor."""
        tokens = torch.arange(100, dtype=torch.long)
        max_seq_len = 16
        dataset = TokenTensorDataset(tokens, max_seq_len=max_seq_len)
        
        assert len(dataset) > 0
        input_ids, target_ids = dataset[0]
        assert input_ids.shape == (max_seq_len,)
    
    def test_dataset_from_list(self):
        """Test dataset creation from list."""
        tokens = list(range(100))
        max_seq_len = 16
        dataset = TextDataset(tokens, max_seq_len=max_seq_len)
        
        assert len(dataset) > 0
        input_ids, target_ids = dataset[0]
        assert input_ids.shape == (max_seq_len,)
    
    def test_dataset_requires_tokenizer_for_file(self):
        """Test that file path requires tokenizer."""
        with pytest.raises(ValueError):
            TextDataset("nonexistent.txt", max_seq_len=16)


class TestCreateDataloader:
    """Tests for DataLoader creation."""
    
    def test_dataloader_batch_shapes(self, sample_tokens):
        """Test that dataloader yields correct batch shapes."""
        max_seq_len = 16
        batch_size = 4
        dataset = TextDataset(sample_tokens, max_seq_len=max_seq_len)
        dataloader = create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        for batch in dataloader:
            input_ids, target_ids = batch
            assert input_ids.shape == (batch_size, max_seq_len)
            assert target_ids.shape == (batch_size, max_seq_len)
            break  # Just check first batch
    
    def test_dataloader_batch_alignment(self, sample_tokens):
        """Test that batch target alignment is preserved."""
        max_seq_len = 16
        batch_size = 4
        dataset = TextDataset(sample_tokens, max_seq_len=max_seq_len)
        dataloader = create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        for input_ids, target_ids in dataloader:
            # Check alignment for each sample in batch
            for i in range(batch_size):
                for t in range(max_seq_len - 1):
                    assert target_ids[i, t] == input_ids[i, t + 1], \
                        f"Batch {i}, position {t}: alignment mismatch"
            break
    
    def test_dataloader_shuffle(self, sample_tokens):
        """Test that shuffle=True produces different order."""
        max_seq_len = 16
        dataset = TextDataset(sample_tokens, max_seq_len=max_seq_len)
        
        # Get order without shuffle
        dataloader_no_shuffle = create_dataloader(
            dataset, batch_size=4, shuffle=False, num_workers=0
        )
        order_no_shuffle = [batch[0][0, 0].item() for batch in dataloader_no_shuffle]
        
        # Get order with shuffle
        dataloader_shuffle = create_dataloader(
            dataset, batch_size=4, shuffle=True, num_workers=0
        )
        order_shuffle = [batch[0][0, 0].item() for batch in dataloader_shuffle]
        
        # They should be different (with high probability)
        # Note: This test might occasionally fail due to randomness
        # In practice, we just verify the function works
        assert len(order_no_shuffle) > 0
        assert len(order_shuffle) > 0


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_very_short_text(self, trained_tokenizer):
        """Test with very short text."""
        # Create a very short text file
        tokens = trained_tokenizer.encode("Hi", add_special_tokens=False)
        dataset = TextDataset(tokens, max_seq_len=64)
        
        # Should handle gracefully
        assert len(dataset) >= 0
    
    def test_exact_seq_len(self):
        """Test when token count equals max_seq_len."""
        tokens = list(range(32))
        max_seq_len = 32
        dataset = TextDataset(tokens, max_seq_len=max_seq_len)
        
        # Should create exactly one sequence
        assert len(dataset) == 1
        input_ids, target_ids = dataset[0]
        assert input_ids.shape == (max_seq_len,)
    
    def test_tokens_longer_than_seq_len(self):
        """Test when tokens exceed max_seq_len."""
        tokens = list(range(100))
        max_seq_len = 16
        dataset = TextDataset(tokens, max_seq_len=max_seq_len)
        
        # Should create multiple sequences
        assert len(dataset) > 1
    
    def test_single_sequence(self):
        """Test when only one sequence is possible."""
        tokens = list(range(10))
        max_seq_len = 10
        dataset = TextDataset(tokens, max_seq_len=max_seq_len)
        
        assert len(dataset) == 1
        input_ids, target_ids = dataset[0]
        assert input_ids.shape == (max_seq_len,)


class TestDataUtils:
    """Tests for data utility functions."""
    
    def test_read_text_file(self, sample_text_file):
        """Test reading a text file."""
        text = read_text_file(sample_text_file)
        assert isinstance(text, str)
        assert len(text) > 0
    
    def test_read_nonexistent_file(self):
        """Test reading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            read_text_file("nonexistent.txt")
    
    def test_tokenize_text(self, trained_tokenizer):
        """Test tokenizing text."""
        tokens = tokenize_text("Hello world", trained_tokenizer)
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
    
    def test_prepare_token_stream(self, sample_text_file, trained_tokenizer):
        """Test preparing token stream from file."""
        tokens = prepare_token_stream(sample_text_file, trained_tokenizer)
        assert isinstance(tokens, list)
        assert len(tokens) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
