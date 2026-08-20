"""
Tests for BabyPix1 data preprocessing.
"""

import sys
import json
import struct
from pathlib import Path
from typing import List

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from babypix1.config import TokenizerConfig
from babypix1.tokenizer import BabyPixTokenizer
from babypix1.preprocess import (
    clean_text,
    read_text_files,
    compute_statistics,
    train_val_split,
    tokens_to_binary,
    binary_to_tokens,
    generate_sequences,
    chunked_tokenize,
    prepare_dataset,
    load_metadata,
)
from babypix1.data_utils import prepare_token_stream


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
The quick brown fox jumps over the lazy dog.

Machine learning is a subset of artificial intelligence.

Deep learning uses neural networks with many layers.

Natural language processing enables computers to understand language.

Transformers are a type of neural network architecture.

"""


@pytest.fixture
def sample_text_file(tmp_path, sample_text):
    """Create a sample text file for testing."""
    file_path = tmp_path / "sample.txt"
    file_path.write_text(sample_text)
    return str(file_path)


@pytest.fixture
def multi_file_dataset(tmp_path, sample_text):
    """Create multiple text files for testing."""
    files = []
    for i in range(3):
        file_path = tmp_path / "data" / f"file_{i}.txt"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(sample_text + f"Document {i}")
        files.append(str(file_path))
    return tmp_path / "data"


@pytest.fixture
def nested_file_dataset(tmp_path, sample_text):
    """Create nested folder structure for testing."""
    # Create nested folders
    for depth in range(3):
        folder = tmp_path / "nested"
        for d in range(depth + 1):
            folder = folder / f"level_{d}"
        folder.mkdir(parents=True, exist_ok=True)
        file_path = folder / f"file_{depth}.txt"
        file_path.write_text(sample_text)
    return tmp_path / "nested"


@pytest.fixture
def trained_tokenizer(sample_text_file, tmp_path):
    """Create and train a tokenizer for testing."""
    config = TokenizerConfig(vocab_size=500, min_frequency=1)
    tokenizer = BabyPixTokenizer(config)
    save_path = str(tmp_path / "tokenizer.json")
    tokenizer.train([sample_text_file], save_path=save_path)
    return tokenizer


class TestCleanText:
    """Tests for text cleaning."""
    
    def test_removes_empty_lines(self):
        """Test that empty lines are removed."""
        text = "Line 1\n\n\nLine 2\n   \nLine 3"
        cleaned = clean_text(text)
        lines = cleaned.split("\n")
        assert len(lines) == 3
        assert all(l.strip() for l in lines)
    
    def test_normalizes_whitespace(self):
        """Test that whitespace is normalized."""
        text = "Hello   world\t\twith  spaces"
        cleaned = clean_text(text)
        assert "  " not in cleaned  # No double spaces
        assert "\t" not in cleaned  # No tabs
    
    def test_removes_corrupted_chars(self):
        """Test that corrupted characters are removed."""
        text = "Hello\x00world\x01test\ufffd"
        cleaned = clean_text(text)
        assert "\x00" not in cleaned
        assert "\x01" not in cleaned
        assert "\ufffd" not in cleaned
    
    def test_preserves_punctuation(self):
        """Test that punctuation is preserved."""
        text = "Hello, world! How's it going?"
        cleaned = clean_text(text)
        assert "," in cleaned
        assert "!" in cleaned
        assert "'" in cleaned
        assert "?" in cleaned
    
    def test_preserves_capitalization(self):
        """Test that capitalization is preserved."""
        text = "Hello WORLD Test"
        cleaned = clean_text(text)
        assert "Hello" in cleaned
        assert "WORLD" in cleaned
        assert "Test" in cleaned


class TestReadTextFiles:
    """Tests for reading text files."""
    
    def test_read_single_file(self, sample_text_file):
        """Test reading a single file."""
        files = read_text_files(sample_text_file)
        assert len(files) == 1
        filepath, text = files[0]
        assert filepath == sample_text_file
        assert len(text) > 0
    
    def test_read_directory(self, multi_file_dataset):
        """Test reading multiple files from directory."""
        files = read_text_files(str(multi_file_dataset))
        assert len(files) == 3
    
    def test_read_nested_directories(self, nested_file_dataset):
        """Test reading files from nested directories."""
        files = read_text_files(str(nested_file_dataset))
        assert len(files) == 3  # 3 files in nested structure
    
    def test_read_nonexistent_path(self):
        """Test that nonexistent path raises error."""
        with pytest.raises(FileNotFoundError):
            read_text_files("nonexistent/path")
    
    def test_read_with_extensions(self, multi_file_dataset, tmp_path):
        """Test filtering by file extension."""
        # Create an additional text file
        (tmp_path / "data" / "extra.txt").write_text("extra content")
        
        files = read_text_files(str(multi_file_dataset), extensions=['.txt'])
        # Should find 4 files (3 original + 1 extra)
        assert len(files) == 4
        
        files_all = read_text_files(str(multi_file_dataset), extensions=['.txt'])
        assert len(files_all) == 4


class TestComputeStatistics:
    """Tests for dataset statistics."""
    
    def test_statistics_basic(self, trained_tokenizer):
        """Test basic statistics computation."""
        texts = [
            "Hello world\nTest sentence",
            "Another document\nWith multiple lines",
        ]
        
        stats = compute_statistics(texts, trained_tokenizer)
        
        assert stats["document_count"] == 2
        assert stats["line_count"] == 4
        assert stats["word_count"] > 0
        assert stats["char_count"] > 0
        assert stats["total_tokens"] > 0
        assert stats["avg_document_words"] > 0
        assert stats["avg_document_tokens"] > 0
    
    def test_statistics_single_text(self, trained_tokenizer):
        """Test statistics with single document."""
        texts = ["Single document here."]
        stats = compute_statistics(texts, trained_tokenizer)
        assert stats["document_count"] == 1
        assert stats["avg_document_words"] == stats["word_count"]
    
    def test_statistics_empty_texts(self, trained_tokenizer):
        """Test statistics with empty list."""
        stats = compute_statistics([], trained_tokenizer)
        assert stats["document_count"] == 0
        assert stats["total_tokens"] == 0
        assert stats["avg_document_words"] == 0.0


class TestTrainValSplit:
    """Tests for train/validation split."""
    
    def test_split_ratio(self):
        """Test that split maintains approximate ratio."""
        items = list(range(100))
        train, val = train_val_split(items, val_ratio=0.05, seed=42)
        
        assert len(train) == 95
        assert len(val) == 5
    
    def test_split_reproducibility(self):
        """Test that split is reproducible with same seed."""
        items = list(range(100))
        train1, val1 = train_val_split(items, val_ratio=0.05, seed=42)
        train2, val2 = train_val_split(items, val_ratio=0.05, seed=42)
        
        assert train1 == train2
        assert val1 == val2
    
    def test_split_no_overlap(self):
        """Test that train and val have no overlap."""
        items = list(range(100))
        train, val = train_val_split(items, val_ratio=0.1, seed=42)
        
        train_set = set(train)
        val_set = set(val)
        assert len(train_set & val_set) == 0
    
    def test_split_all_items_preserved(self):
        """Test that all items are in train or val."""
        items = list(range(100))
        train, val = train_val_split(items, val_ratio=0.1, seed=42)
        
        assert len(train) + len(val) == len(items)
    
    def test_split_small_dataset(self):
        """Test split with small dataset."""
        items = list(range(5))
        train, val = train_val_split(items, val_ratio=0.2, seed=42)
        
        assert len(val) >= 1
        assert len(train) + len(val) == 5


class TestBinaryFormat:
    """Tests for binary file format."""
    
    def test_tokens_to_binary(self, tmp_path):
        """Test writing tokens to binary file."""
        tokens = [1, 2, 3, 4, 5]
        output_path = str(tmp_path / "test.bin")
        
        tokens_to_binary(tokens, output_path)
        
        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size == 20  # 5 tokens * 4 bytes
    
    def test_binary_to_tokens(self, tmp_path):
        """Test reading tokens from binary file."""
        tokens = [10, 20, 30, 40, 50]
        output_path = str(tmp_path / "test.bin")
        
        tokens_to_binary(tokens, output_path)
        loaded = binary_to_tokens(output_path)
        
        assert loaded == tokens
    
    def test_binary_roundtrip(self, tmp_path):
        """Test binary format roundtrip."""
        tokens = list(range(1000))
        output_path = str(tmp_path / "test.bin")
        
        tokens_to_binary(tokens, output_path)
        loaded = binary_to_tokens(output_path)
        
        assert loaded == tokens


class TestGenerateSequences:
    """Tests for sequence generation."""
    
    def test_generate_sequences_alignment(self):
        """Test that generated sequences have correct target alignment."""
        tokens = list(range(20))
        max_seq_len = 8
        sequences = generate_sequences(tokens, max_seq_len)
        
        for input_ids, target_ids in sequences:
            # Target should be input shifted by 1
            for t in range(max_seq_len - 1):
                assert target_ids[t] == input_ids[t + 1]
    
    def test_generate_sequences_count(self):
        """Test correct number of sequences generated."""
        tokens = list(range(20))
        max_seq_len = 8
        sequences = generate_sequences(tokens, max_seq_len)
        
        # n - max_seq_len sequences
        assert len(sequences) == 20 - 8
    
    def test_generate_sequences_stride(self):
        """Test stride parameter."""
        tokens = list(range(20))
        max_seq_len = 4
        
        stride1 = generate_sequences(tokens, max_seq_len, stride=1)
        stride2 = generate_sequences(tokens, max_seq_len, stride=2)
        
        # Stride 2 should produce fewer sequences
        assert len(stride2) < len(stride1)
    
    def test_generate_sequences_empty(self):
        """Test with insufficient tokens."""
        tokens = [1, 2, 3]
        max_seq_len = 8
        sequences = generate_sequences(tokens, max_seq_len)
        
        assert len(sequences) == 0


class TestChunkedTokenize:
    """Tests for chunked tokenization."""
    
    def test_chunked_tokenize_basic(self, trained_tokenizer):
        """Test basic chunked tokenization."""
        texts = ["Hello world"] * 100
        tokens, count = chunked_tokenize(texts, trained_tokenizer, chunk_size=10)
        
        assert isinstance(tokens, list)
        assert count == len(tokens)
    
    def test_chunked_tokenize_count(self, trained_tokenizer):
        """Test token count is correct."""
        texts = ["Hello world test"] * 50
        tokens, count = chunked_tokenize(texts, trained_tokenizer)
        
        # Each text should produce same number of tokens
        single = trained_tokenizer.encode("Hello world test", add_special_tokens=False)
        expected = len(single) * 50
        
        assert count == expected
    
    def test_chunked_tokenize_memory_efficient(self, trained_tokenizer):
        """Test that chunking processes in smaller batches."""
        texts = ["Hello"] * 100
        
        # With small chunk size
        tokens_small, _ = chunked_tokenize(texts, trained_tokenizer, chunk_size=10)
        # With large chunk size
        tokens_large, _ = chunked_tokenize(texts, trained_tokenizer, chunk_size=1000)
        
        # Results should be the same
        assert tokens_small == tokens_large


class TestPrepareDataset:
    """Tests for full dataset preparation."""
    
    def test_prepare_dataset_creates_files(
        self, sample_text_file, trained_tokenizer, tmp_path
    ):
        """Test that prepare_dataset creates output files."""
        output_dir = str(tmp_path / "output")
        
        prepare_dataset(
            input_path=sample_text_file,
            tokenizer=trained_tokenizer,
            output_dir=output_dir,
            max_seq_len=32,
        )
        
        assert (Path(output_dir) / "train.bin").exists()
        assert (Path(output_dir) / "val.bin").exists()
        assert (Path(output_dir) / "metadata.json").exists()
    
    def test_prepare_dataset_metadata(
        self, sample_text_file, trained_tokenizer, tmp_path
    ):
        """Test metadata contains correct information."""
        output_dir = str(tmp_path / "output")
        
        metadata = prepare_dataset(
            input_path=sample_text_file,
            tokenizer=trained_tokenizer,
            output_dir=output_dir,
            max_seq_len=32,
        )
        
        assert "train_stats" in metadata
        assert "val_stats" in metadata
        assert "total_stats" in metadata
        # Total tokens should be > 0 (train might be 0 with small data and 5% split)
        assert metadata["total_stats"]["tokens"] > 0
    
    def test_prepare_dataset_split_ratio(
        self, sample_text_file, trained_tokenizer, tmp_path
    ):
        """Test that train/val split is correct."""
        output_dir = str(tmp_path / "output")
        
        metadata = prepare_dataset(
            input_path=sample_text_file,
            tokenizer=trained_tokenizer,
            output_dir=output_dir,
            max_seq_len=32,
            val_ratio=0.1,
        )
        
        total_tokens = metadata["total_stats"]["tokens"]
        train_tokens = metadata["train_stats"]["tokens"]
        val_tokens = metadata["val_stats"]["tokens"]
        
        assert train_tokens + val_tokens == total_tokens


class TestLoadMetadata:
    """Tests for loading metadata."""
    
    def test_load_metadata(
        self, sample_text_file, trained_tokenizer, tmp_path
    ):
        """Test loading saved metadata."""
        output_dir = str(tmp_path / "output")
        prepare_dataset(
            input_path=sample_text_file,
            tokenizer=trained_tokenizer,
            output_dir=output_dir,
        )
        
        metadata = load_metadata(output_dir)
        assert "train_stats" in metadata
        assert "val_stats" in metadata
    
    def test_load_metadata_nonexistent(self, tmp_path):
        """Test loading from nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            load_metadata(str(tmp_path / "nonexistent"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
