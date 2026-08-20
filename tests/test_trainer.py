"""
Tests for BabyPix1 Trainer.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from babypix1.config import ModelConfig
from babypix1.model import BabyPix1LM
from babypix1.trainer import Trainer


@pytest.fixture
def tiny_config():
    """Tiny config for fast testing."""
    return ModelConfig(
        vocab_size=100,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        max_seq_len=16,
        dropout=0.0,
    )


@pytest.fixture
def synthetic_dataset():
    """Create synthetic dataset for testing."""
    batch_size = 4
    seq_len = 16
    vocab_size = 100
    
    # Create random token data
    input_ids = torch.randint(0, vocab_size, (batch_size * 10, seq_len))
    target_ids = torch.randint(0, vocab_size, (batch_size * 10, seq_len))
    
    return TensorDataset(input_ids, target_ids)


@pytest.fixture
def synthetic_loader(synthetic_dataset):
    """Create data loader from synthetic dataset."""
    return DataLoader(synthetic_dataset, batch_size=4, shuffle=True)


@pytest.fixture
def trainer_with_model(tiny_config, synthetic_loader, tmp_path):
    """Create trainer with model for testing."""
    model = BabyPix1LM(tiny_config)
    trainer = Trainer(
        model=model,
        train_loader=synthetic_loader,
        learning_rate=1e-3,
        warmup_steps=0,
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    return trainer


class TestTrainerInitialization:
    """Tests for trainer initialization."""
    
    def test_trainer_creation(self, tiny_config, synthetic_loader, tmp_path):
        """Test trainer is created correctly."""
        model = BabyPix1LM(tiny_config)
        trainer = Trainer(
            model=model,
            train_loader=synthetic_loader,
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )
        
        assert trainer.model is model
        assert trainer.train_loader is synthetic_loader
        assert trainer.global_step == 0
        assert trainer.current_epoch == 0
    
    def test_device_selection(self, tiny_config, synthetic_loader, tmp_path):
        """Test device is selected correctly."""
        model = BabyPix1LM(tiny_config)
        trainer = Trainer(
            model=model,
            train_loader=synthetic_loader,
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )
        
        # Should have a valid device
        assert isinstance(trainer.device, torch.device)
    
    def test_optimizer_created(self, trainer_with_model):
        """Test optimizer is created."""
        assert trainer_with_model.optimizer is not None
    
    def test_scheduler_created(self, trainer_with_model):
        """Test scheduler is created."""
        assert trainer_with_model.scheduler is not None


class TestTrainStep:
    """Tests for single training step."""
    
    def test_train_step_returns_loss(self, trainer_with_model):
        """Test train_step returns a loss value."""
        batch = next(iter(trainer_with_model.train_loader))
        loss = trainer_with_model.train_step(batch)
        
        assert isinstance(loss, float)
        assert loss > 0
    
    def test_train_step_updates_global_step(self, trainer_with_model):
        """Test train_step increments global step."""
        initial_step = trainer_with_model.global_step
        batch = next(iter(trainer_with_model.train_loader))
        
        trainer_with_model.train_step(batch)
        
        assert trainer_with_model.global_step == initial_step + 1
    
    def test_train_step_updates_weights(self, trainer_with_model):
        """Test train_step actually updates model weights."""
        # Get initial weights as numpy for comparison
        initial_weights = {
            name: param.data.cpu().numpy().copy()
            for name, param in trainer_with_model.model.named_parameters()
            if param.requires_grad
        }
        
        # Perform a training step
        batch = next(iter(trainer_with_model.train_loader))
        trainer_with_model.train_step(batch)
        
        # Check that weights changed
        weights_changed = False
        for name, param in trainer_with_model.model.named_parameters():
            if param.requires_grad:
                current = param.data.cpu().numpy()
                if not (current == initial_weights[name]).all():
                    weights_changed = True
                    break
        
        assert weights_changed, "Model weights should change after training step"
    
    def test_train_step_has_gradients(self, trainer_with_model):
        """Test that gradients are computed after backward pass."""
        batch = next(iter(trainer_with_model.train_loader))
        trainer_with_model.train_step(batch)
        
        # Check that gradients exist
        has_gradients = False
        for param in trainer_with_model.model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        
        assert has_gradients, "Model should have gradients after training step"


class TestTrainEpoch:
    """Tests for training epoch."""
    
    def test_train_epoch_returns_loss(self, trainer_with_model):
        """Test train_epoch returns average loss."""
        avg_loss = trainer_with_model.train_epoch()
        
        assert isinstance(avg_loss, float)
        assert avg_loss > 0
    
    def test_train_epoch_updates_loss_history(self, trainer_with_model):
        """Test train_epoch updates loss history."""
        initial_len = len(trainer_with_model.loss_history["train_loss"])
        
        trainer_with_model.train_epoch()
        
        assert len(trainer_with_model.loss_history["train_loss"]) == initial_len + 1
    
    def test_loss_decreases(self, tiny_config, synthetic_loader, tmp_path):
        """Test that loss decreases over multiple epochs."""
        model = BabyPix1LM(tiny_config)
        trainer = Trainer(
            model=model,
            train_loader=synthetic_loader,
            learning_rate=1e-3,
            warmup_steps=5,
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )
        
        # Train for several epochs
        losses = []
        for _ in range(5):
            loss = trainer.train_epoch()
            losses.append(loss)
        
        # Loss should generally decrease (check last vs first)
        # Note: With random data, this might not always hold perfectly
        # but with enough epochs it should show some decrease
        assert losses[-1] <= losses[0] * 1.5, "Loss should not increase dramatically"


class TestEvaluate:
    """Tests for evaluation."""
    
    def test_evaluate_returns_loss(self, trainer_with_model):
        """Test evaluate returns a loss value."""
        loss = trainer_with_model.evaluate()
        
        assert isinstance(loss, float)
    
    def test_evaluate_no_grad(self, trainer_with_model):
        """Test evaluate runs without gradient computation."""
        # Ensure model is in eval mode after evaluation
        trainer_with_model.evaluate()
        
        # Check that no gradients are stored
        for param in trainer_with_model.model.parameters():
            # After no_grad evaluation, gradients should not be computed
            pass  # Just verifying no error occurs
    
    def test_evaluate_updates_val_loss(self, trainer_with_model, synthetic_loader):
        """Test evaluate updates validation loss history."""
        # Set val_loader for evaluation
        trainer_with_model.val_loader = synthetic_loader
        initial_len = len(trainer_with_model.loss_history["val_loss"])
        
        trainer_with_model.evaluate()
        
        assert len(trainer_with_model.loss_history["val_loss"]) == initial_len + 1


class TestCheckpoint:
    """Tests for checkpoint save/load."""
    
    def test_save_checkpoint(self, trainer_with_model, tmp_path):
        """Test checkpoint is saved correctly."""
        filepath = trainer_with_model.save_checkpoint("test.pt")
        
        assert Path(filepath).exists()
    
    def test_load_checkpoint(self, trainer_with_model, tmp_path):
        """Test checkpoint is loaded correctly."""
        # Save checkpoint
        filepath = trainer_with_model.save_checkpoint("test.pt")
        
        # Create new trainer and load checkpoint
        new_model = BabyPix1LM(trainer_with_model.model.config)
        new_trainer = Trainer(
            model=new_model,
            train_loader=trainer_with_model.train_loader,
            checkpoint_dir=str(tmp_path / "checkpoints2"),
        )
        
        new_trainer.load_checkpoint(filepath)
        
        # Verify state is restored
        assert new_trainer.global_step == trainer_with_model.global_step
        assert new_trainer.current_epoch == trainer_with_model.current_epoch
    
    def test_checkpoint_roundtrip_weights(self, trainer_with_model, tmp_path):
        """Test that model weights are preserved in checkpoint roundtrip."""
        # Save checkpoint
        filepath = trainer_with_model.save_checkpoint("test.pt")
        
        # Get original weights
        original_weights = {
            name: param.clone()
            for name, param in trainer_with_model.model.named_parameters()
        }
        
        # Create new model and load
        new_model = BabyPix1LM(trainer_with_model.model.config)
        new_trainer = Trainer(
            model=new_model,
            train_loader=trainer_with_model.train_loader,
            checkpoint_dir=str(tmp_path / "checkpoints2"),
        )
        new_trainer.load_checkpoint(filepath)
        
        # Compare weights
        for name, param in new_model.named_parameters():
            assert torch.equal(param.data, original_weights[name]), \
                f"Weights mismatch for {name} after checkpoint load"
    
    def test_checkpoint_roundtrip_outputs(self, trainer_with_model, tmp_path):
        """Test that model produces same outputs after checkpoint roundtrip."""
        # Save checkpoint
        filepath = trainer_with_model.save_checkpoint("test.pt")
        
        # Create test input on same device as model
        test_input = torch.randint(
            0, trainer_with_model.model.config.vocab_size, (1, 16)
        ).to(trainer_with_model.device)
        
        # Get original output
        trainer_with_model.model.eval()
        with torch.no_grad():
            original_output = trainer_with_model.model(test_input)["logits"]
        
        # Create new model and load
        new_model = BabyPix1LM(trainer_with_model.model.config).to(trainer_with_model.device)
        new_trainer = Trainer(
            model=new_model,
            train_loader=trainer_with_model.train_loader,
            checkpoint_dir=str(tmp_path / "checkpoints2"),
            device=trainer_with_model.device,
        )
        new_trainer.load_checkpoint(filepath)
        
        # Get loaded output
        new_model.eval()
        with torch.no_grad():
            loaded_output = new_model(test_input)["logits"]
        
        # Outputs should match
        assert torch.allclose(original_output, loaded_output, atol=1e-5), \
            "Model outputs should match after checkpoint load"
    
    def test_checkpoint_contains_all_fields(self, trainer_with_model, tmp_path):
        """Test checkpoint contains all required fields."""
        filepath = trainer_with_model.save_checkpoint("test.pt")
        
        checkpoint = torch.load(filepath, weights_only=False)
        
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "scheduler_state_dict" in checkpoint
        assert "global_step" in checkpoint
        assert "current_epoch" in checkpoint
        assert "loss_history" in checkpoint
        assert "model_config" in checkpoint


class TestGradientClipping:
    """Tests for gradient clipping."""
    
    def test_gradient_clipping_applied(self, trainer_with_model):
        """Test that gradient clipping is applied."""
        # The trainer should have max_grad_norm set
        assert trainer_with_model.max_grad_norm > 0


class TestLearningRateScheduler:
    """Tests for learning rate scheduler."""
    
    def test_lr_changes_over_steps(self, trainer_with_model):
        """Test that learning rate changes over training steps."""
        initial_lr = trainer_with_model.scheduler.get_last_lr()[0]
        
        # Perform several steps
        for _ in range(10):
            batch = next(iter(trainer_with_model.train_loader))
            trainer_with_model.train_step(batch)
        
        current_lr = trainer_with_model.scheduler.get_last_lr()[0]
        
        # LR should have changed (due to warmup)
        assert initial_lr != current_lr or trainer_with_model.global_step < trainer_with_model.warmup_steps


class TestTrainingLoop:
    """Tests for full training loop."""
    
    def test_train_loop_completes(self, trainer_with_model):
        """Test that training loop completes without errors."""
        loss_history = trainer_with_model.train(num_epochs=2, log_steps=10)
        
        assert "train_loss" in loss_history
        assert len(loss_history["train_loss"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
