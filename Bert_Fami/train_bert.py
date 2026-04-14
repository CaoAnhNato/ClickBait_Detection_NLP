#!/usr/bin/env python3
"""
Fine-tune BERT-based models for clickbait detection
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import csv
from datetime import datetime
import time
from typing import Optional, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Check PyTorch and transformers compatibility
def check_compatibility():
    """Check if PyTorch and transformers versions are compatible"""
    torch_version = torch.__version__.split('+')[0]
    torch_major, torch_minor = map(int, torch_version.split('.')[:2])

    print(f"PyTorch version: {torch.__version__}")

    # Check transformers
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")

        # For PyTorch 2.2.x, need transformers < 4.46
        if torch_major == 2 and torch_minor < 4:
            trans_version = transformers.__version__.split('.')
            trans_major, trans_minor = int(trans_version[0]), int(trans_version[1])

            if trans_major >= 4 and trans_minor >= 46:
                print("\n" + "="*80)
                print("WARNING: Version Incompatibility Detected!")
                print("="*80)
                print(f"PyTorch {torch.__version__} is incompatible with transformers {transformers.__version__}")
                print(f"Transformers >= 4.46 requires PyTorch >= 2.4")
                print("\nSOLUTIONS:")
                print("1. Downgrade transformers (Recommended for PyTorch 2.2.1):")
                print("   pip install transformers==4.45.2")
                print("\n2. Upgrade PyTorch (Recommended for RTX 5080):")
                print("   pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                print("="*80 + "\n")
                return False
    except ImportError:
        print("transformers not found!")
        return False

    # Check CUDA compatibility for RTX 5080 (Blackwell sm_120)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if 'RTX 5080' in gpu_name or 'RTX 50' in gpu_name:
            if torch_major == 2 and torch_minor < 5:
                print("\n" + "="*80)
                print("WARNING: GPU Incompatibility Detected!")
                print("="*80)
                print(f"RTX 5080 (Blackwell architecture, sm_120) requires PyTorch >= 2.5")
                print(f"Current PyTorch {torch.__version__} does not support this GPU")
                print("\nSOLUTION:")
                print("Upgrade PyTorch to support RTX 5080:")
                print("  pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                print("="*80 + "\n")

                response = input("Continue training on CPU? (y/n): ")
                if response.lower() != 'y':
                    return False
                return 'cpu'

    return True

# Run compatibility check
compat_result = check_compatibility()
if compat_result == False:
    print("\nExiting due to compatibility issues. Please fix the versions above.")
    sys.exit(1)
elif compat_result == 'cpu':
    print("\nContinuing with CPU training...")
    FORCE_CPU = True
else:
    FORCE_CPU = False

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


class ClickbaitDataset(Dataset):
    """Custom Dataset for clickbait detection"""

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_data(data_path, test_size=0.2, random_state=42):
    """Load and split dataset"""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    texts = df['title'].values
    labels = df['label'].values

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Train distribution - 0: {sum(y_train==0)}, 1: {sum(y_train==1)}")
    print(f"Test distribution - 0: {sum(y_test==0)}, 1: {sum(y_test==1)}")

    return X_train, X_test, y_train, y_test


def train_epoch(model, dataloader, optimizer, scheduler, device, scaler=None, gradient_accumulation_steps=1):
    """Train for one epoch with mixed precision support"""
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    progress_bar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        # Mixed precision forward pass
        if scaler is not None:
            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / gradient_accumulation_steps

            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps
        progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, use_amp=False):
    """Evaluate model with mixed precision support"""
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            if use_amp:
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    macro_f1 = f1_score(true_labels, predictions, average='macro')
    clickbait_f1 = f1_score(true_labels, predictions, pos_label=1)

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'clickbait_f1': clickbait_f1,
        'predictions': predictions,
        'true_labels': true_labels
    }


# ============================================================
# Checkpoint and History Management Functions
# ============================================================

class EarlyStopping:
    """
    Early stopping handler to stop training when metric stops improving
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0001, mode: str = 'max'):
        """
        Args:
            patience: Number of epochs to wait without improvement (default: 5)
            min_delta: Minimum change to qualify as improvement (default: 0.0001)
            mode: 'max' for metrics to maximize (F1), 'min' for loss (default: 'max')
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score: float) -> bool:
        """
        Check if training should stop

        Args:
            current_score: Current metric value (clickbait_f1)

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == 'max':
            improved = current_score > (self.best_score + self.min_delta)
        else:
            improved = current_score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


def save_checkpoint_metadata(
    save_path: Path,
    model_name: str,
    epoch: int,
    total_epochs: int,
    metrics: dict,
    args,
    training_start_time: float
) -> None:
    """
    Save metadata for best checkpoint

    Args:
        save_path: Path to model directory (same as model weights)
        model_name: Original model name from HuggingFace
        epoch: Current epoch number (1-indexed)
        total_epochs: Total epochs configured
        metrics: Dict with keys: accuracy, macro_f1, clickbait_f1
        args: Command-line arguments containing hyperparameters
        training_start_time: Training start timestamp (time.time())
    """
    try:
        metadata = {
            "model_name": model_name,
            "checkpoint_info": {
                "epoch": epoch,
                "total_epochs": total_epochs,
                "best_metric": "clickbait_f1",
                "best_metric_value": float(metrics['clickbait_f1'])
            },
            "metrics": {
                "accuracy": float(metrics['accuracy']),
                "macro_f1": float(metrics['macro_f1']),
                "clickbait_f1": float(metrics['clickbait_f1'])
            },
            "hyperparameters": {
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "use_amp": args.use_amp,
                "compile_model": args.compile_model
            },
            "timestamp": datetime.now().isoformat(),
            "training_time_seconds": float(time.time() - training_start_time)
        }

        # Atomic write: write to temp file, then rename
        metadata_path = save_path / 'best_checkpoint_metadata.json'
        temp_path = save_path / 'best_checkpoint_metadata.json.tmp'

        with open(temp_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        temp_path.replace(metadata_path)

    except Exception as e:
        print(f"⚠️  Warning: Could not save checkpoint metadata: {e}")


def save_training_history(
    save_path: Path,
    epoch: int,
    train_loss: float,
    metrics: dict,
    is_best: bool,
    training_start_time: float
) -> None:
    """
    Append epoch results to training history CSV

    Args:
        save_path: Path to model directory
        epoch: Current epoch number (1-indexed)
        train_loss: Average training loss
        metrics: Dict with keys: accuracy, macro_f1, clickbait_f1
        is_best: Whether this is the best epoch so far
        training_start_time: Training start timestamp
    """
    try:
        history_path = save_path / 'training_history.csv'
        file_exists = history_path.exists()

        with open(history_path, 'a', newline='') as f:
            writer = csv.writer(f)

            # Write header if file is new
            if not file_exists:
                writer.writerow([
                    'epoch', 'train_loss', 'accuracy', 'macro_f1',
                    'clickbait_f1', 'is_best', 'timestamp', 'elapsed_seconds'
                ])

            # Write epoch data
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{metrics['accuracy']:.6f}",
                f"{metrics['macro_f1']:.6f}",
                f"{metrics['clickbait_f1']:.6f}",
                is_best,
                datetime.now().isoformat(),
                f"{time.time() - training_start_time:.2f}"
            ])

    except Exception as e:
        print(f"⚠️  Warning: Could not save training history: {e}")


def save_model_evaluation(
    save_path: Path,
    model_name: str,
    best_epoch: int,
    best_metrics: dict,
    final_metrics: dict,
    early_stopped: bool,
    early_stop_epoch: Optional[int],
    total_epochs_trained: int,
    training_start_time: float
) -> None:
    """
    Save individual model evaluation results

    Args:
        save_path: Path to model directory
        model_name: Model name from HuggingFace
        best_epoch: Epoch that achieved best performance
        best_metrics: Dict of metrics at best epoch
        final_metrics: Dict of metrics at last epoch
        early_stopped: Whether training was early stopped
        early_stop_epoch: Epoch where training stopped (if early stopped)
        total_epochs_trained: Total number of epochs completed
        training_start_time: Training start timestamp
    """
    try:
        evaluation = {
            "model_name": model_name,
            "training_completed": True,
            "early_stopped": early_stopped,
            "early_stop_epoch": early_stop_epoch if early_stopped else None,
            "best_checkpoint": {
                "epoch": best_epoch,
                "metrics": {
                    "accuracy": float(best_metrics['accuracy']),
                    "macro_f1": float(best_metrics['macro_f1']),
                    "clickbait_f1": float(best_metrics['clickbait_f1'])
                }
            },
            "final_metrics": {
                "accuracy": float(final_metrics['accuracy']),
                "macro_f1": float(final_metrics['macro_f1']),
                "clickbait_f1": float(final_metrics['clickbait_f1'])
            },
            "total_epochs_trained": total_epochs_trained,
            "total_training_time_seconds": float(time.time() - training_start_time),
            "timestamp": datetime.now().isoformat()
        }

        # Atomic write
        eval_path = save_path / 'final_evaluation.json'
        temp_path = save_path / 'final_evaluation.json.tmp'

        with open(temp_path, 'w') as f:
            json.dump(evaluation, f, indent=2)

        temp_path.replace(eval_path)

    except Exception as e:
        print(f"⚠️  Warning: Could not save model evaluation: {e}")


def save_aggregate_results(
    output_dir: Path,
    session_start_time: float,
    args,
    model_results: List[dict]
) -> None:
    """
    Save or update aggregate results after each model

    Args:
        output_dir: Base output directory
        session_start_time: When training session started
        args: Command-line arguments
        model_results: List of individual model results
    """
    try:
        # Find best overall model
        best_overall = None
        best_f1 = -1
        for result in model_results:
            if result.get('status') == 'completed':
                f1 = result['best_metrics']['clickbait_f1']
                if f1 > best_f1:
                    best_f1 = f1
                    best_overall = {
                        'model_name': result['model_name'],
                        'clickbait_f1': f1
                    }

        aggregate = {
            "training_session": {
                "start_time": datetime.fromtimestamp(session_start_time).isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_duration_seconds": float(time.time() - session_start_time),
                "hyperparameters": {
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "max_length": args.max_length,
                    "epochs": args.epochs,
                    "patience": args.patience,
                    "use_amp": args.use_amp
                }
            },
            "models": model_results,
            "best_overall_model": best_overall
        }

        # Atomic write
        results_path = output_dir / 'evaluation_results.json'
        temp_path = output_dir / 'evaluation_results.json.tmp'

        with open(temp_path, 'w') as f:
            json.dump(aggregate, f, indent=2)

        temp_path.replace(results_path)

    except Exception as e:
        print(f"⚠️  Warning: Could not save aggregate results: {e}")


def train_model(model_name, X_train, X_test, y_train, y_test, args, device):
    """Train a single model with GPU optimizations"""
    print(f"\n{'='*80}")
    print(f"Training model: {model_name}")
    print(f"{'='*80}")

    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True  # Fix for models pre-trained with different num_labels (e.g. BART-MNLI: 3 classes)
    ).to(device)

    # Enable torch.compile for PyTorch 2.x speedup
    if args.compile_model and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # Initialize tracking
    training_start_time = time.time()
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        mode='max'
    ) if args.patience > 0 else None

    # Create save path early
    save_path = Path(args.output_dir) / model_name.replace('/', '_')
    save_path.mkdir(parents=True, exist_ok=True)

    # Track best metrics (not just best_f1)
    best_f1 = 0
    best_epoch = 0
    best_metrics = {'accuracy': 0, 'macro_f1': 0, 'clickbait_f1': 0}

    # Create datasets
    train_dataset = ClickbaitDataset(X_train, y_train, tokenizer, args.max_length)
    test_dataset = ClickbaitDataset(X_test, y_test, tokenizer, args.max_length)

    # Optimized DataLoader for GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False
    )

    # Setup optimizer (use torch.optim.AdamW instead of transformers.AdamW)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )

    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Setup mixed precision training
    scaler = GradScaler() if args.use_amp else None
    if args.use_amp:
        print("Using Automatic Mixed Precision (AMP) for faster training")

    # Training loop
    best_f1 = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        avg_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            scaler=scaler,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
        print(f"Average training loss: {avg_loss:.4f}")

        # Evaluate
        results = evaluate(model, test_loader, device, use_amp=args.use_amp)
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Macro F1: {results['macro_f1']:.4f}")
        print(f"Clickbait F1: {results['clickbait_f1']:.4f}")

        # Check if best
        is_best = results['clickbait_f1'] > best_f1

        # Save training history
        save_training_history(
            save_path=save_path,
            epoch=epoch + 1,  # 1-indexed
            train_loss=avg_loss,
            metrics=results,
            is_best=is_best,
            training_start_time=training_start_time
        )

        # Save best model + metadata
        if is_best:
            best_f1 = results['clickbait_f1']
            best_epoch = epoch + 1
            best_metrics = {
                'accuracy': results['accuracy'],
                'macro_f1': results['macro_f1'],
                'clickbait_f1': results['clickbait_f1']
            }

            # Unwrap compiled model before saving
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            model_to_save.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

            # Save metadata
            save_checkpoint_metadata(
                save_path=save_path,
                model_name=model_name,
                epoch=epoch + 1,
                total_epochs=args.epochs,
                metrics=results,
                args=args,
                training_start_time=training_start_time
            )
            print(f"✓ Saved best model to {save_path}")

        # Early stopping check
        if early_stopping is not None:
            if early_stopping(results['clickbait_f1']):
                print(f"\n🛑 Early stopping triggered after epoch {epoch + 1}")
                print(f"   Best Clickbait F1: {best_f1:.4f} (epoch {best_epoch})")
                print(f"   No improvement for {early_stopping.patience} consecutive epochs")
                break
            elif early_stopping.counter > 0:
                print(f"⚠️  No improvement for {early_stopping.counter}/{early_stopping.patience} epochs")

    # Final evaluation
    print(f"\nFinal evaluation for {model_name}...")
    final_results = evaluate(model, test_loader, device, use_amp=args.use_amp)

    # Calculate final stats
    training_duration = time.time() - training_start_time
    early_stopped = early_stopping is not None and early_stopping.early_stop
    actual_epochs_trained = epoch + 1  # Last completed epoch

    # Save individual model evaluation
    save_model_evaluation(
        save_path=save_path,
        model_name=model_name,
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        final_metrics=final_results,
        early_stopped=early_stopped,
        early_stop_epoch=actual_epochs_trained if early_stopped else None,
        total_epochs_trained=actual_epochs_trained,
        training_start_time=training_start_time
    )

    # Return enhanced results
    return {
        'model_name': model_name,
        'status': 'completed',
        'best_epoch': best_epoch,
        'best_metrics': best_metrics,
        'final_metrics': {
            'accuracy': final_results['accuracy'],
            'macro_f1': final_results['macro_f1'],
            'clickbait_f1': final_results['clickbait_f1']
        },
        'early_stopped': early_stopped,
        'epochs_trained': actual_epochs_trained,
        'training_time_seconds': training_duration
    }


def main():
    parser = argparse.ArgumentParser(description='Fine-tune BERT models for clickbait detection')

    # Data arguments
    parser.add_argument('-d', '--data', type=str,
                       default='clickbait_data.csv',
                       help='Path to dataset CSV file')
    parser.add_argument('-o', '--output-dir', type=str,
                       default='weights',
                       help='Output directory for saved models')

    # Model arguments
    parser.add_argument('-m', '--models', nargs='+',
                       default=[
                           'google-bert/bert-large-cased-whole-word-masking',
                           'FacebookAI/roberta-large',
                           'facebook/bart-large-mnli'
                       ],
                       help='List of model names from HuggingFace')

    # Training arguments
    parser.add_argument('-lr', '--learning-rate', type=float,
                       default=3e-5,
                       help='Learning rate')
    parser.add_argument('-b', '--batch-size', type=int,
                       default=16,
                       help='Batch size (increased for RTX 5080 16GB)')
    parser.add_argument('-ml', '--max-length', type=int,
                       default=100,
                       help='Maximum sequence length')
    parser.add_argument('-e', '--epochs', type=int,
                       default=50,
                       help='Number of epochs')
    parser.add_argument('-nw', '--num-workers', type=int,
                       default=15,
                       help='Number of workers for DataLoader (4-8 recommended for GPU training)')
    parser.add_argument('-ga', '--gradient-accumulation-steps', type=int,
                       default=1,
                       help='Gradient accumulation steps (simulate larger batch size)')

    # GPU Optimization arguments
    parser.add_argument('--use-amp', action='store_true',
                       help='Use Automatic Mixed Precision for faster training (recommended)')
    parser.add_argument('--compile-model', action='store_true',
                       help='Compile model with torch.compile() for PyTorch 2.x speedup')

    # Other arguments
    parser.add_argument('-ts', '--test-size', type=float,
                       default=0.2,
                       help='Test set size (default: 0.2 for 80-20 split)')
    parser.add_argument('-s', '--seed', type=int,
                       default=42,
                       help='Random seed')

    # Early stopping arguments
    parser.add_argument('--patience', type=int,
                       default=10,
                       help='Early stopping patience (epochs without improvement). Set to 0 to disable. (default: 10)')
    parser.add_argument('--min-delta', type=float,
                       default=0.0001,
                       help='Minimum metric improvement to count as progress (default: 0.0001)')

    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup device (respect FORCE_CPU from compatibility check)
    if FORCE_CPU:
        device = torch.device('cpu')
        print("Using device: cpu (forced due to GPU incompatibility)")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

    # GPU optimizations for RTX 5080
    if device.type == 'cuda':
        # Enable cuDNN auto-tuner to find the best algorithm
        torch.backends.cudnn.benchmark = True

        # Enable TF32 for faster matmul on Ampere+ GPUs (RTX 30xx, 40xx, 50xx)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.2f} GB")
        print(f"Mixed Precision (AMP): {'Enabled' if args.use_amp else 'Disabled'}")
        print(f"Model Compilation: {'Enabled' if args.compile_model else 'Disabled'}")
        print(f"Effective Batch Size: {args.batch_size * args.gradient_accumulation_steps}")
    else:
        # Disable GPU-only features
        args.use_amp = False
        print("Mixed Precision (AMP): Disabled (CPU mode)")
        print(f"Model Compilation: {'Enabled' if args.compile_model else 'Disabled'}")

    # Load data
    X_train, X_test, y_train, y_test = load_data(
        args.data,
        test_size=args.test_size,
        random_state=args.seed
    )

    # Track session start time
    session_start_time = time.time()

    # Train each model
    all_results = []
    for model_name in args.models:
        try:
            results = train_model(
                model_name,
                X_train, X_test, y_train, y_test,
                args, device
            )
            all_results.append(results)

            # Save aggregate results after EACH model
            save_aggregate_results(
                output_dir=Path(args.output_dir),
                session_start_time=session_start_time,
                args=args,
                model_results=all_results
            )

        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            # Track error
            all_results.append({
                'model_name': model_name,
                'status': 'error',
                'error_message': str(e),
                'training_time_seconds': 0
            })
            # Still save aggregate
            save_aggregate_results(
                output_dir=Path(args.output_dir),
                session_start_time=session_start_time,
                args=args,
                model_results=all_results
            )
            continue

    # Final summary
    print(f"\n{'='*80}")
    print("Training completed! Results summary:")
    print(f"{'='*80}")
    for result in all_results:
        if result.get('status') == 'completed':
            print(f"\nModel: {result['model_name']}")
            print(f"  Status: {'Early Stopped' if result['early_stopped'] else 'Completed'}")
            print(f"  Best Epoch: {result['best_epoch']}")
            print(f"  Best Clickbait F1: {result['best_metrics']['clickbait_f1']:.4f}")
            print(f"  Best Accuracy: {result['best_metrics']['accuracy']:.4f}")
            print(f"  Best Macro F1: {result['best_metrics']['macro_f1']:.4f}")
            print(f"  Epochs Trained: {result['epochs_trained']}/{args.epochs}")
            print(f"  Training Time: {result['training_time_seconds']:.1f}s")
        elif result.get('status') == 'error':
            print(f"\nModel: {result['model_name']}")
            print(f"  Status: ERROR - {result['error_message']}")

    results_path = Path(args.output_dir) / 'evaluation_results.json'
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
