#!/usr/bin/env python3
"""
Optimized clickbait classification using OpenAI API with async concurrency,
checkpointing, and incremental result persistence.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from openai import AsyncOpenAI, RateLimitError, APIError
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 150
DEFAULT_MAX_CONCURRENT = 5
DEFAULT_FEW_SHOT_K = 5
CHUNK_SIZE = 100
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1.0

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_few_shot_examples(train_file: str, k: int = 5) -> List[Dict[str, str]]:
    """Load balanced few-shot examples from training data."""
    df = pd.read_csv(train_file, encoding='utf-8')

    # Sample balanced examples (half from each class)
    k_per_class = k // 2
    positive = df[df['label'] == 1].sample(n=min(k_per_class, len(df[df['label'] == 1])), random_state=42)
    negative = df[df['label'] == 0].sample(n=min(k - k_per_class, len(df[df['label'] == 0])), random_state=42)

    examples = []
    for _, row in pd.concat([positive, negative]).iterrows():
        examples.append({
            'title': row['title'],
            'label': 'Yes' if row['label'] == 1 else 'No'
        })

    return examples


def build_prompt(title: str, few_shot_examples: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Build optimized prompt with system message and few-shot examples."""
    system_message = (
        "You are a clickbait detector. Determine if a title is clickbait.\n"
        "Answer ONLY with 'Yes' (if clickbait) or 'No' (if not clickbait).\n"
        "Do not provide any explanation or additional text."
    )

    messages = [{"role": "system", "content": system_message}]

    # Add few-shot examples
    for example in few_shot_examples:
        messages.append({
            "role": "user",
            "content": f"Title: {example['title']}"
        })
        messages.append({
            "role": "assistant",
            "content": example['label']
        })

    # Add the actual query
    messages.append({
        "role": "user",
        "content": f"Title: {title}"
    })

    return messages


def parse_label(response: str) -> Tuple[Optional[int], bool]:
    """
    Parse model response to extract label.
    Returns: (label, is_ambiguous)
        label: 1 for clickbait, 0 for not clickbait, None if ambiguous
        is_ambiguous: True if response couldn't be parsed clearly
    """
    # Normalize: lowercase, strip punctuation and whitespace
    normalized = re.sub(r'[^\w\s]', '', response.lower().strip())

    # Check for whole word matches
    if re.search(r'\byes\b', normalized):
        return 1, False
    elif re.search(r'\bno\b', normalized):
        return 0, False
    else:
        # Ambiguous response
        logger.warning(f"Ambiguous response: '{response}'")
        return None, True


async def api_call_with_retry(
    client: AsyncOpenAI,
    messages: List[Dict[str, str]],
    model: str,
    max_tokens: int,
    temperature: float
) -> str:
    """Make API call with exponential backoff retry on rate limit errors."""
    delay = INITIAL_RETRY_DELAY

    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()

        except (RateLimitError, APIError) as e:
            if attempt == MAX_RETRIES - 1:
                logger.error(f"Max retries exceeded: {e}")
                raise

            logger.warning(f"API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {delay}s...")
            await asyncio.sleep(delay)
            delay *= 2  # Exponential backoff

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

    raise Exception("Should not reach here")


def read_checkpoint(checkpoint_file: Path) -> int:
    """Read last processed index from checkpoint file."""
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                return int(f.read().strip())
        except (ValueError, IOError) as e:
            logger.warning(f"Could not read checkpoint: {e}")
    return -1


def write_checkpoint(checkpoint_file: Path, index: int):
    """Write last processed index to checkpoint file."""
    with open(checkpoint_file, 'w') as f:
        f.write(str(index))


def append_predictions(
    output_file: Path,
    predictions: List[Dict],
    write_header: bool = False
):
    """Append predictions to output CSV file."""
    df = pd.DataFrame(predictions)
    df.to_csv(
        output_file,
        mode='w' if write_header else 'a',
        header=write_header,
        index=False,
        encoding='utf-8'
    )


def compute_and_save_metrics(
    predictions_file: Path,
    metrics_file: Path
):
    """Compute classification metrics and save to JSON."""
    df = pd.read_csv(predictions_file, encoding='utf-8')

    # Filter out rows with None predictions (ambiguous responses)
    valid_df = df[df['predicted_label'].notna()].copy()

    if len(valid_df) == 0:
        logger.error("No valid predictions to compute metrics")
        return

    y_true = valid_df['true_label'].astype(int)
    y_pred = valid_df['predicted_label'].astype(int)

    # Compute confusion matrix components
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()

    metrics = {
        'TP': int(tp),
        'FP': int(fp),
        'FN': int(fn),
        'TN': int(tn),
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'F1': float(f1_score(y_true, y_pred, zero_division=0)),
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'Total_Samples': len(df),
        'Valid_Predictions': len(valid_df),
        'Ambiguous_Responses': len(df) - len(valid_df)
    }

    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved to {metrics_file}")
    logger.info(f"Accuracy: {metrics['Accuracy']:.4f}, F1: {metrics['F1']:.4f}")


# ============================================================================
# ASYNC INFERENCE
# ============================================================================

async def process_sample(
    client: AsyncOpenAI,
    index: int,
    title: str,
    true_label: int,
    few_shot_examples: List[Dict[str, str]],
    model: str,
    max_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore
) -> Dict:
    """Process a single sample with concurrency control."""
    async with semaphore:
        messages = build_prompt(title, few_shot_examples)

        try:
            raw_response = await api_call_with_retry(
                client, messages, model, max_tokens, temperature
            )
            predicted_label, is_ambiguous = parse_label(raw_response)

            return {
                'index': index,
                'title': title,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'raw_response': raw_response
            }

        except Exception as e:
            logger.error(f"Failed to process index {index}: {e}")
            return {
                'index': index,
                'title': title,
                'true_label': true_label,
                'predicted_label': None,
                'raw_response': f"ERROR: {str(e)}"
            }


async def run_inference(
    train_file: str,
    test_file: str,
    api_key: str,
    model: str,
    output_dir: Path,
    max_concurrent: int,
    few_shot_k: int,
    max_tokens: int,
    temperature: float,
    base_url: Optional[str] = None
):
    """Main async inference function with checkpointing and batch processing."""
    # Initialize
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = output_dir / 'predictions.csv'
    checkpoint_file = output_dir / 'checkpoint.txt'

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    semaphore = asyncio.Semaphore(max_concurrent)

    # Load few-shot examples
    logger.info(f"Loading {few_shot_k} few-shot examples from {train_file}")
    few_shot_examples = load_few_shot_examples(train_file, few_shot_k)

    # Read checkpoint
    last_processed = read_checkpoint(checkpoint_file)
    logger.info(f"Resuming from index {last_processed + 1}")

    # Detect encoding and load test data
    try:
        test_df = pd.read_csv(test_file, encoding='gbk')
    except UnicodeDecodeError:
        test_df = pd.read_csv(test_file, encoding='utf-8')

    # Filter to unprocessed rows
    test_df = test_df.reset_index(drop=True)
    test_df = test_df[test_df.index > last_processed]

    if len(test_df) == 0:
        logger.info("All samples already processed")
        return

    logger.info(f"Processing {len(test_df)} samples in chunks of {CHUNK_SIZE}")

    # Write header if starting fresh
    write_header = last_processed == -1

    # Process in chunks
    for chunk_start in range(0, len(test_df), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(test_df))
        chunk = test_df.iloc[chunk_start:chunk_end]

        logger.info(f"Processing chunk {chunk_start}-{chunk_end}")

        # Create async tasks for chunk
        tasks = []
        for idx, row in chunk.iterrows():
            task = process_sample(
                client=client,
                index=idx,
                title=row['title'],
                true_label=int(row['label']),
                few_shot_examples=few_shot_examples,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                semaphore=semaphore
            )
            tasks.append(task)

        # Execute chunk concurrently
        results = await asyncio.gather(*tasks)

        # Save chunk results
        append_predictions(predictions_file, results, write_header)
        write_header = False

        # Update checkpoint to last index in chunk
        last_idx = results[-1]['index']
        write_checkpoint(checkpoint_file, last_idx)

        logger.info(f"Chunk complete. Checkpoint updated to {last_idx}")

    logger.info("Inference complete")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Optimized clickbait classification with OpenAI API'
    )
    parser.add_argument(
        '--train_file',
        type=str,
        default='clickbait_train.csv',
        help='Path to training CSV file'
    )
    parser.add_argument(
        '--test_file',
        type=str,
        default='clickbait_test.csv',
        help='Path to test CSV file'
    )
    parser.add_argument(
        '--api_key',
        type=str,
        required=True,
        help='OpenAI API key'
    )
    parser.add_argument(
        '--base_url',
        type=str,
        default='https://api-v2.shopaikey.com/v1',
        help='OpenAI API base URL (optional, for custom endpoints)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL,
        help=f'OpenAI model to use (default: {DEFAULT_MODEL})'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./gpt_output',
        help='Directory for output files (default: ./gpt_output)'
    )
    parser.add_argument(
        '--max_concurrent',
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help=f'Max concurrent API requests (default: {DEFAULT_MAX_CONCURRENT})'
    )
    parser.add_argument(
        '--few_shot_k',
        type=int,
        default=DEFAULT_FEW_SHOT_K,
        help=f'Number of few-shot examples (default: {DEFAULT_FEW_SHOT_K})'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f'Max tokens in response (default: {DEFAULT_MAX_TOKENS})'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f'Model temperature (default: {DEFAULT_TEMPERATURE})'
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    # Run inference
    try:
        asyncio.run(run_inference(
            train_file=args.train_file,
            test_file=args.test_file,
            api_key=args.api_key,
            model=args.model,
            output_dir=output_dir,
            max_concurrent=args.max_concurrent,
            few_shot_k=args.few_shot_k,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            base_url=args.base_url
        ))

        # Compute final metrics
        predictions_file = output_dir / 'predictions.csv'
        metrics_file = output_dir / 'metrics.json'

        if predictions_file.exists():
            compute_and_save_metrics(predictions_file, metrics_file)
        else:
            logger.error("No predictions file found")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
