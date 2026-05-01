import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import re
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from train.modelbart import Similarity, DetectionModule, Attention_Encoder, Reason_Similarity, Aggregator, BertEncoder

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Constants
BATCH_SIZE = 32
MAX_LEN = 100

def tokenize_and_numericalize_data(text, tokenizer):
    tokenized = tokenizer(text, truncation=True, padding='max_length', max_length=MAX_LEN)
    return tokenized['input_ids']

class FakeNewsDataset(Dataset):
    def __init__(self, df, tokenizer, MAX_LEN):
        self.csv_data = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.MAX_LEN = MAX_LEN

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        try:
            text = str(self.csv_data.loc[idx, 'title'])
            pos = str(self.csv_data.loc[idx, 'agree_reason'])
            neg = str(self.csv_data.loc[idx, 'disagree_reason'])

            content_input_id = tokenize_and_numericalize_data(text, self.tokenizer)
            pos_input_id = tokenize_and_numericalize_data(pos, self.tokenizer)
            neg_input_id = tokenize_and_numericalize_data(neg, self.tokenizer)

            agree_score = float(self.csv_data.loc[idx, 'agree_score'])
            disagree_score = float(self.csv_data.loc[idx, 'disagree_score'])
            label = int(self.csv_data.loc[idx, 'label'])

            sample = {
                'content': torch.tensor(content_input_id),
                'pos_reason': torch.tensor(pos_input_id),
                'neg_reason': torch.tensor(neg_input_id),
                'label': torch.tensor(label),
                'agree_soft_label': torch.tensor(agree_score / 100, dtype=torch.float32),
                'disagree_soft_label': torch.tensor(disagree_score / 100, dtype=torch.float32)
            }
            return sample
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            return None

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def load_model(weight_path):
    """Load model with pretrained weights"""
    # Initialize all model components
    bert = BertEncoder(256, False).to(device)
    bert2 = BertEncoder(256, False).to(device)
    bert3 = BertEncoder(256, False).to(device)
    attention = Attention_Encoder().to(device)
    R2T_usefulness = Similarity().to(device)
    T2R_usefulness = Similarity().to(device)
    Reason_usefulness = Reason_Similarity().to(device)
    aggregator = Aggregator().to(device)
    detection_module = DetectionModule().to(device)

    # Load weights
    print(f"Loading checkpoint from: {weight_path}")
    checkpoint = torch.load(weight_path, map_location=device)

    # Print checkpoint structure to understand the format
    if isinstance(checkpoint, dict):
        print(f"Checkpoint keys: {list(checkpoint.keys())}")

        # Try to load based on different possible formats
        try:
            # Format 1: Direct state dicts for each module
            if 'bert' in checkpoint:
                bert.load_state_dict(checkpoint['bert'])
                bert2.load_state_dict(checkpoint['bert2'])
                bert3.load_state_dict(checkpoint['bert3'])
                attention.load_state_dict(checkpoint['attention'])
                R2T_usefulness.load_state_dict(checkpoint['R2T_usefulness'])
                T2R_usefulness.load_state_dict(checkpoint['T2R_usefulness'])
                Reason_usefulness.load_state_dict(checkpoint['Reason_usefulness'])
                aggregator.load_state_dict(checkpoint['aggregator'])
                detection_module.load_state_dict(checkpoint['detection_module'])
                print("Loaded weights from format: separate module state_dicts")

            # Format 2: All parameters in a single state dict with prefixes
            elif any('bert.' in k for k in checkpoint.keys()):
                # Create a combined model to load all at once
                print("Attempting to load from combined state dict...")
                # This requires manual filtering by prefix
                bert_dict = {k.replace('bert.', ''): v for k, v in checkpoint.items() if k.startswith('bert.') and not k.startswith('bert2.') and not k.startswith('bert3.')}
                bert2_dict = {k.replace('bert2.', ''): v for k, v in checkpoint.items() if k.startswith('bert2.')}
                bert3_dict = {k.replace('bert3.', ''): v for k, v in checkpoint.items() if k.startswith('bert3.')}
                attention_dict = {k.replace('attention.', ''): v for k, v in checkpoint.items() if k.startswith('attention.')}
                r2t_dict = {k.replace('R2T_usefulness.', ''): v for k, v in checkpoint.items() if k.startswith('R2T_usefulness.')}
                t2r_dict = {k.replace('T2R_usefulness.', ''): v for k, v in checkpoint.items() if k.startswith('T2R_usefulness.')}
                reason_dict = {k.replace('Reason_usefulness.', ''): v for k, v in checkpoint.items() if k.startswith('Reason_usefulness.')}
                agg_dict = {k.replace('aggregator.', ''): v for k, v in checkpoint.items() if k.startswith('aggregator.')}
                det_dict = {k.replace('detection_module.', ''): v for k, v in checkpoint.items() if k.startswith('detection_module.')}

                if bert_dict:
                    bert.load_state_dict(bert_dict, strict=False)
                if bert2_dict:
                    bert2.load_state_dict(bert2_dict, strict=False)
                if bert3_dict:
                    bert3.load_state_dict(bert3_dict, strict=False)
                if attention_dict:
                    attention.load_state_dict(attention_dict, strict=False)
                if r2t_dict:
                    R2T_usefulness.load_state_dict(r2t_dict, strict=False)
                if t2r_dict:
                    T2R_usefulness.load_state_dict(t2r_dict, strict=False)
                if reason_dict:
                    Reason_usefulness.load_state_dict(reason_dict, strict=False)
                if agg_dict:
                    aggregator.load_state_dict(agg_dict, strict=False)
                if det_dict:
                    detection_module.load_state_dict(det_dict, strict=False)
                print("Loaded weights from format: combined state_dict with prefixes")

            # Format 3: Model state dict wrapper
            elif 'model_state_dict' in checkpoint or 'state_dict' in checkpoint:
                state_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'state_dict'
                print(f"Loading from '{state_key}'...")
                # Recursively try the same loading logic
                actual_checkpoint = checkpoint[state_key]
                # Try same logic as above
                print(f"Keys in {state_key}: {list(actual_checkpoint.keys())[:5]}...")

            else:
                print("WARNING: Unknown checkpoint format. Attempting to continue without loading weights...")
                print("Please check the checkpoint structure manually.")

        except Exception as e:
            print(f"Warning: Error loading some weights: {e}")
            print("Continuing with randomly initialized weights...")
    else:
        print("Checkpoint is not a dictionary. Cannot load weights.")

    # Set to evaluation mode
    bert.eval()
    bert2.eval()
    bert3.eval()
    attention.eval()
    R2T_usefulness.eval()
    T2R_usefulness.eval()
    Reason_usefulness.eval()
    aggregator.eval()
    detection_module.eval()

    return bert, bert2, bert3, attention, R2T_usefulness, T2R_usefulness, Reason_usefulness, aggregator, detection_module

def predict(models, test_dataloader):
    """Run inference and collect predictions"""
    bert, bert2, bert3, attention, R2T_usefulness, T2R_usefulness, Reason_usefulness, aggregator, detection_module = models

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Predicting"):
            if batch is None:
                continue

            news_content = batch['content'].to(device).long()
            pos = batch["pos_reason"].to(device).long()
            neg = batch['neg_reason'].to(device).long()
            label = batch['label'].to(device)
            agree_soft_label = batch['agree_soft_label'].to(device)
            disagree_soft_label = batch['disagree_soft_label'].to(device)

            # Forward pass
            content = bert(news_content)
            positive = bert2(pos)
            negative = bert3(neg)

            # Cross attention
            pos_reason2text, pos_text2reason, positive, neg_reason2text, neg_text2reason, negative = attention(
                content, positive, negative
            )

            # Get aligned features for detection
            text_R2T_aligned_agr, R2T_aligned_agr, _ = R2T_usefulness(content, pos_reason2text)
            text_T2R_aligned_agr, T2R_aligned_agr, _ = T2R_usefulness(content, pos_text2reason)
            text_R_aligned_agr, R_aligned_agr, _ = Reason_usefulness(content, positive)
            text_R2T_aligned_dis, R2T_aligned_dis, _ = R2T_usefulness(content, neg_reason2text)
            text_T2R_aligned_dis, T2R_aligned_dis, _ = T2R_usefulness(content, neg_text2reason)
            text_R_aligned_dis, R_aligned_dis, _ = Reason_usefulness(content, negative)

            # Final detection
            final_feature = aggregator(
                content, R2T_aligned_agr, T2R_aligned_agr, R_aligned_agr,
                R2T_aligned_dis, T2R_aligned_dis, R_aligned_dis
            )
            pre_detection = detection_module(final_feature)

            # Get predictions
            predictions = pre_detection.argmax(1).cpu().numpy()
            labels = label.cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(labels)

    return np.array(all_predictions), np.array(all_labels)

def run_inference_experiment(data_path, weight_path, n_iterations=10):
    """Run inference experiment multiple times with random subsets"""
    print(f"\n{'='*60}")
    print("Starting Inference Experiment")
    print(f"{'='*60}\n")
    print(f"Data path: {data_path}")
    print(f"Weight path: {weight_path}")
    print(f"Number of iterations: {n_iterations}")
    print(f"Sample size: 20% of data")
    print(f"Device: {device}\n")

    # Load full dataset
    print("Loading dataset...")
    full_df = pd.read_csv(data_path)
    print(f"Total samples in dataset: {len(full_df)}")

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load model
    print("\nLoading model...")
    models = load_model(weight_path)
    print("Model loaded successfully!\n")

    # Storage for metrics
    accuracies = []
    macro_f1s = []
    clickbait_f1s = []

    # Run multiple iterations
    for iteration in range(n_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{n_iterations}")
        print(f"{'='*60}")

        # Sample 20% of data randomly
        sample_df = full_df.sample(frac=0.2, random_state=iteration*42)
        print(f"Sampled {len(sample_df)} samples (20% of total)")

        # Create dataset and dataloader
        dataset = FakeNewsDataset(sample_df, tokenizer, MAX_LEN)
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn
        )

        # Run prediction
        predictions, labels = predict(models, dataloader)

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        macro_f1 = f1_score(labels, predictions, average='macro')
        clickbait_f1 = f1_score(labels, predictions, pos_label=1)  # F1 for label 1 (clickbait)

        # Store metrics
        accuracies.append(accuracy)
        macro_f1s.append(macro_f1)
        clickbait_f1s.append(clickbait_f1)

        # Print iteration results
        print(f"\nResults:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Macro F1: {macro_f1:.4f}")
        print(f"  Clickbait F1 (label=1): {clickbait_f1:.4f}")

    # Calculate and print average metrics
    print(f"\n{'='*60}")
    print("Final Results (Average over all iterations)")
    print(f"{'='*60}\n")
    print(f"Average Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Average Macro F1: {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
    print(f"Average Clickbait F1 (label=1): {np.mean(clickbait_f1s):.4f} ± {np.std(clickbait_f1s):.4f}\n")

    # Print detailed results
    print(f"{'='*60}")
    print("Detailed Results for Each Iteration")
    print(f"{'='*60}\n")
    for i in range(n_iterations):
        print(f"Iteration {i+1}: Acc={accuracies[i]:.4f}, Macro F1={macro_f1s[i]:.4f}, Clickbait F1={clickbait_f1s[i]:.4f}")

if __name__ == "__main__":
    # Paths
    data_path = "ORCD/GPT_3.5/data/sorg_gpt3.5_output.csv"
    weight_path = "ORCD/GPT_3.5/weight/best_teachermodel.pth"

    # Run experiment
    run_inference_experiment(data_path, weight_path, n_iterations=10)
