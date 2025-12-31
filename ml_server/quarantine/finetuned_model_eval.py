"""
Fine-tuned MarianNMT Evaluation Script for Automotive Domain

This script evaluates a fine-tuned MarianNMT model for English to Spanish translation
specifically trained for the automotive domain, using test_set.csv.
"""
import sys
import pathlib
import os
import time
import csv
import re
import warnings
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import nltk
import shutil

# Add bert_score to path
repo = pathlib.Path("~/bert_score/")
sys.path.append(str(repo.expanduser()))
from bert_score.score import score

from transformers import MarianMTModel, MarianTokenizer
from sacrebleu.metrics import BLEU, CHRF
from nltk.translate.meteor_score import meteor_score

# Disable NLTK downloads silently
nltk.download = lambda *args, **kwargs: None

# Configuration
os.environ["TRANSFORMERS_OFFLINE"] = "1"
TEST_CSV_PATH = os.path.join(os.getenv("SCRATCH"), "test_set.csv")
OUTPUT_FILE = os.path.join(os.getenv("SCRATCH"), "finetuned_marianmt_results.csv")
BATCH_SIZE = 64
FINETUNED_MODEL_PATH = os.path.join(os.getenv("HOME"), "synthesis-II/marianNMT_automobiles")
ORIGINAL_MODEL_PATH = os.path.join(os.getenv("HOME"), "synthesis-II/opus-mt-en-es")

# Suppress warnings
warnings.filterwarnings("ignore")

def load_test_data(csv_path):
    """Load test data from a CSV file"""
    try:
        df = pd.read_csv(csv_path, delimiter=',', encoding='utf-8-sig')
    except:
        df = pd.read_csv(csv_path, delimiter=';', encoding='utf-8-sig')

    # Ensure the required columns exist
    required_cols = ['source', 'target']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV file must contain columns: {required_cols}")

    df = df[required_cols]
    print(f"Loaded {len(df)} test translation pairs")
    return df

class CustomMarianMTModel(MarianMTModel):
    """Custom version of MarianMTModel that gives more details on errors during saving/loading"""
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        try:
            return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        except Exception as e:
            print(f"Detailed error during model loading: {str(e)}")
            raise e

def fix_state_dict_keys(state_dict):
    """Fix state dict keys if they don't match the base model architecture"""
    # Common key mapping from fine-tuned to base model 
    fixed_state_dict = {}
    for key, value in state_dict.items():
        # Remove module. prefix if present (from DataParallel)
        if key.startswith('module.'):
            key = key[7:]
            
        # Add model. prefix if missing
        if not key.startswith('model.') and not key.startswith('encoder.') and not key.startswith('decoder.'):
            key = 'model.' + key
            
        fixed_state_dict[key] = value
    return fixed_state_dict

class FinetunedMarianModel:
    """Class to load the MarianMT fine-tuned model"""
    def __init__(self, finetuned_path, original_path):
        self.name = "MarianNMT_Finetuned_Automotive"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Try different approaches to load the model
        try:
            # APPROACH 1: Try loading the fine-tuned model directly (might fail with HeaderTooLarge)
            print("ATTEMPT 1: Loading fine-tuned model directly...")
            try:
                # First try loading as standard MarianMT model
                self.tokenizer = MarianTokenizer.from_pretrained(original_path)
                self.model = CustomMarianMTModel.from_pretrained(finetuned_path)
                print("SUCCESS: Loaded fine-tuned model directly!")
                
            except Exception as e1:
                print(f"FAILED direct loading: {str(e1)}")
                
                # APPROACH 2: Try with safetensors - load original model then replace weights
                print("\nATTEMPT 2: Loading original model and replacing with fine-tuned weights...")
                
                # First load original model 
                self.tokenizer = MarianTokenizer.from_pretrained(original_path)
                self.model = CustomMarianMTModel.from_pretrained(original_path)
                
                # Try loading weights from safetensors or PyTorch files
                safetensors_path = os.path.join(finetuned_path, "model.safetensors")
                pytorch_path = os.path.join(finetuned_path, "pytorch_model.bin")
                
                if os.path.exists(safetensors_path):
                    # Load from safetensors
                    print("Loading weights from safetensors file...")
                    from safetensors import safe_open
                    with safe_open(safetensors_path, framework="pt") as f:
                        state_dict = {k: f.get_tensor(k) for k in f.keys()}
                    
                    # Fix state dict keys if needed
                    state_dict = fix_state_dict_keys(state_dict)
                    
                    # Check for incompatible tensor sizes
                    for key, tensor in state_dict.items():
                        if key in self.model.state_dict():
                            if self.model.state_dict()[key].shape != tensor.shape:
                                print(f"WARNING: Shape mismatch for {key}: {self.model.state_dict()[key].shape} vs {tensor.shape}")
                    
                    # Load weights
                    missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                    
                    # Report on missing/unexpected keys
                    if len(missing) > 0:
                        print(f"Missing keys: {missing[:5]}... and {len(missing)-5} more")
                    if len(unexpected) > 0:
                        print(f"Unexpected keys: {unexpected[:5]}... and {len(unexpected)-5} more")
                    
                    print("SUCCESS: Loaded fine-tuned weights from safetensors!")
                
                elif os.path.exists(pytorch_path):
                    # Load from PyTorch weights
                    print("Loading weights from pytorch_model.bin file...")
                    state_dict = torch.load(pytorch_path, map_location="cpu")
                    
                    # Fix state dict keys if needed
                    state_dict = fix_state_dict_keys(state_dict)
                    
                    # Load weights
                    missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                    
                    # Report on missing/unexpected keys
                    if len(missing) > 0:
                        print(f"Missing keys: {missing[:5]}... and {len(missing)-5} more")
                    if len(unexpected) > 0:
                        print(f"Unexpected keys: {unexpected[:5]}... and {len(unexpected)-5} more")
                    
                    print("SUCCESS: Loaded fine-tuned weights from PyTorch bin!")
                
                else:
                    raise FileNotFoundError("Could not find model.safetensors or pytorch_model.bin")
                    
        except Exception as e:
            print(f"ERROR: All approaches failed to load fine-tuned model: {str(e)}")
            print("FALLBACK: Using original model without fine-tuning")
            
            self.tokenizer = MarianTokenizer.from_pretrained(original_path)
            self.model = MarianMTModel.from_pretrained(original_path)
            self.name = "MarianNMT_Original"
        
        # Move model to appropriate device
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        print(f"Model loaded on {self.device}")
        
    def detokenize(self, text):
        """Fix tokenization issues with punctuation marks"""
        # Fix spaces before punctuation marks
        text = re.sub(r'\s+([.,;:!?)\]}])', r'\1', text)
        # Fix spaces after opening parentheses/brackets
        text = re.sub(r'([\({\[])\s+', r'\1', text)
        return text

    def translate(self, texts):
        """Translate a list of texts"""
        try:
            # Create inputs
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            
            # Generate translations with no_grad for efficiency
            with torch.no_grad():
                translated = self.model.generate(
                    **inputs, 
                    num_beams=4, 
                    max_length=512,
                    early_stopping=True
                )
                
            # Decode translations
            decoded = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
            
            # Apply detokenization fix
            return [self.detokenize(text) for text in decoded]
            
        except Exception as e:
            print(f"ERROR during translation: {str(e)}")
            return ["Error during translation"] * len(texts)

    def translate_batch(self, texts, batch_size=8):
        """Translate texts in batches to avoid memory issues"""
        results = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Translating with {self.name}"):
            batch = texts[i:i+batch_size]
            translations = self.translate(batch)
            results.extend(translations)
        return results

def setup_nltk_resources():
    """Setup NLTK resources using local installation without downloading"""
    home_dir = os.path.expanduser("~")
    nltk_data_path = os.path.join(home_dir, "nltk_data")
    
    if not os.path.exists(nltk_data_path):
        print(f"Warning: {nltk_data_path} directory not found")
        return False

    # Add the custom path to NLTK's search paths
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_path)

    # Verify wordnet installation
    try:
        nltk.data.find('corpora/wordnet')
        print("Wordnet found through NLTK data finder")
        return True
    except LookupError:
        print("Wordnet not found in local installation. METEOR score will be skipped.")
        return False

def evaluate_translations(references, hypotheses, source_texts=None):
    """Evaluate translations using multiple metrics"""
    results = {}

    # Setup NLTK resources
    nltk_available = setup_nltk_resources()

    # BLEU score (0-100)
    bleu = BLEU()
    bleu_score = bleu.corpus_score(hypotheses, [references]).score
    results['bleu'] = bleu_score

    # CHRF score (0-100)
    chrf = CHRF()
    chrf_score = chrf.corpus_score(hypotheses, [references]).score
    results['chrf'] = chrf_score

    # METEOR score - only calculate if NLTK resources are available (0-1)
    if nltk_available:
        max_meteor_samples = min(len(references), 20000)
        meteor_scores = []
        for i in range(max_meteor_samples):
            ref_tokens = references[i].split()
            hyp_tokens = hypotheses[i].split()
            meteor_scores.append(meteor_score([ref_tokens], hyp_tokens))
        results['meteor'] = np.mean(meteor_scores)
    else:
        results['meteor'] = None

    # BERTScore (semantic similarity)
    print("Computing BERTScore (this may take a while)...")
    P, R, F1 = score(lang="es", rescale_with_baseline=False, model_type="/home/cvmsct06/bert-base-multilingual-cased", cands=hypotheses, refs=references)
    results['bertscore_f1'] = F1.mean().item()

    return results

def print_metric_info(metric, value):
    """Helper function to print metric information with ranges"""
    ranges = {
        'bleu': '0-100 (>30: good, >50: excellent)',
        'chrf': '0-100 (>50: good, >70: excellent)',
        'meteor': '0-1 (>0.4: good, >0.6: excellent)',
        'bertscore_f1': '0-1 (>0.85: good, >0.9: excellent)',
        'time_taken': 'seconds'
    }

    if metric == 'model':
        print(f"  {metric}: {value}")
    elif metric in ranges:
        print(f"  {metric}: {value:.4f} [Range: {ranges[metric]}]")
    else:
        print(f"  {metric}: {value:.4f}")

def main():
    # Load test data
    df = load_test_data(TEST_CSV_PATH)
    source_texts = df['source'].tolist()
    reference_translations = df['target'].tolist()

    # Initialize the model using our improved approach
    model = FinetunedMarianModel(FINETUNED_MODEL_PATH, ORIGINAL_MODEL_PATH)
    
    # Start timing
    start_time = time.time()

    # Translate texts
    model_translations = model.translate_batch(source_texts, batch_size=BATCH_SIZE)

    # Show first 3 samples with their translations
    print("\nFirst 3 samples with translations:")
    print("=" * 80)
    for i in range(min(3, len(source_texts))):
        print(f"\nSample {i+1}:")
        print(f"Source:     {source_texts[i]}")
        print(f"Model:      {model_translations[i]}")
        print(f"Reference:  {reference_translations[i]}")
        print("-" * 80)

    # Print whether we're using finetuned or original model
    print(f"\nUSING MODEL: {model.name}")
    if model.name == "MarianNMT_Original":
        print("WARNING: Failed to load fine-tuned model, using original model instead.")

    # Evaluate translations
    results = evaluate_translations(reference_translations, model_translations, source_texts)

    # Calculate time taken
    time_taken = time.time() - start_time
    results['model'] = model.name
    results['time_taken'] = time_taken

    # Print results with ranges
    print(f"\nResults for {model.name}:")
    for metric, value in results.items():
        if metric != 'model':
            print_metric_info(metric, value)

    # Save results to CSV
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)

    print(f"\nResults saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()