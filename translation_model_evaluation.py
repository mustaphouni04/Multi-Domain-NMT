"""
Translation Model Evaluation Script

This script evaluates three different open-source translation models for English to Spanish translation
using a CSV file containing reference translations. The evaluation includes semantic and grammatical metrics.
"""

import pandas as pd
import numpy as np
import torch
import csv
import time
import argparse
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from bert_score import BERTScorer
from sacrebleu.metrics import BLEU, CHRF
from nltk.translate.meteor_score import meteor_score
import nltk
import warnings

# Configuration
CSV_PATH = r"Splits/PHP_test_set.csv"
MAX_SAMPLES = 500  # Size of the test set
OUTPUT_FILE = "warmup_results/php_true.csv"
BATCH_SIZE = 1

# Suppress warnings
warnings.filterwarnings("ignore")

def load_translations(csv_path, max_samples=None):
    """
    Load translations from a CSV file
    
    Args:
        csv_path: Path to the CSV file
        max_samples: Maximum number of samples to load (None for all)
        
    Returns:
        DataFrame with source and target texts
    """
    try:
        # Try with semicolon delimiter first (common in European CSV files)
        df = pd.read_csv(csv_path, delimiter=';', encoding='utf-8-sig')
    except:
        # Fall back to comma delimiter
        df = pd.read_csv(csv_path, delimiter=',', encoding='utf-8-sig')
    
    # Ensure the required columns exist
    required_cols = ['source', 'target']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV file must contain columns: {required_cols}")
    
    # Select only the required columns
    df = df[required_cols]
    
    # Limit the number of samples if specified
    if max_samples is not None and max_samples < len(df):
        df = df.sample(max_samples, random_state=42)
    
    print(f"Loaded {len(df)} translation pairs")
    return df

class TranslationModel:
    """Base class for translation models"""
    def __init__(self, name):
        self.name = name
        
    def translate(self, texts):
        """Translate a list of texts"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def translate_batch(self, texts, batch_size=8):
        """Translate texts in batches to avoid memory issues"""
        results = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Translating with {self.name}"):
            batch = texts[i:i+batch_size]
            translations = self.translate(batch)
            results.extend(translations)
        return results

class MarianNMTModel(TranslationModel):
    """Helsinki-NLP's MarianMT model for translation"""
    def __init__(self, model_name="maml_ckpt2/"):
        super().__init__(name="MarianNMT")
        print(f"Loading {model_name}...")
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Model loaded on {self.device}")
    
    def translate(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        translated = self.model.generate(**inputs)
        return self.tokenizer.batch_decode(translated, skip_special_tokens=True)

class M2M100Model(TranslationModel):
    """Facebook's M2M100 model for translation"""
    def __init__(self, model_name="facebook/m2m100_418M"):
        super().__init__(name="M2M100")
        print(f"Loading {model_name}...")
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Model loaded on {self.device}")
    
    def translate(self, texts):
        # Set source language to English
        self.tokenizer.src_lang = "en"
        # Encode the texts
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        # Set target language to Spanish
        self.tokenizer.tgt_lang = "es"
        # Force the decoder to generate Spanish by setting the forced_bos_token
        translated = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.get_lang_id("es")
        )
        return self.tokenizer.batch_decode(translated, skip_special_tokens=True)

class NLLBModel(TranslationModel):
    """Meta's No Language Left Behind model for translation"""
    def __init__(self, model_name="facebook/nllb-200-distilled-600M"):
        super().__init__(name="NLLB")
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Model loaded on {self.device}")
    
    def translate(self, texts):
        # Set source language to English (Latin script)
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        # Force the decoder to generate Spanish (Latin script)
        translated = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.encode("spa_Latn")[0]
        )
        return self.tokenizer.batch_decode(translated, skip_special_tokens=True)

def evaluate_translations(references, hypotheses, source_texts=None):
    """
    Evaluate translations using multiple metrics
    
    Args:
        references: List of reference translations
        hypotheses: List of model translations
        source_texts: List of source texts (optional, for some metrics)
        
    Returns:
        Dictionary with evaluation scores
    """
    results = {}
    
    # Download NLTK resources if needed
    try:
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    
    # BLEU score (0-100)
    bleu = BLEU()
    bleu_score = bleu.corpus_score(hypotheses, [references]).score
    results['bleu'] = bleu_score  # Already in 0-100 range
    
    # CHRF score (0-100)
    chrf = CHRF()
    chrf_score = chrf.corpus_score(hypotheses, [references]).score
    results['chrf'] = chrf_score  # Already in 0-100 range
    
    # METEOR score (0-1)
    max_meteor_samples = min(len(references), 500)  # Limit to 500 samples for speed
    meteor_scores = []
    for i in range(max_meteor_samples):
        ref_tokens = references[i].split()
        hyp_tokens = hypotheses[i].split()
        meteor_scores.append(meteor_score([ref_tokens], hyp_tokens))
    results['meteor'] = np.mean(meteor_scores)
    
    # BERTScore (semantic similarity) (-1 to 1, rescaled with baseline)
    print("Computing BERTScore (this may take a while)...")
    scorer = BERTScorer(lang="es", rescale_with_baseline=True)
    P, R, F1 = scorer.score(hypotheses, references)
    results['bertscore_f1'] = F1.mean().item()
    
    return results

def print_metric_info(metric, value):
    """Helper function to print metric information with ranges"""
    ranges = {
        'bleu': '0-100 (>30: good, >50: excellent)',
        'chrf': '0-100 (>50: good, >70: excellent)',
        'meteor': '0-1 (>0.4: good, >0.6: excellent)',
        'bertscore_f1': '0-1 (>0.6: good, >0.8: excellent)',
        'time_taken': 'seconds'
    }
    
    if metric == 'model':
        print(f"  {metric}: {value}")  # Don't format strings with .4f
    elif metric in ranges:
        print(f"  {metric}: {value:.4f} [Range: {ranges[metric]}]")
    else:
        print(f"  {metric}: {value:.4f}")

def main():
    # Load translations using the configured path and sample limit
    df = load_translations(CSV_PATH, max_samples=MAX_SAMPLES)
    source_texts = df['source'].tolist()
    reference_translations = df['target'].tolist()
    
    # Initialize models
    models = [
        MarianNMTModel(),
        #M2M100Model(),
        #NLLBModel()
    ]
    
    # Store results
    all_results = []
    
    # Evaluate each model
    for model in models:
        print(f"\nEvaluating {model.name}...")
        print("-" * 50)
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
        
        all_results.append(results)
    
    # Save results to CSV
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\nResults saved to {OUTPUT_FILE}")
    
    # Print comparison with ranges
    print("\nModel Comparison:")
    print("=" * 50)
    metrics = ['bleu', 'chrf', 'meteor', 'bertscore_f1']
    for metric in metrics:
        print(f"\n{metric.upper()}:")
        for result in all_results:
            print_metric_info('model', result['model'])
            print_metric_info(metric, result[metric])

if __name__ == "__main__":
    main() 
