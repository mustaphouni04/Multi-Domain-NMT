# Translation Model Evaluation

This project evaluates the performance of three different open-source translation models for English to Spanish translation using a CSV file containing reference translations.

## Models Evaluated

1. **MarianNMT** (Helsinki-NLP/opus-mt-en-es) - A specialized English to Spanish translation model
2. **M2M100** (facebook/m2m100_418M) - Facebook's multilingual translation model
3. **NLLB** (facebook/nllb-200-distilled-600M) - Meta's No Language Left Behind model

## Evaluation Metrics

The script evaluates translations using the following metrics:

- **BLEU** - Measures n-gram precision with brevity penalty
- **CHRF** - Character n-gram F-score
- **METEOR** - Metric for Evaluation of Translation with Explicit Ordering
- **BERTScore** - Semantic similarity using BERT embeddings

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Evaluation Script

```bash
python translation_model_evaluation.py --csv path/to/your/translations.csv
```

#### Command Line Arguments

- `--csv`: Path to CSV file with translations (required)
- `--output`: Path to output CSV file (default: translation_evaluation_results.csv)
- `--samples`: Maximum number of samples to evaluate (default: all)
- `--batch-size`: Batch size for translation (default: 8)

### Visualization Script

After running the evaluation, you can visualize the results:

```bash
python visualize_results.py --csv translation_evaluation_results.csv
```

#### Command Line Arguments

- `--csv`: Path to CSV file with evaluation results (required)
- `--output-dir`: Directory to save charts (default: charts)

### CSV Format

The CSV file should contain at least two columns:
- `source`: The source text in English
- `target`: The reference translation in Spanish

## Example

```bash
# Run evaluation
python translation_model_evaluation.py --csv translations.csv --samples 1000 --batch-size 16

# Visualize results
python visualize_results.py --csv translation_evaluation_results.csv
```

## Output

The evaluation script will:
1. Print evaluation results for each model
2. Show example translations
3. Save detailed results to a CSV file
4. Print a comparison of models across different metrics

The visualization script will generate:
1. Bar charts comparing models across different metrics
2. A radar chart showing the performance of each model across all metrics
3. Separate charts for grammatical and semantic metrics
