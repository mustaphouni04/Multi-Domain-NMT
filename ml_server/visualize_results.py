"""
Visualization Script for Translation Model Evaluation Results

This script creates visualizations of the translation model evaluation results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
INPUT_CSV = r"results/merged_results.csv"


OUTPUT_DIR = "results/merged_charts"  # Directory to save the charts

def load_results(csv_path):
    """Load results from CSV file"""
    df = pd.read_csv(csv_path)
    return df

def create_bar_chart(df, metrics, title, output_file=None):
    """Create a bar chart comparing models across metrics"""
    models = df['model'].tolist()
    n_models = len(models)
    n_metrics = len(metrics)
    
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Set width of bars
    bar_width = 0.8 / n_metrics
    
    # Set positions of bars on X axis
    r = np.arange(n_models)
    
    # Create bars
    for i, metric in enumerate(metrics):
        values = df[metric].tolist()
        plt.bar(r + i * bar_width, values, width=bar_width, label=metric.upper())
    
    # Add labels and title
    plt.xlabel('Models', fontweight='bold', fontsize=12)
    plt.ylabel('Score', fontweight='bold', fontsize=12)
    plt.title(title, fontweight='bold', fontsize=14)
    
    # Add xticks on the middle of the group bars
    plt.xticks(r + bar_width * (n_metrics - 1) / 2, models)
    
    # Create legend
    plt.legend()
    
    # Add value labels on top of bars
    for i, metric in enumerate(metrics):
        values = df[metric].tolist()
        for j, value in enumerate(values):
            plt.text(r[j] + i * bar_width, value + 0.01, f'{value:.3f}', 
                     ha='center', va='bottom', fontsize=9, rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the figure
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {output_file}")
    else:
        plt.show()

def create_radar_chart(df, metrics, title, output_file=None):
    """Create a radar chart comparing models across metrics"""
    models = df['model'].tolist()
    n_models = len(models)
    n_metrics = len(metrics)
    
    # Set up the figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Compute angle for each metric
    angles = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Get values for each model
    for i, model in enumerate(models):
        values = df.loc[df['model'] == model, metrics].values.flatten().tolist()
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Set labels for each metric
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.upper() for m in metrics])
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Add title
    plt.title(title, fontweight='bold', fontsize=14)
    
    # Save or show the figure
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {output_file}")
    else:
        plt.show()

def main():
    # Load results using the configured path
    df = load_results(INPUT_CSV)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Define metrics to visualize
    grammatical_metrics = ['bleu', 'chrf', 'meteor']
    semantic_metrics = ['bertscore_f1']  # We only save F1 from BERTScore
    all_metrics = grammatical_metrics + semantic_metrics
    
    # Create bar charts
    create_bar_chart(
        df, 
        all_metrics, 
        'Translation Model Comparison - All Metrics',
        os.path.join(OUTPUT_DIR, 'all_metrics_bar.png')
    )
    
    create_bar_chart(
        df, 
        grammatical_metrics, 
        'Translation Model Comparison - Grammatical Metrics',
        os.path.join(OUTPUT_DIR, 'grammatical_metrics_bar.png')
    )
    
    # Only create semantic metrics chart if we have more than one metric
    if len(semantic_metrics) > 1:
        create_bar_chart(
            df, 
            semantic_metrics, 
            'Translation Model Comparison - Semantic Metrics',
            os.path.join(OUTPUT_DIR, 'semantic_metrics_bar.png')
        )
    
    # Create radar chart
    create_radar_chart(
        df, 
        all_metrics, 
        'Translation Model Comparison - Radar Chart',
        os.path.join(OUTPUT_DIR, 'radar_chart.png')
    )
    
    print(f"\nResults loaded from: {INPUT_CSV}")
    print(f"Charts saved to directory: {OUTPUT_DIR}")
    print("All visualizations created successfully!")

if __name__ == "__main__":
    main() 