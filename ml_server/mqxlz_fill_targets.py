#!/usr/bin/env python3
"""
Fill Empty Targets in memoQ .mqxlz Using MarianNMT and Output CSV

Usage:
    python mqxlz_fill_targets.py input_file.mqxlz

This script takes a single .mqxlz file, fills empty/missing <target> elements in all .mqxliff files inside
using the marianNMT_automobiles translation model, and outputs a CSV to results/translated_mqxlz_files/
with columns 'id', 'source', 'target' (all missing targets filled).
"""

import zipfile
import xml.etree.ElementTree as ET
import os
import sys
import csv
from pathlib import Path
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

# Model path
MODEL_PATH = r"C:\Users\Miguel\OneDrive\Escritorio\4t curs\second_semester\synthetsis_project_II\marianNMT_automobiles"
BATCH_SIZE = 16
OUTPUT_DIR = Path("results/translated_mqxlz_files/")

class MarianTranslator:
    def __init__(self, model_path=MODEL_PATH):
        print(f"Loading MarianNMT model from {model_path} ...")
        self.tokenizer = MarianTokenizer.from_pretrained(model_path)
        self.model = MarianMTModel.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def translate_batch(self, texts, batch_size=BATCH_SIZE):
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                translated = self.model.generate(**inputs, num_beams=4, max_length=512, early_stopping=True)
            decoded = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
            results.extend(decoded)
        return results

def extract_and_fill_segments(mqxlz_path, translator):
    segments = []
    to_translate = []
    to_translate_indices = []
    with zipfile.ZipFile(mqxlz_path, 'r') as zin:
        mqxliff_files = [f for f in zin.namelist() if f.endswith('.mqxliff')]
        for mqxliff_file in mqxliff_files:
            with zin.open(mqxliff_file) as file:
                content = file.read().decode('utf-8')
                root = ET.fromstring(content)
                trans_units = root.findall('.//trans-unit') or root.findall('.//{*}trans-unit')
                for i, trans_unit in enumerate(trans_units, 1):
                    source_elem = trans_unit.find('.//source') or trans_unit.find('.//{*}source')
                    target_elem = trans_unit.find('.//target') or trans_unit.find('.//{*}target')
                    source_text = (source_elem.text or "") if source_elem is not None else ""
                    target_text = (target_elem.text or "") if target_elem is not None else ""
                    unit_id = trans_unit.get('id', str(i))
                    seg_id = f"{mqxliff_file}_{unit_id}"
                    if source_text.strip() and (target_elem is None or not target_text.strip()):
                        # Needs translation
                        to_translate.append(source_text)
                        to_translate_indices.append(len(segments))
                        segments.append({'id': seg_id, 'source': source_text, 'target': None})
                    else:
                        segments.append({'id': seg_id, 'source': source_text, 'target': target_text})
    # Translate missing targets
    if to_translate:
        print(f"Translating {len(to_translate)} missing targets ...")
        translations = translator.translate_batch(to_translate)
        for idx, translation in zip(to_translate_indices, translations):
            segments[idx]['target'] = translation
    return segments

def write_csv(segments, output_csv_path):
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['id', 'source', 'target'])
        writer.writeheader()
        for seg in segments:
            writer.writerow(seg)
    print(f"✓ CSV written to: {output_csv_path}")

def main():
    if len(sys.argv) != 2 or sys.argv[1] in {"-h", "--help"}:
        print("Usage: python mqxlz_fill_targets.py input_file.mqxlz")
        sys.exit(1)
    input_file = Path(sys.argv[1])
    if not input_file.exists() or not input_file.suffix.lower() == ".mqxlz":
        print(f"Error: {input_file} does not exist or is not a .mqxlz file")
        sys.exit(1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = OUTPUT_DIR / (input_file.stem + ".csv")
    translator = MarianTranslator()
    print(f"Processing {input_file} ...")
    segments = extract_and_fill_segments(input_file, translator)
    write_csv(segments, output_csv)

if __name__ == "__main__":
    main()
