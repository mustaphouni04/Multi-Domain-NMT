import os
import random
import torch
import math
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from _modules import (
    load_and_freeze_experts, 
    load_and_split_data, 
    load_student,
    build_samples,
    evaluate_student_bleu
)
import yaml

torch.autograd.set_detect_anomaly(True)

def pad_sequences(sequences, pad_token_id):
    # Pad a list of 1D tensors to the same length
    max_len = max(seq.size(1) for seq in sequences)
    padded = []
    for seq in sequences:
        pad_len = max_len - seq.size(1)
        if pad_len > 0:
            pad = torch.full((seq.size(0), pad_len), pad_token_id, dtype=seq.dtype, device=seq.device)
            padded_seq = torch.cat([seq, pad], dim=1)
        else:
            padded_seq = seq
        padded.append(padded_seq)
    return torch.cat(padded, dim=0)

# --- Config ---
with open('config.yml', 'r') as f:
    data = yaml.load(f, Loader=yaml.SafeLoader)

models_path    = data['models_path']
splits         = data['splits']
batch_size     = data['batch_size']
max_epochs     = data['max_epochs']
device         = data['device']
patience       = data['patience']
student_path   = data['student_model_path']
distill_config = data['distill']
inner_lr       = float(data['inner_lr'])
meta_lr        = float(data['meta_lr'])

# Load experts and student
experts, num_experts = load_and_freeze_experts(models_path, device)
student_model, student_tokenizer, _, student_logits = load_student(student_path, device)

# Enable gradients for student
for p in student_model.parameters(): p.requires_grad = True
for p in student_logits.model.parameters(): p.requires_grad = True

# Data
datasets_train, datasets_test, domains, _ = load_and_split_data(splits)
domains = ['elrc', 'neulab', 'pubmed', 'automobile']

# Hyperparams
k_s = 8
k_q = 8  # only used for build_samples batch size

# Optimizer
student_optim = AdamW(student_logits.parameters(), lr=meta_lr)
ce_loss_fn = nn.CrossEntropyLoss(ignore_index=student_tokenizer.pad_token_id)
scheduler = CosineAnnealingLR(student_optim, T_max=max_epochs, eta_min=1e-6)

# Generation settings for pseudo-targets
num_beams = distill_config.get('num_beams', 5)
max_gen_len = distill_config.get('max_gen_len', 128) # was 128

print("Warming up student with sequence-level distillation...")
best_bleu = 0.0

for step in tqdm(range(5000), desc="Sequence KD steps"):
    # Build a mixed-domain batch
    batch = build_samples(datasets_train, k_s + k_q, domains)
    src_texts, _, domain_ids = zip(*batch[:batch_size])

    # Tokenize inputs
    inputs = student_tokenizer(src_texts, return_tensors='pt', padding=True,
                                truncation=True, max_length=512).to(device)

    # 1) Generate pseudo-targets from each domain expert
    pseudo_seqs = []
    for i, d_idx in enumerate(domain_ids):
        model = experts[d_idx][-1]
        out_ids = model.model.generate(
            input_ids=inputs.input_ids[i].unsqueeze(0),
            attention_mask=inputs.attention_mask[i].unsqueeze(0),
            max_length=max_gen_len,
            num_beams=num_beams,
            early_stopping=True
        )  # [1, L]
        pseudo_seqs.append(out_ids)
    # Pad generated sequences to batch
    gen_ids = pad_sequences(pseudo_seqs, student_tokenizer.pad_token_id)  # [B, L]

    # 2) Student forward with pseudo targets
    student_outputs = student_model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        decoder_input_ids=gen_ids,
        return_dict=True
    )
    logits = student_outputs.logits  # [B, L, V]

    # 3) Compute CE against pseudo-targets shifted
    pred = logits[:, :-1, :].contiguous()
    targ = gen_ids[:, 1:].contiguous()
    loss = ce_loss_fn(pred.view(-1, pred.size(-1)), targ.view(-1))

    # 4) Backprop and step
    loss.backward()
    student_optim.step()
    student_optim.zero_grad()

    # 5) Evaluate occasionally
    if step % 250 == 0:
        bleu, per_domain = evaluate_student_bleu(datasets_test, student_model, student_tokenizer, device)
        if bleu > best_bleu:
            best_bleu = bleu
            # torch.save(student_logits.state_dict(), 'best_seq_kd.pt')
        print(f"Step {step}: Loss={loss.item():.3f} | BLEU={bleu:.2f} | PER_DOMAIN_BLEU={per_domain}")

print(f"Best seq-KD BLEU: {best_bleu:.2f}")
