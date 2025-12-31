import os
import random
import torch
import math
import pandas as pd
import higher
import torch.nn.functional as F
from tqdm import tqdm
import learn2learn as l2l
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
# Ensure deterministic domain ordering
domains = ['elrc', 'neulab', 'pubmed', 'automobile']

# Hyperparams
k_s = 4
k_q = 4  # only used for build_samples batch size

# Optimizer
student_optim = AdamW(student_logits.parameters(), lr=meta_lr)
ce_loss_fn = nn.CrossEntropyLoss(ignore_index=student_tokenizer.pad_token_id)
scheduler = CosineAnnealingLR(student_optim, T_max=max_epochs, eta_min=1e-6)

# Distillation settings
T_max = distill_config.get('temperature', 2.0)
T_min = 1.0
alpha_max = distill_config.get('alpha', 0.5)
ramp_up_steps = distill_config.get('ramp_up_steps', 1000)

print("Warming up student with Adaptive KD...")
best_bleu = 0.0

for step in tqdm(range(5000), desc="Warmup steps"):
    # Build a mixed-domain batch
    batch = build_samples(datasets_train, k_s + k_q, domains)
    src_texts, tgt_texts, domain_ids = zip(*batch[:batch_size])

    # Tokenize
    inputs = student_tokenizer(src_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    labels = student_tokenizer(tgt_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)

    # Expert logits (detached)
    with torch.no_grad():
        expert_outs = []
        for i, d_idx in enumerate(domain_ids):
            out = experts[d_idx][-1](
                input_ids=inputs.input_ids[i].unsqueeze(0),
                attention_mask=inputs.attention_mask[i].unsqueeze(0),
                decoder_input_ids=labels.input_ids[i].unsqueeze(0)
            )
            expert_outs.append(out)
        expert_logits = torch.cat(expert_outs, dim=0)  # [B, T, V]

    # Student logits
    student_outputs = student_model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        decoder_input_ids=labels.input_ids,
        return_dict=True
    )
    logits = student_outputs.logits  # [B, T, V]

    # Shift for next-token prediction
    pred = logits[:, :-1, :].contiguous()
    targ = labels.input_ids[:, 1:].contiguous()

    # 1) Cross-entropy loss
    ce = ce_loss_fn(pred.view(-1, pred.size(-1)), targ.view(-1))

        # 2) Distillation KL
    s_logp = F.log_softmax(pred / T_max, dim=-1)
    t_prob = F.softmax(expert_logits[:, :-1, :] / T_max, dim=-1)
    kl = F.kl_div(
        s_logp,
        t_prob,
        reduction='batchmean'
    ) * (T_max**2)

    # 3) Adaptive KD weight schedule
    alpha_t = alpha_max * min(1.0, step / ramp_up_steps)

    # 4) Combine losses
    loss = ce + alpha_t * kl 
    loss.backward()
    student_optim.step()
    student_optim.zero_grad()

    # 4) Eval
    if step % 250 == 0:
        bleu, per_domain = evaluate_student_bleu(datasets_test, student_model, student_tokenizer, device)
        if bleu > best_bleu:
            best_bleu = bleu
            # torch.save(student_logits.state_dict(), 'best_warmup_kd_alternate.pt')
        print(f"Step {step}: CE={ce.item():.3f} | KL={kl.item():.3f} | BLEU={bleu:.2f} | PER_DOMAIN_BLEU={per_domain}")

# End of warmup
print(f"Best warmup BLEU: {best_bleu:.2f}")

# Hyperparameters
num_epochs = max_epochs
accumulation_steps = 4  # For larger effective batch sizes

# Optimizer
optimizer = AdamW(student_logits.parameters(), lr=meta_lr)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

best_bleu = 0
no_improve = 0

for epoch in range(num_epochs):
    student_logits.train()
    total_loss = 0
    
    # Build full training batches with domain info
    batches = []
    for domain_idx, domain in enumerate(domains):
        domain_samples = build_samples(datasets_train, k_s+k_q, [domain])
        batches.extend([(s[0], s[1], domain_idx) for s in domain_samples])
    
    random.shuffle(batches)
    
    optimizer.zero_grad()
    
    for i in tqdm(range(0, len(batches), batch_size)):
        batch = batches[i:i+batch_size]
        src_texts, tgt_texts, domain_ids = zip(*batch)
        
        # Prepare inputs
        inputs = student_tokenizer(list(src_texts), return_tensors='pt', padding=True, truncation=True).to(device)
        labels = student_tokenizer(list(tgt_texts), return_tensors='pt', padding=True, truncation=True).to(device)
        
        # Get expert outputs
        with torch.no_grad():
            expert_logits = []
            for i, domain_idx in enumerate(domain_ids):
                expert_out = experts[domain_idx][-1](
                    input_ids=inputs.input_ids[i].unsqueeze(0),
                    attention_mask=inputs.attention_mask[i].unsqueeze(0),
                    decoder_input_ids=labels.input_ids[i].unsqueeze(0)
                )
                expert_logits.append(expert_out)
            expert_logits = torch.cat(expert_logits, dim=0)
        
        # Forward pass
        student_out = student_logits(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            decoder_input_ids=labels.input_ids
        )
        
        # Loss computation
        ce_loss = ce(student_out.view(-1, student_out.size(-1)), labels.input_ids.view(-1))
        logit_loss = F.kl_div(
            F.log_softmax(student_out, dim=-1),
            F.softmax(expert_logits, dim=-1),
            reduction='batchmean'
        )
        loss = (ce_loss + logit_loss) / accumulation_steps
        
        loss.backward()
        total_loss += loss.item()
        
        if (i // batch_size + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(student_logits.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
    
    scheduler.step()
    
    # Evaluation
    print(f"Epoch {epoch}: Loss={total_loss:.4f}")
    overall_bleu, per_domain = evaluate_student_bleu(
        datasets_test, student_model, student_tokenizer, device
    )

    print(overall_bleu)
    print(per_domain)
    
    # Save best model
    if overall_bleu > best_bleu:
        best_bleu = overall_bleu
        no_improve = 0
        #torch.save(student_logits.state_dict(), "best_student.pt")
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping")
            break
