import os
import random
import torch
import math
import pandas as pd
import higher
from tqdm import tqdm
import learn2learn as l2l
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.optim import AdamW
from _modules import (
                      train_step,
                      eval_epoch_aggregator,
                      load_and_freeze_experts, 
                      load_and_split_data, 
                      infer_hidden_dim, 
                      load_student,
                      build_meta_batch,     
                      build_samples,
                      train_step_query,
                      evaluate_student_bleu
                     )
import importlib
import yaml
torch.autograd.set_detect_anomaly(True)

# --- Import Aggregator dynamically ---
module_path = "Meta-DMoE.src.transformer"
transformer_module = importlib.import_module(module_path)
Aggregator = getattr(transformer_module, "Aggregator")

# --- Config ---
with open('config.yml', 'r') as f:
    data = yaml.load(f, Loader=yaml.SafeLoader)

models_path    = data["models_path"]
splits         = data["splits"]
batch_size     = data["batch_size"]
max_epochs     = data["max_epochs"]
device         = data["device"]
patience       = data["patience"]
student_path   = data["student_model_path"]
distill_config = data["distill"]
inner_lr       = float(data["inner_lr"])
meta_lr        = float(data["meta_lr"])

experts, num_experts = load_and_freeze_experts(models_path, device)
# student_logits: MarianLogits object, student_feat_ext: MarianMAMLFeatures
student_model, student_tokenizer, student_feat_ext, student_logits = load_student(student_path, device)
for p in student_model.parameters():
    p.requires_grad = True

for p in student_logits.model.parameters():
    p.requires_grad = True

datasets_train, datasets_test, domains, test_samples = load_and_split_data(splits)

# --- 3) Instantiate Aggregator + optimizer + scheduler + loss ---

hid = infer_hidden_dim(experts, batch_size, device)
    
aggregator = Aggregator(
    hidden_dim=hid,
    num_experts=num_experts,
    depth=4, heads=8, mlp_dim=hid*4, dropout=0.2
).to(device)

maml_agg = l2l.algorithms.MAML(aggregator, lr=inner_lr, first_order=True)
maml_stu = l2l.algorithms.MAML(student_logits, lr=inner_lr, first_order=True)

optimizer = AdamW(
    aggregator.parameters(),
    lr=1e-4,
    weight_decay=1e-2
)

criterion = nn.CrossEntropyLoss()

# Hyperparams:
meta_batch_size = len(domains)   # one “task” per domain
k_s = 4   # support examples per domain
k_q = 4   # query  examples per domain
inner_steps = 1

meta_optimizer = torch.optim.AdamW(
    list(aggregator.parameters()) + list(student_logits.parameters()), 
    lr=meta_lr
)
ce = nn.CrossEntropyLoss()

scheduler = CosineAnnealingLR(
    meta_optimizer, 
    T_max = max_epochs,
    eta_min = 1e-6
)

print("Warming up student...")
student_optim = AdamW(student_logits.parameters(), lr=meta_lr)
for _ in tqdm(range(1500)):  # 1.5k warmup steps
    batch = build_samples(datasets_train, k_s+k_q, domains)[:batch_size]
    loss = train_step_query(student_logits, student_tokenizer, batch, device)
    loss.backward()
    student_optim.step()
    student_optim.zero_grad()

print("Warming up Aggregator...")
agg_optim = AdamW(aggregator.parameters(), lr=1e-4)

domain_weights = torch.Tensor([
    1/0.6263,
    1/0.1037,
    1/0.5464,
    1/0.7008
]).to(device)
criterion = nn.CrossEntropyLoss(weight=domain_weights)

for warmup_epoch in tqdm(range(20)):
    total_loss = 0.0
    domains_shuffled = domains.copy()
    random.shuffle(domains_shuffled)

    for domain in domains_shuffled + ['elrc']:
        domain_idx = domains.index(domain) if domain != 'elrc' else domains.index('elrc')

        k_per_domain = 10 if domain == 'elrc' else 8
        batch = build_samples(datasets_train, k_per_domain=k_per_domain, domains=[domain])
        batch = [(src, tgt, domain_idx) for src, tgt, _ in batch]

        agg_loss, _ = train_step(
                aggregator,
                student_logits,
                student_tokenizer,
                experts,
                batch,
                distill_config,
                device,
                criterion,
                only_agg=True
        )
        agg_loss.backward()
        agg_optim.step()
        agg_optim.zero_grad()
        total_loss += agg_loss.item()
    
    print(f"Aggregator Warmup Epoch {warmup_epoch}: Loss={total_loss:.3f}")

# --- After aggregator warmup, freeze it ---
print("Freezing Aggregator after warmup...")
aggregator.eval()
for param in aggregator.parameters():
    param.requires_grad = False

meta_optimizer = torch.optim.AdamW(
    list(student_logits.parameters()),  # Only student parameters
    lr=meta_lr
)

   # Main training loop with MAML adaptation for student only
for meta_epoch in range(max_epochs):
    meta_optimizer.zero_grad()
    meta_tasks = build_meta_batch(datasets_train, domains, k_s, k_q)
    meta_loss = 0.0
    
    # Only adapt STUDENT in inner loop
    for support, query, _ in meta_tasks:
        # Create differentiable student clone for this task
        with higher.innerloop_ctx(
            student_logits,
            torch.optim.SGD(student_logits.parameters(), lr=inner_lr),
            copy_initial_weights=False,
            track_higher_grads=True
        ) as (fstu, fstu_opt):
            
            # Inner loop adaptation (using FROZEN aggregator)
            for inner_step in range(inner_steps):
                # Use FROZEN aggregator + adapted student
                _, distill_loss = train_step(
                    aggregator,  # Frozen!
                    fstu,        # Adapted student
                    student_tokenizer,
                    experts,
                    support,
                    distill_config,
                    device,
                    criterion,
                    only_agg=False
                )
                fstu_opt.step(distill_loss)
            
            # Compute query loss with adapted student
            query_loss = train_step_query(
                fstu,
                student_tokenizer,
                query,
                device
            )
            meta_loss += query_loss / len(meta_tasks)

    # Outer loop update (student only)
    meta_loss.backward()
    
    # Monitor gradients (student only now)
    stu_grads = [p.grad for p in student_logits.parameters() if p.grad is not None]
    stu_grad_norm = torch.norm(torch.stack([g.norm() for g in stu_grads])) if stu_grads else torch.tensor(0.0)
    print(f"Student Gradient Norm: {stu_grad_norm:.4f}")
    
    # Clip and update
    torch.nn.utils.clip_grad_norm_(student_logits.parameters(), 1.0)
    meta_optimizer.step() 

    if meta_epoch == 0:
        test_acc, improved = eval_epoch_aggregator(
            aggregator,
            experts,
            test_samples[:1000],
            domains,
            batch_size,
            scheduler,
            patience,
            device,
            meta_epoch
        )
        if improved:
            #torch.save(aggregator.state_dict(), "best_aggregator.pt")
            print(f"  ↳ New best model saved (Acc={test_acc*100:.2f}%)")

        if eval_epoch_aggregator.no_improve >= patience:
            print("Early stopping triggered.")
            break

    print(f"Meta Epoch {meta_epoch}: Meta‐Loss {meta_loss.item():.4f}")

    print("⇨ Evaluating student BLEU on all test domains …")
    overall_bleu, per_domain_bleu = evaluate_student_bleu(
            datasets_test,
            student_model,
            student_tokenizer,
            device,
            max_length=512
    )

    print(f"Overall student BLEU: {overall_bleu:.2f}")
    for dom, score in per_domain_bleu.items():
        print(f"  {dom:12s}: {score:.2f}")

