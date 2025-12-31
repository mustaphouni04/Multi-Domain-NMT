import os
import random
import torch
import math
import pandas as pd
import higher
from io import BytesIO
from tqdm import tqdm
import learn2learn as l2l
from learn2learn.algorithms.meta_sgd import meta_sgd_update
import argparse

import learn2learn.algorithms.meta_sgd as _ms

def simple_adapt(self, loss, first_order=None, retain_graph=False):
    # Compute gradients w.r.t. the *underlying* module's parameters
    grads = torch.autograd.grad(
        loss,
        list(self.module.parameters()),
        create_graph=not self.first_order,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    # In-place update: p <- p - lr * grad
    for p, g, lr in zip(self.module.parameters(), grads, self.lrs):
        if g is not None:
            p.sub_(lr * g)
            #p.data = p.data - lr * g
    return self

_ms.MetaSGD.adapt = simple_adapt

from learn2learn.algorithms.meta_sgd import MetaSGD
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
                      evaluate_student_bleu,
                      get_ood_query,
                      train_step_ood,
                     )
import importlib
import yaml

torch.autograd.set_detect_anomaly(True)
parser = argparse.ArgumentParser()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser.add_argument("--warmup", type=str2bool, nargs='?', const=True, help = "Whether to warmup the model or proceed with meta-learning in OOD.", required=True)

def main(args: int):

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
    ood_path       = data["ood"]
    ood_test       = data["ood_test"]

    experts, num_experts = load_and_freeze_experts(models_path, device)
    # student_logits: MarianLogits object, student_feat_ext: MarianMAMLFeatures
    student_model, student_tokenizer, student_feat_ext, student_logits = load_student(student_path, device)

    for p in student_model.parameters():
        p.requires_grad = True

    for p in student_logits.model.parameters():
        p.requires_grad = True

    datasets_train, datasets_test, domains, test_samples = load_and_split_data(splits)
    #domains = ['elrc', 'neulab', 'pubmed', 'automobile']

    # --- 3) Instantiate Aggregator + optimizer + scheduler + loss ---

    hid = infer_hidden_dim(experts, batch_size, device)
        
    aggregator = Aggregator(
        hidden_dim=hid,
        num_experts=num_experts,
        depth=4, heads=8, mlp_dim=hid*4, dropout=0.2
    ).to(device)

    for p in aggregator.parameters():
        p.requires_grad = True

    maml_stu = MetaSGD(student_logits, lr=inner_lr, first_order=True)
    num_params = sum(1 for _ in student_logits.model.parameters())
    num_lrs    = len(maml_stu.lrs)
    print("student parameters:", num_params, "learning rates:", num_lrs)

    optimizer = AdamW(
        aggregator.parameters(),
        lr=1e-4,
        weight_decay=1e-2
    )

    criterion = nn.CrossEntropyLoss()

    # Hyperparams:
    meta_batch_size = len(domains)   # one "task" per domain
    k_s = 7   # support examples per domain
    k_q = 7   # query  examples per domain
    inner_steps = 2

    meta_optimizer = torch.optim.AdamW(
        list(student_logits.parameters()), 
        lr=meta_lr
    )
    ce = nn.CrossEntropyLoss()
    
    if args == 1:
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

        for p in aggregator.parameters():
            p.requires_grad = False

        print("Warming up student...")
        student_optim = AdamW(student_logits.parameters(), lr=meta_lr) # meta_lr
        auto = float('-inf')
        steps = 0
        for step in tqdm(range(5250)):  # 1.5k warmup steps
            if steps >= patience:
                break

            if step % 50 == 0:
                overall_bleu, per_domain_bleu, ood_bleu = evaluate_student_bleu(
                    datasets_test, student_model, student_tokenizer, device, ood_test
                )
                print(f"Overall student BLEU: {overall_bleu:.2f}")
                for dom, score in per_domain_bleu.items():
                    print(f"  {dom:12s}: {score:.2f}")
                    if per_domain_bleu['automobile'] >= auto:
                        auto = per_domain_bleu['automobile']
                        #student_model.save_pretrained("ckpt2/")
                        #student_tokenizer.save_pretrained("ckpt2/")
                        #torch.save(student_model.state_dict(), "ckpt.pt")
                    else:
                        steps += 1
                print(f"BLEU for OOD: {ood_bleu:.2f}")

            #student_model.train()
            batch = build_samples(datasets_train, k_s+k_q, domains)[:batch_size]
            loss = train_step_query(student_logits, student_tokenizer, batch,
                                    experts, distill_config, device)
            loss.backward()
            student_optim.step()
            student_optim.zero_grad()
    
    # usage: get_ood_query(ood_path, tasks = 4, k_q = k_q)
    track = float('-inf') 
    for meta_epoch in tqdm(range(max_epochs), desc="Training..."):
        meta_optimizer.zero_grad()
        meta_tasks = build_meta_batch(datasets_train, domains, k_s, k_q)
        meta_loss_total = 0.0

        for support, in_batch, _ in meta_tasks:
            query = get_ood_query(ood_path, tasks=4, k_q=k_q)

            # Clone both aggregator and student for inner-loop adaptation
            student_clone = maml_stu.clone()

            # 1. Inner Loop Adaptation
            for inner_step in range(inner_steps):
                # Compute combined loss: aggregator + distillation
                loss_distill = train_step_query(
                    student_clone,  
                    student_tokenizer, 
                    support,
                    experts,
                    distill_config,
                    device,
                )

                student_clone.adapt(loss_distill, first_order=True, retain_graph=True)


            # 2. Outer Loop: Query Loss 
            query_loss = train_step_ood(student_clone, 
                                        student_tokenizer, 
                                        query,
                                        device)

            in_loss = train_step_query(student_clone,
                                       student_tokenizer,
                                       in_batch,
                                       experts,
                                       distill_config,
                                       device)

            meta_loss_total += 0.9*query_loss + 0.1*in_loss
        
        # 3. Meta-Update: Backprop through adaptation steps
        meta_loss_total.backward()

        # 4. Gradient Monitoring & Clipping
        stu_grads = [p.grad for p in student_logits.parameters() if p.grad is not None]

        if stu_grads:
            stu_norm = torch.norm(torch.stack([g.norm() for g in stu_grads]))
        else:
            stu_norm = torch.tensor(0.0)

        print(f"Student={stu_norm:.4f}")

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(student_logits.parameters(), 1.0)

        # Meta-parameter update
        meta_optimizer.step()

        # --- Evaluation & Logging ---
        print(f"Meta Epoch {meta_epoch}: Meta-Loss {meta_loss_total.item():.4f}")
        
        if meta_epoch % 5 == 0:
            # To keep track of catastrophic forgetting
            print("⇨ Evaluating student BLEU on all domains …")
            overall_bleu, per_domain_bleu, ood_bleu = evaluate_student_bleu(
                datasets_test, student_model, student_tokenizer, device, ood_test
            )
            print(f"Overall student BLEU on all in-domains: {overall_bleu:.2f}")
            for dom, score in per_domain_bleu.items():
                print(f"  {dom:12s}: {score:.2f}")
            
            print(f"BLEU for OOD: {ood_bleu:.2f}")

            if ood_bleu > track:
                track = ood_bleu
                student_model.save_pretrained("maml_ckpt2/")
                student_tokenizer.save_pretrained("maml_ckpt2/")


if __name__ == '__main__':
    args = parser.parse_args() 
    main(args.warmup)
    
