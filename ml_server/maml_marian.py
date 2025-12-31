from transformers import MarianMTModel, MarianTokenizer
import torch as th
import learn2learn as l2l
from torch import nn, optim
from pathlib import Path
from domain_utils import DomainProcessing
from typing import Tuple, Dict, List, Union, Optional
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import os
import wandb

device = "cuda" if th.cuda.is_available() else "cpu"
wandb.require("core")

wandb.login()

model_name = "Helsinki-NLP/opus-mt-en-es"
tokenizer = MarianTokenizer.from_pretrained(model_name)
base_model = MarianMTModel.from_pretrained(model_name).to(device)

class MarianMAMLWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
        )
        return outputs.loss

def support_query_split(df: pd.DataFrame, Sd: int, Qd:int, dataset_type: str):
    data = {}

    if "train" in df:
        df = df["train"]
    else:
        pass
    
    cols = ["sentence_en", "sentence_es", "source", "target"]
    
    if dataset_type == "Dataset":
        column_names = df.column_names if hasattr(df, 'column_names') else df.columns
        ls = [col for col in cols if col in column_names]
        if ls == []:
            en, es = df["en"], df["es"]
        else:
            en, es = df[ls[0]], df[ls[1]]
    else:
        en, es = df["en"], df["es"]
    
    support_pairs = [(eng_sample, esp_sample) for eng_sample, esp_sample in zip(en[:Sd], es[:Sd])]
    query_pairs = [(eng_sample, esp_sample) for eng_sample, esp_sample in zip(en[Sd:Qd+Sd], es[Sd:Qd+Sd])]
    data["support"] = support_pairs
    data["query"] = query_pairs
    return data

def translation_pairs( 
        dataset_type: str, 
        path: str,
        Sd: int,
        Qd: int,
        limit: Optional[int] = None) -> Dict[str,List[Tuple[str,str]]]:
   
    if dataset_type == "Dataset":
        data = {}
        dataset = DomainProcessing(path)
        df = dataset.process()
        data = support_query_split(df,Sd,Qd,dataset_type)
        return data

    elif dataset_type == "TM":
        tms = os.listdir(path)
        paths = []
        for tm in tqdm(tms[:limit], desc="Looking for TMs"):
            if tm.endswith(".tmx"):
                path = Path(path)
                q = path / str(tm)
                #print(q)
                paths.append(str(q))
        try:
            dataset = DomainProcessing(paths)
            df = dataset.process()
                
            data = support_query_split(df,Sd,Qd,dataset_type)
            return data
        except Exception as e:
            print(f"Error: {e}")

def tokenize(pairs: List[Tuple[str,str]]) -> Tuple[Dict[str, th.Tensor], Dict[str, th.Tensor]]:
    source_texts = [en for en,es in pairs]
    target_texts = [es for en,es in pairs]

    source_batch = tokenizer(source_texts, return_tensors="pt", padding=True, truncation=True)
    target_batch = tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True)

    return source_batch, target_batch["input_ids"]

def create_loader(inputs, labels, batch_size):
    dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

tasks = []
Sd = 800
Qd = 200
webcrawl_en_es = translation_pairs(dataset_type="Dataset", 
        path="Thermostatic/parallel_corpus_webcrawl_english_spanish_1",
        Sd = Sd,
        Qd = Qd)
tasks.append(webcrawl_en_es)
financial_phrasebank = translation_pairs(dataset_type="Dataset", 
        path="NickyNicky/financial_phrasebank_traslate_En_Es",
        Sd = Sd,
        Qd = Qd)

tasks.append(financial_phrasebank)
dgt_en_es = translation_pairs(dataset_type="TM", 
        path="Vol_2012_1/",
        Sd = Sd,
        Qd = Qd,
        limit=100)
tasks.append(dgt_en_es)

automobile = translation_pairs(dataset_type="Dataset", 
        path="train_set.csv",
        Sd = Sd,
        Qd = Qd)
tasks.append(automobile)

marianNMT = MarianMAMLWrapper(base_model)

maml = l2l.algorithms.MAML(marianNMT, lr=1e-4, first_order=True)
num_epochs = 5
outer_optimizer = optim.Adam(maml.parameters(), lr=1e-4)
wandb.init(
    project="maml_marian",
    name=f"run",
    config={
        "learning_rate":1e-4,
        "architecture":"MarianMT",
        "dataset":"auto/finance/agreements/legal",
        "epochs":5,
    })

def train(num_epochs: int, outer_optimizer, maml):
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        for task in tqdm(tasks, desc=f"Adapting to different domains (Epoch {epoch+1})"):
            support_inputs, support_labels = tokenize(task["support"])
            query_inputs, query_labels = tokenize(task["query"])

            support_inputs = {k: v.to(device) for k, v in support_inputs.items()}
            #support_labels = {k: v.to(device) for k, v in support_labels.items()}
            support_labels = support_labels.to(device)
            query_inputs = {k: v.to(device) for k, v in query_inputs.items()}
            #query_labels = {k: v.to(device) for k, v in query_labels.items()}
            query_labels = query_labels.to(device)

            support_loader = create_loader(support_inputs, support_labels, batch_size=12)
            learner = maml.clone()

            inner_steps = 3
            for batch_input_ids, batch_attention_mask, batch_labels in tqdm(support_loader):
                batch_inputs = {
                    "input_ids": batch_input_ids.to(device),
                    "attention_mask": batch_attention_mask.to(device),
                }
                batch_labels = batch_labels.to(device)
                for _ in range(inner_steps):
                    support_loss = learner(
                        input_ids=batch_inputs["input_ids"],
                        attention_mask=batch_inputs["attention_mask"],
                        labels=batch_labels,
                    )
                    learner.adapt(support_loss,
                            allow_unused=True,
                            allow_nograd=True,)
                wandb.log({"support_loss":support_loss})

            query_loader = create_loader(query_inputs, query_labels, batch_size=12)
            outer_optimizer.zero_grad()

            for batch_q_ids, batch_q_attn, batch_q_labels in tqdm(query_loader):
                batch_q_ids = batch_q_ids.to(device)
                batch_q_attn = batch_q_attn.to(device)
                batch_q_labels = batch_q_labels.to(device)
                batch_q_loss = learner(
                    input_ids=batch_q_ids,
                    attention_mask=batch_q_attn,
                    labels=batch_q_labels,
                )
                batch_q_loss.backward()
                wandb.log({"meta_loss":batch_q_loss})
            
            th.nn.utils.clip_grad_norm_(maml.parameters(), max_norm=1.0)
            outer_optimizer.step()

    wandb.finish()
# Warm up training
train(num_epochs, outer_optimizer, maml)

save_dir = "./maml_opus_mt_en_es"
hf_model: MarianMTModel = maml.module.model

hf_model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)






