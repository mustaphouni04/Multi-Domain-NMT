from domain_utils import DomainProcessing
from typing import Optional, List, Tuple
from pathlib import Path
import os
import pandas as pd
from tqdm import tqdm

def translation_pairs( 
        dataset_type: str, 
        path: str,
        Sd: Optional[int] = None,
        Qd: Optional[int] = None,
        limit: Optional[int] = None,
        name: Optional[str] = None):
   
    if dataset_type == "Dataset":
        data = {}
        dataset = DomainProcessing(path, name=name)
        df = dataset.process()
        return df 

    elif dataset_type == "TM":
        tms = os.listdir(path)
        paths = []
        for tm in tqdm(tms[:limit], desc="Looking for TMs"):
            if tm.endswith(".tmx"):
                path = Path(path)
                q = path / str(tm)
                # print(q)
                paths.append(str(q))
        try:
            dataset = DomainProcessing(paths)
            df = dataset.process()
                
            return df 
        except Exception as e:
            print(f"Error: {e}")

def feed_dicts(path: str, file: str):
    dict_general = {}
    if file.endswith(".en"):
        with open(path+file, "r") as src:
            return {"source":[sent.strip("\n") for sent in src.readlines()]}
                    
    elif file.endswith(".es"):
        with open(path+file, "r") as tgt:
            return {"target":[sent.strip("\n") for sent in tgt.readlines()]}

    return {}

def load_from_opus(paths: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dict1 = {}
    dict2 = {}
    for idx, path in enumerate(paths):
        file_list = os.listdir(path)
        for file in file_list:
            d = feed_dicts(path, file)
            if idx == 0:
                dict1.update(d)
            else:
                dict2.update(d)
    df1 = pd.DataFrame.from_dict(dict1, orient='index').transpose()
    df2 = pd.DataFrame.from_dict(dict2, orient='index').transpose()

    return df1, df2

paths = ["ELRC_EUIPO/", "NeuLab_TT/"]

elrc, neulab = load_from_opus(paths)
elrc.to_csv("elrc.csv")
neulab.to_csv("neulab.csv")

print(elrc.head())
print(neulab.head())
print("CSVs saved succesfully!!")




