import pandas as pd 
import time
from sklearn.model_selection import train_test_split
import os

def split_domain_datasets(path:str):
    for dom in domains:
        df = pd.read_csv(os.path.join(path,dom), sep=",")
        train, valid_and_test = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
        valid, test = train_test_split(valid_and_test, test_size=0.5, shuffle=False)
        
        names = ["train_set", "valid_set", "test_set"]
        splits = [train, valid, test]
        file = dom.split(".")[0]
        
        for split, name in zip(splits, names):
            split = split[~split["source"].isnull()]
            split.to_csv(f'Splits/{file}_{name}.csv', index=False)
        
    print("splitted succesfully")

if __name__ == "__main__":
    # jnitially you have merged training_dataset with nulls
    path = "Domains/"
    domains = os.listdir(path)
    split_domain_datasets(path)

