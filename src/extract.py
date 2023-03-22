
import pandas as pd
from os.path import join
from glob import glob
from load import save_data
import json
from typing import List

def get_files(directory: str) -> List:
    return glob(join(directory, "*.json"))


def read_files(data_files: str) -> pd.DataFrame:
        
    dfs_list = []
    
    for file in data_files:    
        with open(file, "r") as sample:
            data = json.load(sample)

        item = data["item"]['name']

        if data["annotations"]:
            anno_name = data["annotations"][0]["name"]
        else:
            anno_name = "Undetected"
        
        df = pd.DataFrame(data=[[item, anno_name]], columns=["item_id", "class"])
        dfs_list.append(df)
    
    return pd.concat(dfs_list, axis=0)

def load_table(data_path: str, file_name: str) -> pd.DataFrame:
    return pd.read_parquet(join(data_path, file_name))

if __name__ == "__main__":
    actuals = get_files("data/raw/ground_truth/")
    predictions = get_files("data/raw/predictions/")
    df_truth = read_files(actuals)
    df_preds = read_files(predictions)
    save_data(df_truth, join("data", "interim"), "actuals_table.parquet")
    save_data(df_preds, join("data", "interim"), "predicted_table.parquet")
