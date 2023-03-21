
import pandas as pd
from os.path import join
from extract import load_table
from load import save_data

def merge_truth_preds(df1, df2, **kwargs):
    return pd.merge(left=df1, right=df2, **kwargs)

if __name__ == "__main__":
    df1 = load_table(join("data", "interim"), "actuals_table.parquet")
    df2 = load_table(join("data", "interim"), "predicted_table.parquet")
    df_combined = merge_truth_preds(df1, df2, left_on="item_id", right_on="item_id", suffixes=("_truth", "_pred"))
    save_data(df_combined, join("data", "processed"), "combined_table.parquet")
