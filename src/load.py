
from pathlib import Path
import pandas as pd
from extract import load_table
import ibis

def create_db(path_in, path_out, file_name, table_name):
    path = Path(path_out)
    conn = ibis.duckdb.connect(path.joinpath(file_name))
    conn.register(path_in, table_name=table_name)
    print(f"Successfully loaded the {table_name} table!")

def create_parquet(path_in, path_out, file_name):
    path = Path(path_out)
    pd.read_parquet(path_in).to_parquet(path.joinpath(file_name))
    print(f"Successfully loaded the {file_name} table!")

def save_data(data, path_out, file_name):
    path_out = Path(path_out)
    if not path_out.exists(): path_out.mkdir(parents=True)
    data.to_parquet(path_out.joinpath(file_name))
    print(f"Successfully loaded the {file_name} table!")
    
if __name__ == "__main__":
    create_db(
        path_in=join("data", "processed", "transformed_data.parquet"),
        path_out=join("data", "dwarehouse"),
        file_name="db_analytics.ddb",
        table_name="truth_preds_challenge"
    )
