from pathlib import Path
import pandas as pd

def save_data(data, path_out, file_name):
    path_out = Path(path_out)
    if not path_out.exists(): path_out.mkdir(parents=True)
    data.to_parquet(path_out.joinpath(file_name))
    print(f"Successfully loaded the {file_name} table!")