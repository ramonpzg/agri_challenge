
import sklearn.metrics as m
import pandas as pd
from os.path import join
from omegaconf import OmegaConf
from extract import load_table

def get_metrics(df, y_truth, y_pred, label):
    return dict(
        precision=round(m.precision_score(df[y_truth], df[y_pred], pos_label=label) * 100, 2),
        recall=round(m.recall_score(df[y_truth], df[y_pred], pos_label=label) * 100, 2),
        f1_score=round(m.f1_score(df[y_truth], df[y_pred], pos_label=label) * 100, 2),
        accuracy=round(m.accuracy_score(df[y_truth], df[y_pred]) * 100, 2)
    )

def confused_mtx(df, y_truth, y_pred):
    return m.confusion_matrix(df[y_truth], df[y_pred])

def generate_config(metrics, mtx, path, file_name):
    conf = OmegaConf.create({
        "facts": {
            "sample_size": int(sum(sum(mtx))),
            "healthy_preds": int(mtx[0][0]),
            "wrong_preds": int(mtx[1][0] - mtx[0][1])
        },
        "metrics": {k: float(v) for k, v in metrics.items()},
        "matrix": {
            "true_pos": int(mtx[0][0]),
            "true_neg": int(mtx[1][1]),
            "false_pos": int(mtx[0][1]),
            "false_neg": int(mtx[1][0])
        }
    })
    
    OmegaConf.save(conf, join(path, file_name))
    
    print(f"Config Successfully saved as {join(path, file_name)}")

if __name__ == "__main__":
    df = load_table(join("data", "processed"), "combined_table.parquet")
    metrics = get_metrics(df, "class_truth", "class_pred", 'Alternaria spp.')
    conf_mtx = confused_mtx(df, "class_truth", "class_pred")
    generate_config(metrics, conf_mtx, join("src", "configs"), "config.yml")
