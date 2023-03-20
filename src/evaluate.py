from sklearn import metrics as m
from extract import load_table

def get_metrics(data, y_truth, y_pred):
    y_truth = data['y_truth'].values
    y_preds = data['y_pred'].values
    
    accuracy = m.accuracy_score
    precision = m.accuracy_score
    f1_score = m.accuracy_score
    recall = m.accuracy_score
    specificity = m.accuracy_score