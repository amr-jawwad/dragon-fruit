import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix

def classification_evaluation(y_true: pd.Series, y_pred: np.array, y_pred_proba: np.array = None):
    Confusion_Matrix = confusion_matrix(y_true, y_pred)
    Classification_Report = classification_report(y_true, y_pred, output_dict=True)

    if y_pred_proba is not None:
        AUC = roc_auc_score(y_true, y_pred_proba)
    else:
        AUC = None

    return (Confusion_Matrix, Classification_Report, AUC)