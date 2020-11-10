import numpy as np
import csv
import pickle

def save_confusion_matrix(Confusion_Matrix: np.ndarray, FilePath: str):
    try:
        with open(FilePath, "w") as File:
            csvWriter = csv.writer(File, delimiter= '\t')
            csvWriter.writerows(Confusion_Matrix)
    except:
        raise PermissionError("Couldn't write the file %s." % FilePath)

def save_classification_report(Classification_Report: str, FilePath: str):
    try:
        with open(FilePath, "w") as text_file:
            print(Classification_Report, file=text_file)
    except:
        raise PermissionError("Couldn't write the file %s." % FilePath)

def save_model(model, FilePath: str):
    try:
        with open(FilePath, 'wb') as pickle_file:
            pickle.dump(model, pickle_file)

    except:
        raise PermissionError("Couldn't write the file %s." % FilePath)