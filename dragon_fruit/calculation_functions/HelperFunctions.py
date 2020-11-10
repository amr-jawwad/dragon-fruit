import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def calculate_cyclic_encoding(Column: pd.Series, range_size: int, mode: str = 'sin'):
    if mode == 'cos':
        return np.cos(Column*(2.*np.pi/range_size))
    else:
        return np.sin(Column*(2.*np.pi/range_size))

def my_train_valid_split(Data: pd.DataFrame, target_col:str, random_seed:int,
                        validation_data_size: float = 0.2,
                        split_data: str = 'normal', stratify:bool = False,):
    
    if split_data == 'chronological':
        Data = Data.sort_values('order_time')
        split_index = int(np.round(len(Data) * (1-validation_data_size)))
        Data_train = Data.iloc[:split_index].copy()
        Data_valid = Data.iloc[split_index:].copy()

        X_train = Data_train.drop([target_col,'order_time'],axis=1)
        y_train = Data_train[target_col]
        X_valid = Data_valid.drop([target_col,'order_time'],axis=1)
        y_valid = Data_valid[target_col]

    else:
        if stratify:
            stratify_col = Data[target_col]
        else:
            stratify_col = None
        
        X_train, X_valid, y_train, y_valid = train_test_split(Data.drop([target_col,'order_time'], axis=1),
                                                            Data[target_col],
                                                            test_size=validation_data_size,
                                                            stratify=stratify_col,
                                                            random_state=random_seed)

    return (X_train, X_valid, y_train, y_valid)
        