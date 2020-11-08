import pandas as pd
import numpy as np

def calculate_cyclic_encoding(Column: pd.Series, range_size: int, mode: str = 'sin'):
    if mode == 'cos':
        return np.cos(Column*(2.*np.pi/range_size))
    else:
        return np.sin(Column*(2.*np.pi/range_size))