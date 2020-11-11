import pandas as pd

def get_predictions(Testing_Data: pd.DataFrame):
        
    y_pred = (Testing_Data.customer_order_rank>1).to_numpy()

    return (y_pred,
            None,
            None,
            None)