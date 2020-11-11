from sklearn.linear_model import LogisticRegressionCV
import pandas as pd


def get_predictions(Training_Data: pd.DataFrame,
                    Testing_Data: pd.DataFrame,
                    Feature_Columns: list,
                    target_col: str,
                    inf_time_to_next_order: float,
                    random_seed: int):
    
    X_train, y_train = Training_Data[Feature_Columns], Training_Data[target_col]

    X_train['time_since_last_order'] = X_train['time_since_last_order'].fillna(inf_time_to_next_order)
    Testing_Data['time_since_last_order'] = Testing_Data['time_since_last_order'].fillna(inf_time_to_next_order)

    model = LogisticRegressionCV(cv =5, random_state= random_seed).fit(X_train, y_train)

    y_pred = model.predict(Testing_Data[Feature_Columns])
    y_proba = model.predict_proba(Testing_Data[Feature_Columns])[:,1]

    return (y_pred,
            y_proba,
            None,
            model)