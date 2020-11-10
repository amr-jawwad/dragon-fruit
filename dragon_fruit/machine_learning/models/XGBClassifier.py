import pandas as pd
from xgboost import XGBClassifier
from dragon_fruit.calculation_functions.HelperFunctions import my_train_valid_split

def get_predictions(Training_Data: pd.DataFrame,
                    Testing_Data: pd.DataFrame,
                    Feature_Columns: list,
                    target_col: str,
                    inf_time_to_next_order: float,
                    split_data: str,
                    validation_data_size: float,
                    random_seed: int,
                    early_stopping_rounds: int = 20):
    
    X_train, X_valid, y_train, y_valid = my_train_valid_split(Data= Training_Data[Feature_Columns+[target_col,'order_time']],
                                                            target_col= target_col,
                                                            random_seed= random_seed,
                                                            validation_data_size= validation_data_size,
                                                            split_data= split_data,
                                                            stratify= True)

    model = XGBClassifier(random_state= random_seed)
    model.fit(X_train, y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    verbose=True, early_stopping_rounds = early_stopping_rounds)

    Feature_Importance = pd.DataFrame(model.get_booster().get_score(importance_type='weight').items(),columns=['feature', 'importance'])

    y_pred = model.predict(Testing_Data[Feature_Columns])
    y_proba = model.predict_proba(Testing_Data[Feature_Columns])[:,1]

    return (y_pred,
            y_proba,
            Feature_Importance,
            model)