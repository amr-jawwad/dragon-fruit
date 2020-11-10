import pandas as pd
from xgboost import XGBRegressor
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
                                                            split_data= split_data)
    
    #This will replace NaNs in time-to-next-order for the last orders
    y_train = y_train.fillna(inf_time_to_next_order)
    y_valid = y_valid.fillna(inf_time_to_next_order)

    model = XGBRegressor(random_state= random_seed)
    model.fit(X_train, y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    verbose=True, early_stopping_rounds = early_stopping_rounds)

    Feature_Importance = pd.DataFrame(model.get_booster().get_score(importance_type='weight').items(),columns=['feature', 'importance'])

    Testing_Data['predicted_time_to_next_order'] = model.predict(Testing_Data[Feature_Columns])
    Testing_Data['predicted_order_time'] = pd.to_datetime(Testing_Data.order_time) + pd.to_timedelta(Testing_Data.predicted_time_to_next_order, unit='h')

    #end_date is the last day in the orders + 180 days
    end_date = (pd.to_datetime(Training_Data.order_time.max()) + pd.to_timedelta(180,unit='D')).round(freq='D')

    y_pred = Testing_Data.predicted_order_time <= end_date

    return (y_pred,
            None,
            Feature_Importance,
            model)