import pandas as pd

from dragon_fruit.calculation_functions.CalculateFeatures import calculate_time_between_orders
from dragon_fruit.calculation_functions.HelperFunctions import calculate_cyclic_encoding

def run_data_engineering(original_Data: pd.DataFrame,
                         count_failed_orders:bool = True,
                         start_date: str = '2015-03-01'):

    Data = original_Data.copy()
    #Data Cleaning
    Data = Data[Data.order_date>= start_date]
    Data = Data.drop_duplicates()

    #Calculate Target features (+ time_since_last_order feature)
    Data = calculate_time_between_orders(Data, count_failed_orders)

    #Time features

    #As it is, we're not using the information embedded in 'order_date', so let's extract it!
    Data['day_of_month'] = Data.order_time.dt.day #To capture monthly seasonality
    Data['day_of_week'] = Data.order_time.dt.weekday #To capture weekly seasonality
    Data['month'] = Data.order_time.dt.month #To capture within year seasonality
    Data['year'] = Data.order_time.dt.year #To capture YoY growth

    #Cyclic features
    Data['hour_sin'] = calculate_cyclic_encoding(Data.order_hour, 24, 'sin')
    Data['hour_cos'] = calculate_cyclic_encoding(Data.order_hour, 24, 'cos')

    Data['day_of_month_sin'] = calculate_cyclic_encoding(Data.day_of_month-1, 30, 'sin') #We subtract 1 because it doesn't start from 0
    Data['day_of_month_cos'] = calculate_cyclic_encoding(Data.day_of_month-1, 30, 'cos')

    Data['day_of_week_sin'] = calculate_cyclic_encoding(Data.day_of_week, 7, 'sin')
    Data['day_of_week_cos'] = calculate_cyclic_encoding(Data.day_of_week, 7, 'cos')

    Data['month_sin'] = calculate_cyclic_encoding(Data.month-1, 12, 'sin') #We subtract 1 because it doesn't start from 0
    Data['month_cos'] = calculate_cyclic_encoding(Data.month-1, 12, 'cos')

    #Dropping irrelevant features
    #Categories that are too many, with no extra details, don't add much information, and hence will be dropped
    Data = Data.drop(['restaurant_id', 'city_id'], axis=1)

    #Dropping already-processed features:
    Data = Data.drop(['order_date', 'order_hour', 'day_of_month', 'day_of_week', 'month',], axis=1)

    #Encoding categorical variables
    Data = pd.get_dummies(Data, columns = ['payment_id', 'platform_id', 'transmission_id'])

    #If we don't count failed orders, then it could be a good idea to remove them for now
    if count_failed_orders == False:
        Data = Data[Data.is_failed==0].copy()
        Data = Data.drop('is_failed',axis=1)
    
    return Data

def enrich_testing_data(Training_Data: pd.DataFrame,
                        original_Testing_Data: pd.DataFrame,
                        Feature_Columns: list):

    Testing_Data = original_Testing_Data.copy()

    #Getting the last order per customer
    Last_orders_per_customer = Training_Data.drop_duplicates(subset='customer_id', keep = 'last')

    #Merging with the Testing data
    Testing_Data = pd.merge(Testing_Data, Last_orders_per_customer[Feature_Columns + ['order_time', 'customer_id']], on='customer_id')
    
    return Testing_Data