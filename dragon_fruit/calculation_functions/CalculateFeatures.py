import pandas as pd

def calculate_time_between_orders(Data: pd.DataFrame, COUNT_FAILED_ORDERS: bool):
    #I had to also sort by customer_rank, because there were orders that were on the same day, at the same hour
    #So the only information about their true order was in the rank.

    #First let's combine order_date and order_hr to create order_time

    Data['order_time'] = pd.to_datetime(Data.order_date + ' ' + Data.order_hour.astype(str) +':00:00')


    Data = Data.sort_values(['customer_id', 'order_time','customer_order_rank'])

    if COUNT_FAILED_ORDERS:
        #For recalculating the rank
        Data['new_rank'] = 1 #Every order counts!
        Grouped = Data.groupby('customer_id').agg({"order_time":"diff", "new_rank":"cumsum"})
        Data = Data.drop(['customer_order_rank','new_rank'],axis=1)
        Grouped = Grouped.rename({"new_rank":"customer_order_rank"},axis=1)
    else:
        Grouped = Data[Data.is_failed==0].groupby('customer_id').agg({"order_time":"diff"})

    Grouped = Grouped.rename({"order_time":"time_since_last_order"},axis=1)
    Grouped['time_to_next_order'] = Grouped.time_since_last_order.shift(-1)
    Grouped.time_since_last_order = (Grouped.time_since_last_order.dt.total_seconds())//3600
    Grouped.time_to_next_order = (Grouped.time_to_next_order.dt.total_seconds())//3600

    Data = pd.merge(Data,Grouped, left_index=True, right_index=True,how='left')

    Data['is_returning_customer'] = 0
    Data.loc[Data.time_to_next_order<=4320,'is_returning_customer'] = 1 #4320 is the number of hours in 180 days (~6 months)
    
    return Data