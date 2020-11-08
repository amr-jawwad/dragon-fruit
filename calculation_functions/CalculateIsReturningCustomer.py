import pandas as pd

def CalculateIsReturningCustomer(Data: pd.DataFrame, COUNT_FAILED_ORDERS: bool):
    #I had to also sort by customer_rank, because there were orders that were on the same day, at the same hour
    #So the only information about their true order was in the rank.
    Data = Data.sort_values(['customer_id', 'order_date', 'order_hour','customer_order_rank'])

    #Converting order_date to datetime
    Data['order_date'] = pd.to_datetime(Data.order_date)

    if COUNT_FAILED_ORDERS:
        #For recalculating the rank
        Data['new_rank'] = 1 #Every order counts!
        Grouped = Data.groupby('customer_id').agg({"order_date":"diff", "new_rank":"cumsum"})
        Data = Data.drop(['customer_order_rank','new_rank'],axis=1)
        Grouped = Grouped.rename({"new_rank":"customer_order_rank"},axis=1)
    else:
        Grouped = Data[Data.is_failed==0].groupby('customer_id').agg({"order_date":"diff"})

    Grouped = Grouped.rename({"order_date":"day_diff"},axis=1)
    Grouped.day_diff = Grouped.day_diff.shift(-1)
    Grouped.day_diff = Grouped.day_diff.dt.days

    Data = pd.merge(Data,Grouped, left_index=True, right_index=True,how='left')

    Data['is_returning_customer'] = 0
    Data.loc[Data.day_diff<=180,'is_returning_customer'] = 1
    
    return Data