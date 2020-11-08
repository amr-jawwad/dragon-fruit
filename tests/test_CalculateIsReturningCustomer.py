import pytest
import numpy as np
import pandas as pd

from calculation_functions.CalculateIsReturningCustomer import CalculateIsReturningCustomer

mock_orders_data = {'customer_id':
                    {0: 'a',
                     1: 'a',
                     2: 'a',
                     3: 'a',
                     4: 'a',
                     5: 'a'},
                    'order_date': 
                    {0: '2016-09-01',
                     1: '2016-09-02',
                     2: '2016-09-04',
                     3: '2016-09-05',
                     4: '2016-09-14',
                     5: '2016-10-31'},
                    'order_hour':
                    {0: 14, 1: 20, 2: 14, 3: 11, 4: 15, 5: 19},
                    'customer_order_rank':
                    {0: np.nan, 1: 1.0, 2: 2.0, 3: np.nan, 4: 3.0, 5: 4.0},
                    'is_failed':
                    {0: 1, 1: 0, 2: 0, 3: 1, 4: 0, 5: 0}
                    }
def is_matching(List1: np.array, List2: np.array):
    if len(List1) != len(List2):
        return False
    return ((List1 == List2) | (np.isnan(List1) & np.isnan(List2))).all()


def test_calculate_is_returning_customer_counting_failed():
    result_df = CalculateIsReturningCustomer(Data= pd.DataFrame(mock_orders_data), COUNT_FAILED_ORDERS= True)

    assert (is_matching(result_df.customer_order_rank.to_numpy(),[1, 2, 3, 4, 5, 6])), "Some values in the calculated order rank were incorrect"
    assert (is_matching(result_df.day_diff.to_numpy(),[1.0, 2.0, 1.0, 9.0, 47.0, np.nan])), "Some values in the day_diff were incorrect"
    assert (is_matching(result_df.is_returning_customer.to_numpy(),[1, 1, 1, 1, 1, 0])), "Some values in the is_returning_customer were incorrect"

def test_calculate_is_returning_customer_not_counting_failed():
    result_df = CalculateIsReturningCustomer(Data= pd.DataFrame(mock_orders_data), COUNT_FAILED_ORDERS= False)

    assert (is_matching(result_df.day_diff.to_numpy(),[np.nan, 2.0, 10.0, np.nan, 47.0, np.nan])), "Some values in the day_diff were incorrect"
    assert (is_matching(result_df.is_returning_customer.to_numpy(),[0, 1, 1, 0, 1, 0])), "Some values in the is_returning_customer were incorrect"


