# Dragon Fruit Documentation

# Contents
***
1. **[Functional Description](#1-functional-description)**
    * Task Descripton
    * Target Variable Definition and Calculation
    * Data Preparation and Feature Engineering
    * Training and Testing Strategy
    * Evaluation and Results
    * Assumptions
    * Technologies used
    * Future Work
2. **Technical Description**
    * How to run    
    * Code repository layout
    * Technologies used
    * Future Work

# 1. Functional Description
## Task Descripton
For the original description of the task, and the meaning of the complete list of variables, kindly check this [file](./original_README.md).
The task is to predict whether a customer will place an order in the upcoming 6 months, given the customer id.

The given data is on order-level *(time range:  2015-03-01 : 2017-02-27)*, detailing orders made and whether they failed or not, and by which customer, and some other details about the order like payment method and so on. While the testing data is given on a customer level, having one unique entry per customer, and whether this specific customer placed an order in the following 6 months.

## Target Variable Definition and Calculation
Our target variable is ***is_returning_customer***, which is a boolean variable that tells if a customer is going to order again in the following 6 months.

The main events here are orders; in the periods of time between orders for a specific customer the features don't change much, only the time features do. But, for example, other features like, which methods the customer historically used for payments, or which platform they used, do not change until the following order. Hence there is not much information in the periods of time between orders.

In the light of this, we can further define the target variable as:
>For each customer, for each order, it tells whether there is going to be another order within the span of 6 months from the order date.

#### Question?
Here comes an important point of definition: Are failed orders... orders? Meaning: should they be counted?

So, if a customer makes an order **x**, and within less than 6 months they make an order **y**.

if either **x** or **y** (or both) were failed orders, would the customers still count as returning?

As per my understanding, they should. For multiple reasons:
1. From business point of view, there is value of keeping customers on our platform, even if their orders fail sometimes. In the long term still having them in the customer base, increases the likelihood of more purchases.
2. In the specification, it did not specify that we want to predict a "successful" order in the following 6 months, just any order.
3. If we don't count Failed Orders, we may lose the information in these orders.

However, it's not very complicated to implement both scenarios (counting and ignoring failed orders), so I'm going to implement the logic in both cases, with a config boolen variable **COUNT_FAILED_ORDERS** to determine which definition to use.

### Target Variable Modelling
From a Machine Learning point of view, I thought of two ways to model this:
(Both are implemented)
#### 1. Classification Problem
We directly calculate the *is_returning_customer* variable by labelling the order entries by 0 if they don't have a subsequent order, from the same customer, in the subsequent 180 days, and 1 if they do. And that becomes our target label for the training purposes.
#### 2. Regression Problem
We calculate the intermediate variable *time_to_next_order*, and that labels each order entry with how much time (in hrs) until the following order from the same customer (if any). And that becomes our target variable for the training purposes.
However, this remains a classification task, so when it is time for prediction, the *time_to_next_order* is calculated, and then added to the *order_time* then it is calculated whether that new *predicted_next_order_time* falls within 6 months forward.

One possible advantage of that extra step in modelling is that it takes into consideration the information of when we are running this prediction, or how much time has passed since the last order by that customer.
In other words, the model will not be able to give us information relative to the time of running prediction, only to the time of the last order.

### Calculation method
1. We make sure Data is truly sorted by customer_id, order_time, order_hr.
2. We group by customer_id.
3. We calculate the order time difference (in hours) between each order and the following order.

Now the time difference is assigned in each order, how long in the past the previous order was. What we want is the difference being assigned to the previous order, how long in the future is the following order going to be. So...

4. We shift the newly-calculated time difference one row up, and this becomes the target variable.

5. (For the classification approach) The target variable is whether the date difference variable is <= 4320 hours (180 days).

**Important calculation note:** From the definition of the data (and also from my initial exploration), failed orders are not counted in the customer_order_rank. So if **COUNT_FAILED_ORDERS** is set to True then the rank will be recalculated.
[Notebook of Target Variable Calculation](./exploration_notebooks/00_Target_Variable_Calculation.ipynb)

[Calculation Implementation](./dragon_fruit/calculation_functions/CalculateFeatures.py)

## Assumptions
1. 6 months = 180 days