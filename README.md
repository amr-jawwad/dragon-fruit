# Dragon Fruit Documentation

# Contents

1. **[Functional Description](#1-functional-description)**
    * [Task Description](#task-description)
    * [Target Variable Definition and Calculation](#target-variable-definition-and-calculation)
    * [Data Preparation and Feature Engineering](#data-preparation-and-feature-engineering)
    * [Training and Testing Strategy](#training-and-testing-strategy)
    * [ML Models and the Rationales behind their Selection](##ml-models-and-the-rationales-behind-their-selection)
    * [Evaluation and Results](#evaluation-and-results)
    * [Assumptions](#assumptions)
    * [Technologies used](#technologies-used)
    * [Future Work](#future-work)
2. **[Technical Description](#2-technical-description)**
    * [How to run](#how-to-run)
    * [Code repository layout](#code-repository-layout)
    * [Technologies used](#technologies-used-1)
    * [Future Work](#future-work-1)

3. **[Insights](#3-insights)**

# 1. Functional Description
## Task Description
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

## Data Preparation and Feature Engineering
### Data Cleaning
1. As revealed from the [EDA](./exploration_notebooks/01_EDA.ipynb),  53 records had invalid outlying order dates before the beeginning date, they were removed.
2. There were 546 duplicates in the data, they were removed.

### Feature Engineering
1. Historical Feature
    * Time since last order
2. Time Features (from the order date)
    * day of month
    * day of week
    * month
    * year

##### Cyclic Encoding
Now handling time features is always tricky, we lose so much information if we encode them as categorical variables, because indeed there is some ordinality there.

So to keep the ordinality information we encode them as numeric features, **BUT** there's a problem with encoding them just as they are, we lose the periodic information. For example, in hours 1 is as close to 2, as 0 is to 23!

**Solution: Use cyclic encoding.**

**Cyclic encoding** essentially plots the variable range on a circle, by projecting it using sine and cosine.

Helpful source: http://blog.davidkaleko.com/feature-engineering-cyclical-features.html

Cyclic encoding was used for all the time features except for *year*.

#### Categorical Variables Encoding
We have three variables that are categorical in nature, but that are represented as numbers. Since there is no true ordinal relationship between them, this will be misleading to the model.

So One-Hot-Encoding was used for *payment_id, platform_id,* and *transmission_id.*
The rest of the ids had too many categories, with no further information, so they were dropped.

[Feature Engineering Notebook](./exploration_notebooks/02_Feature_Engineering.ipynb)

[Feature Engineering Code](./dragon_fruit/machine_learning/feature_engineering.py)

## Training and Testing Strategy
This problem is peculiar when it comes to training and especially testing the ML models.

First of all, it's time-based data, with all the consideration that comes from that, including trying to avoid label leakage, and taking care of causality for example when splitting the data, and so on.

Secondly, and more importantly, the testing data doesn't have the features that we used to train the model, only the customer_ids. This could have lead us to a solution where we build prediction models/forecasts per customer, but given:
1. That the majority of the customers (~60%) in the data are one-time orderers. (will lead to sparsity)
2. The lack of data about the customers themselves, but only about the orders.

I decided to take the approach of prediction per order.

In that case, the testing data will have to look like the training data, i.e. it will have to look like orders. Because that's how the model works, it takes an order, and gives the prediction whether there is going to be another order in 6 months from that same customer.

That is why, I decided to enrich the testing data with the details of the **last order** of that particular customer.

Ideally, to avoid leakage in general, I would have removed the entries that would be used to enrich testing from the training data to avoid leakage, but given that in the test data has ALL the customers in the training data, that means that we will remove all the one-time orderers, which is the majority of the data, so I decided not to do that at least in the first version of the model. But we certainly do not leak labels into the training data.

There are 2 customers that are in the testing data, but not in the training data, so I will drop them for now, but a future model can just give the median value, or another definition of a default value.

Now as explained in the **[Target Variable Definition](#target-variable-definition-and-calculation)** I said I will try 2 approaches, classification and regression, here's how each would work:

#### Classification approach:
1. Train model
2. Enrich Testing data
3. Predict the labels for the Testing data
4. Evaluate classification scores.

#### Regression approach:
1. Train model
2. Enrich Testing data
3. Predict the time-to-next-orders for the Testing data (in hours)
4. Add the predicted hours to the order_time to get the predicted_order_time
5. The 'is_returning_customer' boolean variable that is going to be calculated as whether the predicted_order_time is before or after '2017-08-28' (which is 6 months after '2017-02-28').
6. Evaluate classification scores.

#### Train-Validation Split
We have the testing data ready for us, but I decided to also split the training data into data for training and for validation, to use **validation** to choose the best parameters, and to avoid overfitting.

The data is time-dependent, so the sensible way to split the data is chronologically, meaning testing data (validation in this case), should always happen after the training data, chrnologically-speaking, to avoid leakage and not to violate the temporal property of the problem.

But in some cases, when we capture enough time information in the features, the problem becomes less (or totally) time-independent.

So I implemented both approaches: both *random* and *chronological* splits, with a config that chooses amongst them.
*Random* splitting comes with **stratification** based on the target label.

[Training and Testing Notebook](./exploration_notebooks/03_Training_and_Testing.ipynb)

[Models' directory](./dragon_fruit/machine_learning/)

## ML Models and the Rationales behind their Selection
ML Models Currently implemented:
1. Baseline
    It is always good to have some baseline algorithm, with a basic heuristic (preferably non-ML solution) to have something to compare to.
    In this case the baseline algorithm was simply the following:
    If a customer had ordered more than once in the training data, they are expected to order again in the following 6 months, if they had ordered only once, then they are not.

2. LogisticRegression
    One of the simplest classifiers out there, and I always like to also examine the performance of simpler models before moving to more sophisticaed ones.
    Because the many features introduced the risk of overfitting, I used **Cross-Validation** (5-fold) with this algorithm.

3. XGBoost (Regressor and Classifier) was chosen for the follwing reasons:
    * Robust to outliers and long-tailed distributions, which are prevalent according to [EDA](./exploration_notebooks/01_EDA.ipynb).
    * Advantageous in the case of many features (for feature sampling in bagging)
    * Though boosting sometimes has the risk of overfitting, **validation** was used to alleviate that, along with **early stopping**.
    * Can easily tackle **class imbalance** using class weights, which was implemented as an option
    Currently **class imbalance** is calculated internally as:
    number of negative examples / number of positive examples.

## Evaluation and Results
For evaluation, I was looking at:
1. Confusion Matrix
2. Classification report (specially: accuracy and f1-scores)
3. Area under the ROC curve (only for the classification approach for now)

As of the moment of writing this, the best results were achieved with these settings:
Parameter|Value
|---|---|
|Approach|Classification with XGBClassifier|
|Split|Chronological|
|Counting False Orders|No|
|Split|Chronological|
|Balancing Classes|Yes|

And the results were:
Metric|Value
|---|---|
|Accuracy|76.3%|
|F1 [No orders]|0.844|
|F1 [Orders]|0.509|
|Area Under ROC|0.745|

## Assumptions
* 6 months = 180 days

## Technologies used
* MLFlow
* XGBoost

## Future Work
* Getting classifcation probabilities even in the regression solution, by estimating confidence intervals
* Combine both approaches of counting failed orders and not counting them, by having features representing both (e.g. rank_with_failed, rank_without_failed)
* Further tackling class imbalance, by:
    * Supersampling the underrepresented class or Undersampling the majority one
    * More interestingly, further manipulate class weights or class sensetivities in the training, these weights may depend on:
        * The frequency of the class in the training data, naturally, and/or **[already implemented]**
        * More interestingly, the **penalty of misclassification from the business' point of view**, i.e. is it worse to misclassify a customer that they will buy if they will eventaully not, or the other way around?

# 2. Technical Description
## How to run
1. Make sure you're in the main directory of the repo
2. Make sure you have the right requirements, this can be ensured using two methods:
    a. Using pipenv (for virtual environments) (**Recommended**)
    1. Make sure you have [pipenv installed](https://pypi.org/project/pipenv/).
    2. Install the environment requirements using ```pipenv install```.
    3. Log into the environment using ```pipenv shell```
    
    b. Alternatively, if you prefer not to use virtual environments, you can install the requirements from  _[requirements.txt](./requirements.txt)_
    You can do that by running ```pip install -r requirements.txt```
3. Modify the config file:
    The config file has the settings for the run, including the file paths, ML model parameters, and all the configurations described in this document.
4. Run using ```PYTHONPATH=. python main.py```
5. Your run's results will be logged with MLFlow
#### How to run tests
1. Make sure that all the requirements are installed (from the previous section)
2. Run ```PYTHONPATH=. py.test```

**IMPORTANT NOTE**: The code checks for the existence of an already feature-engineered data file in ```data/```. to avoid running feature engineering every time, if you want to make sure that you re-run the feature engineering delete ```data/engineered_order_data*.csv```.


## Code repository layout
* **Main directory/**
    * Virtual Env and requirements file
    * '**config.json**': contains the settings for the run, check the meanings of the parameters [here](#config-file-parameters-meanings).
    * '**main.py**': has the main logic for the run, including reading the config file, logging into MLFlow, and running the different parts of the repo.
    * '**utils.py**': Contains some adapters to write different files into the artifacts directory.
    * **artifacts/**
    Contains the model pickle, and performance metrics, it's however recommended to view the results using MLFlow.
    * **data/**
    Contains the data files, including the input files, and an intermediate engineered data file.
    * **dragon_fruit/**
    Contains the main code files of the repo
        * **calculation_functions/**
            * **CalculateFeatures.py** Calculates the time difference features
            * **HelperFunctions.py** Some helper functions for feature engineering and train/valid split
        * **machine_learning/**
            * **models/** Contains a separate py file for each ML model, containing its implementation, including splitting the data, and the training, and the predictions.
            * **evaluation.py** For evaluating the predictions of the trained models.
            * **feature_engineering.py** The implementation of the feature engineering.
    * **exploration_notebooks/** Contains the exploration notebooks for EDA, feature engineering, training, testing,,, etc.
    * **mlruns/0** The output file for MLFlow containing the runs' results
    * **tests/** Contains tests

#### Config file parameters meanings
|Parameter|Meaning|Notes
|---|---|---|
File paths' parameters|| not recommended to change
random_seed| Fixed random seed for the entire project|not recommended to change
split_data|For the train/validation split|Options: ['normal', 'chronological']
model_of_selection|The ML model for the run|Options: ['Baseline', 'LogisticRegression', 'XGBClassifier', 'XGBRegressor']
DEFAULT_MODEL|The ML model to fall back to if the model_of_selection wasn't recognised|Currently fixed to XGBClassifier
|INF_TIME_TO_NEXT_ORDER|A numerical value that replaces infinity/NaT values for orders that were never followed by another order from the same customer
VALIDATION_DATA_SIZE|As a fraction of the training data size|0 < size < 1
COUNT_FAILED_ORDERS|Whether to count failed orders as orders (true), or to exclude them (false)|Options: [true, false]
early_stopping_rounds|For the ML model validation
balance_classes|Adjust class weights to account for class imbalance| Options: [true, false]. Implemented only to XGBClassifier for now

## Technologies used
* MLFlow
* Pipenv
* Git
* Pytest
* VSCode

## Future Work
* Implement Classes to make the project more object-oriented.
* Make an abstract class for the ML models, and specific models implement it.
* Use a tool to create a pipeline (e.g. luigi).
* Implement a pathway to consume models' pickles.
* Proper logging tools instead of the prints

# 3. Insights
* Baseline performance was already not bad at all (accuracy: 70%), meaning the heuristic of: one-time orderers are more likely not to order in the future, while multiple-timers are likely to is quite good. It's quite straightforward, yet an important insight.
* Due to the nature of the data, especially:
    * The lack of information on customers themselves, outside their ordering activity, because
    * The training data being on orders' level, and
    * The testing data being on customers' level, which leads to,
    * The training data being biased to represent orders rather than "non-orders"

    The False Positive rate is quite high.

* It would be interesting to get the business' perspective on which is more "harmful" in this case, a false positive, or a false negative, and the models could be tuned accordingly.
If I would have an opinion, I would need to know more information about the usage of such model.
One example I can think of is the following:
We want to send out promotion vouchers, but only to customers we suspect that they are **not** going to order in the upcoming period of time, to encourage them to do so.
In that case we lose money, in a way, if we offer that voucher to someone who would have ordered anyway regardless whether they get a voucher.
In that case for example it could be calculated by a simple risk analysis that depends on the value of the voucher, the mean value of orders by that customer, and the false positive rate, and of course, how much money we gain from converting a non-orderer to an orderer.
But usually, business-wise, it's preferable to convert a non-orderer, even at the expense of giving a voucher to a an orderer-anyway, so in that scenario I would think that False Negatives are relatively benign.

* Enriching this data with customer-specific information could enhance the performance.
* From the fact that the performance of the models is consistently better when we do not count failed orders, I assume that the testing data does not do so, as well.
* For other potential soliutions to some of the problems mentioned here, kindly check the [future work](#future-work) section.