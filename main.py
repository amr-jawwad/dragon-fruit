import pandas as pd
import os.path
import json

from dragon_fruit.machine_learning.feature_engineering import run_data_engineering, enrich_testing_data
from dragon_fruit.machine_learning.evaluation import classification_evaluation

CONFIG_FILE_PATH = 'config.json'

with open(CONFIG_FILE_PATH, 'r') as config_file:
    config_dict = json.load(config_file)


#Files' Paths
TRAINING_DATA_PATH = config_dict['TRAINING_DATA_PATH']
ENGINEERED_DATA_PATH = config_dict['ENGINEERED_DATA_PATH']
TESTING_DATA_PATH = config_dict['TESTING_DATA_PATH']

#Logic Config
COUNT_FAILED_ORDERS = config_dict['COUNT_FAILED_ORDERS']

#General Constants
random_seed = config_dict['random_seed']

#ML Model Selection
model_of_selection = config_dict['model_of_selection']
DEFAULT_MODEL = config_dict['DEFAULT_MODEL']

#ML Model config
split_data = config_dict['split_data'] #options: ['normal', 'chronological']
early_stopping_rounds = config_dict['early_stopping_rounds']

#ML Training Constants
INF_TIME_TO_NEXT_ORDER = config_dict['INF_TIME_TO_NEXT_ORDER'] #This will replace NaNs in time-to-next-order for the last orders
VALIDATION_DATA_SIZE = config_dict['VALIDATION_DATA_SIZE'] #As a fraction of the training data



#Checking if Engineered data already exists, to skip remaking it.
if os.path.isfile(ENGINEERED_DATA_PATH):
    print("Engineered data already found at %s." % ENGINEERED_DATA_PATH)
    print("Loading Engineered Data.")
    try:
        Engineered_Data = pd.read_csv(ENGINEERED_DATA_PATH)
    except:
        raise RuntimeError("Couldn't load the file %s for reading." %ENGINEERED_DATA_PATH)

#Engineered data doesn't exist, running feature engineering.
else:
    print("Engineered data not found at %s." % ENGINEERED_DATA_PATH)
    print("Will run Feature Engineering.")
    print("Loading Original Data from %s." % TRAINING_DATA_PATH)
    try:
        Data = pd.read_csv(TRAINING_DATA_PATH)
    except:
        raise FileNotFoundError("Couldn't load the file %s for reading." %TRAINING_DATA_PATH)
    print("Original Data loaded successfully.")
    print("Running Feature Engineering")

    Engineered_Data = run_data_engineering(original_Data= Data,
                                           count_failed_orders= COUNT_FAILED_ORDERS,
                                           start_date= '2015-03-01')
    print("Feature Engineering concluded successfully.")
    print("Saving Engineered Data at %s." % ENGINEERED_DATA_PATH)
    try:
        Engineered_Data.to_csv(ENGINEERED_DATA_PATH,index=False)
    except:
        raise PermissionError("Unable to write the Engineered Data at %s" % ENGINEERED_DATA_PATH)

    #Enriching Testing Data to send it to the model

print("Loading Testing Data from %s." % TESTING_DATA_PATH)
try:
    Testing_Data = pd.read_csv(TESTING_DATA_PATH)
except:
    raise FileNotFoundError("Couldn't load the file %s for reading." % TESTING_DATA_PATH)

print("Loaded Testing Data successfully.")
print("Enriching Testing Data...")
Feature_Columns = list(Engineered_Data.drop(
                                            ['customer_id', 'order_time',
                                            'time_to_next_order', 'is_returning_customer']
                                            ,axis=1).columns)
Enriched_Test_Data = enrich_testing_data(Training_Data= Engineered_Data,
                                        original_Testing_Data= Testing_Data,
                                        Feature_Columns= Feature_Columns)

print("Enriched Testing Data successfully.")

if model_of_selection == 'XGBRegressor':
    from dragon_fruit.machine_learning.models.XGBRegressor import get_predictions

    y_pred, y_proba, Feature_Importance, model = get_predictions(Training_Data= Engineered_Data,
                                                                Testing_Data= Enriched_Test_Data,
                                                                Feature_Columns= Feature_Columns,
                                                                target_col= 'time_to_next_order',
                                                                inf_time_to_next_order= INF_TIME_TO_NEXT_ORDER,
                                                                split_data= split_data,
                                                                validation_data_size= VALIDATION_DATA_SIZE,
                                                                random_seed= random_seed,
                                                                early_stopping_rounds= early_stopping_rounds)
    
    Confusion_Matrix, Classification_Report, _ = classification_evaluation(y_true= Enriched_Test_Data['is_returning_customer'],
                                                                           y_pred= y_pred)

    # print(Confusion_Matrix)
    # print(Classification_Report)


elif model_of_selection == 'XGBClassifier':
    from dragon_fruit.machine_learning.models.XGBClassifier import get_predictions

    y_pred, y_proba, Feature_Importance, model = get_predictions(Training_Data= Engineered_Data,
                                                                Testing_Data= Enriched_Test_Data,
                                                                Feature_Columns= Feature_Columns,
                                                                target_col= 'is_returning_customer',
                                                                inf_time_to_next_order= INF_TIME_TO_NEXT_ORDER,
                                                                split_data= split_data,
                                                                validation_data_size= VALIDATION_DATA_SIZE,
                                                                random_seed= random_seed,
                                                                early_stopping_rounds= early_stopping_rounds)
    
    Confusion_Matrix, Classification_Report, AUC = classification_evaluation(y_true= Enriched_Test_Data['is_returning_customer'],
                                                                           y_pred= y_pred,
                                                                           y_pred_proba= y_proba)
    # print(Confusion_Matrix)
    # print(Classification_Report)
    # print(AUC)

else:
    print("Unrecognised model selection: %s." % model_of_selection)
    print("Falling back to default model: %s." % DEFAULT_MODEL)
    from dragon_fruit.machine_learning.models.XGBClassifier import get_predictions

    y_pred, y_proba, Feature_Importance, model = get_predictions(Training_Data= Engineered_Data,
                                                                Testing_Data= Enriched_Test_Data,
                                                                Feature_Columns= Feature_Columns,
                                                                target_col= 'is_returning_customer',
                                                                inf_time_to_next_order= INF_TIME_TO_NEXT_ORDER,
                                                                split_data= split_data,
                                                                validation_data_size= VALIDATION_DATA_SIZE,
                                                                random_seed= random_seed,
                                                                early_stopping_rounds= early_stopping_rounds)
    
    Confusion_Matrix, Classification_Report, AUC = classification_evaluation(y_true= Enriched_Test_Data['is_returning_customer'],
                                                                           y_pred= y_pred,
                                                                           y_pred_proba= y_proba)



