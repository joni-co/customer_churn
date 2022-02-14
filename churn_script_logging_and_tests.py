'''
Library for functions in the customer churn project

Author: Jonas
Date: 11.02.2022
'''

import logging
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering, train_models


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info(
            "SUCCESS: There are {} rows in your dataframe".format(
                df.shape[0]))
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        perform_eda(df)
        logging.info("SUCCESS: The eda created some viz")
    except BaseException:
        logging.error("ERROR: The eda did not work")

    try:
        assert my_plot != 0
    except BaseException:
        logging.error("ERROR: There is no figure to save")


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        df = encoder_helper(df, category_column_lst, response)
        logging.info("SUCCESS: The encoder worked for our categorical columns")
    except BaseException:
        logging.error("ERROR: The encoder did not work")
    
    try:
        assert df != 0
    except:
        logging.error("ERROR: Could not encode categorical variables")


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            df, keep_cols, response)
        logging.info("SUCCESS: The feature engineering part worked")
    except BaseException:
        logging.error("ERROR: The feature engineering did not work")
    
    try:
        assert X_train != 0
    except:
        logging.error("ERROR: Feature Engineering")
    
    try:
        assert y_train != 0
    except:
        logging.error("ERROR: Feature Engineering")


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        train_models(X_train, X_test, y_train, y_test)
        logging.info("SUCCESS: The model training part worked")
    except BaseException:
        logging.error("ERROR: The model training did not work")


if __name__ == "__main__":
    test_import(import_data)
    test_eda(perform_eda)
    test_encoder_helper(encoder_helper)
    test_perform_feature_engineering(perform_feature_engineering)
    test_train_models(train_models)