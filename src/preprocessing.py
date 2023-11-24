import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
import utils as util


# load data
def load_data(config_data: dict) -> pd.DataFrame:

    X_train = util.pickle_load('../' + config_data['train_test_data']['directory'] + config_data['train_test_data']['X_train'])
    y_train = util.pickle_load('../' + config_data['train_test_data']['directory'] + config_data['train_test_data']['y_train'])

    X_test = util.pickle_load('../' + config_data['train_test_data']['directory'] + config_data['train_test_data']['X_test'])
    y_test = util.pickle_load('../' + config_data['train_test_data']['directory'] + config_data['train_test_data']['y_test'])

    # concatenate x and y each set
    train_set = pd.concat(
        [X_train, y_train],
        axis = 1
    )
    test_set = pd.concat(
        [X_test, y_test],
        axis = 1
    )

    return train_set, test_set


# encoding fitting
def ohe_fit(data_tobe_fitted: dict, ohe_path: str) -> OneHotEncoder:

    ohe_obj = OneHotEncoder(sparse_output = False)
    ohe_obj.fit(np.array(data_tobe_fitted).reshape(-1, 1))

    util.pickle_dump(ohe_obj, ohe_path)
    
    return ohe_obj


# encoding transform
def ohe_transform(set_data: pd.DataFrame, tranformed_column: str, ohe_obj: OneHotEncoder) -> pd.DataFrame:
    
    set_data = set_data.copy()
    features = ohe_obj.transform(np.array(set_data[tranformed_column].to_list()).reshape(-1, 1))

    features = pd.DataFrame(
        features,
        columns = list(ohe_obj.categories_[0])
    )

    # set index by original set data index
    features.set_index(
        set_data.index,
        inplace = True
    )

    # concatenate new features with original set data
    set_data = pd.concat(
        [features, set_data],
        axis = 1
    )

    # drop columns
    set_data.drop(
        columns = tranformed_column,
        inplace = True
    )

    new_col = [str(col_name) for col_name in set_data.columns.to_list()]
    set_data.columns = new_col

    return set_data


# RandomUnderSampling Method
def rus_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:

    set_data = set_data.copy()
    rus = RandomUnderSampler(random_state = 26)

    x_rus, y_rus = rus.fit_resample(set_data.drop(config_data['data_source']['target_name'], axis = 1), 
                                    set_data[config_data['data_source']['target_name']])

    set_data_rus = pd.concat([x_rus, y_rus], axis = 1)
    return set_data_rus


# RandomOverSampling Method
def ros_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    
    set_data = set_data.copy()
    ros = RandomOverSampler(random_state = 11)

    x_ros, y_ros = ros.fit_resample(set_data.drop(config_data['data_source']['target_name'], axis = 1), 
                                    set_data[config_data['data_source']['target_name']])

    set_data_ros = pd.concat([x_ros, y_ros], axis = 1)
    return set_data_ros


# SMOTE Method
def smote_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    
    set_data = set_data.copy()
    sm = SMOTE(random_state = 112)

    x_sm, y_sm = sm.fit_resample(set_data.drop(config_data['data_source']['target_name'], axis = 1),
                                 set_data[config_data['data_source']['target_name']])
    
    set_data_sm = pd.concat([x_sm, y_sm], axis = 1)
    return set_data_sm


if __name__ == '__main__':

    # load configuration file
    config_data = util.load_config()

    # load dataset
    train_set, test_set = load_data(config_data)

    # looping for encoding
    columns_to_encode = ['Education', 'City', 'Gender', 'EverBenched']
    for column in columns_to_encode:
        ohe = ohe_fit(
            config_data['data_defense'][column]['value'],
            '../' + config_data[f'ohe_{column.lower()}_path']
        )

        train_set = ohe_transform(train_set, column, ohe)
        test_set = ohe_transform(test_set, column, ohe)

    # data resampling
    train_set_rus = rus_fit_resample(train_set)
    train_set_ros = ros_fit_resample(train_set)
    train_set_smote = smote_fit_resample(train_set)

    # data dump for train and test
    X_train = {
        'WithoutResampling' : train_set.drop(columns = config_data['data_source']['target_name']),
        'Undersampling' : train_set_rus.drop(columns = config_data['data_source']['target_name']),
        'Oversampling' : train_set_ros.drop(columns = config_data['data_source']['target_name']),
        'SMOTE' : train_set_smote.drop(columns = config_data['data_source']['target_name'])
    }

    y_train = {
        'WithoutResampling' : train_set[config_data['data_source']['target_name']],
        'Undersampling' : train_set_rus[config_data['data_source']['target_name']],
        'Oversampling' : train_set_ros[config_data['data_source']['target_name']],
        'SMOTE' : train_set_smote[config_data['data_source']['target_name']]
    }

    X_train_feng_path = '../' + config_data['train_test_data']['directory'] + config_data['train_test_data']['X_train_feng']
    y_train_feng_path = '../' + config_data['train_test_data']['directory'] + config_data['train_test_data']['y_train_feng']

    X_test_feng_path = '../' + config_data['train_test_data']['directory'] + config_data['train_test_data']['X_test_feng']
    y_test_feng_path = '../' + config_data['train_test_data']['directory'] + config_data['train_test_data']['y_test_feng']

    util.pickle_dump(X_train, X_train_feng_path)
    util.pickle_dump(y_train, y_train_feng_path)

    util.pickle_dump(test_set.drop(columns = config_data['data_source']['target_name']), X_test_feng_path)
    util.pickle_dump(test_set[config_data['data_source']['target_name']], y_test_feng_path)

    print('preprocessing good')
