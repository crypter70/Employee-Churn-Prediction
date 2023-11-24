import pandas as pd
from sklearn.model_selection import train_test_split
import utils as util


# read raw data
def read_raw_data(config: dict) -> pd.DataFrame:
    raw_dataset = pd.DataFrame()

    raw_dataset_dir = '../' + config['data_source']['directory'] + config['data_source']['file_name']  
    raw_dataset = pd.read_csv(raw_dataset_dir, encoding='utf-8')
    
    return raw_dataset


# check data
def check_data(input_data, config, api = False):

    # Education
    assert input_data['Education'][0] in config['data_defense']['Education']['value'] or\
        input_data['Education'][0] != '',\
        f"Education must be in list {config['data_defense']['Education']['value']}, and cannot be empty."
    
    # City
    assert input_data['City'][0] in config['data_defense']['City']['value'] or\
        input_data['City'][0] != '',\
        f"City must be in list {config['data_defense']['City']['value']}, and cannot be empty."
    
    # Gender
    assert input_data['Gender'][0] in config['data_defense']['Gender']['value'] or\
        input_data['Gender'][0] != '',\
        f"Gender must be in list {config['data_defense']['Gender']['value']}, and cannot be empty."
    
    # EverBenched
    assert input_data['EverBenched'][0] in config['data_defense']['EverBenched']['value'] or\
        input_data['EverBenched'][0] != '',\
        f"EverBenched must be in list {config['data_defense']['EverBenched']['value']}, and cannot be empty."
    
    # JoiningYear
    assert input_data.JoiningYear.between(config['data_defense']['JoiningYear'][0], config['data_defense']['JoiningYear'][1]).sum() == len(input_data),\
        "an error occurs in JoiningYear range."
    
    # PaymentTier
    assert input_data.PaymentTier.between(config['data_defense']['PaymentTier'][0], config['data_defense']['PaymentTier'][1]).sum() == len(input_data),\
        "an error occurs in PaymentTier range."
    
    # Age
    assert input_data.Age.between(config['data_defense']['Age'][0], config['data_defense']['Age'][1]).sum() == len(input_data),\
        "an error occurs in Age range."
    
    # ExperienceInCurrentDomain
    assert input_data.ExperienceInCurrentDomain.between(config['data_defense']['ExperienceInCurrentDomain'][0], config['data_defense']['ExperienceInCurrentDomain'][1]).sum() == len(input_data),\
        "an error occurs in ExperienceInCurrentDomain range."


if __name__ == "__main__":

    # load configuration file
    config_data = util.load_config()

    # read data
    raw_dataset = read_raw_data(config_data)

    # data defence
    check_data(raw_dataset, config_data)

    # data splitting 
    X = raw_dataset[config_data['data_source']['features']].copy()
    y = raw_dataset[config_data['data_source']['target_name']].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = config_data['data_source']['test_size'], 
                                                    random_state = config_data['data_source']['random_state'], stratify = y)

    # data dump
    X_train_path = '../' + config_data['train_test_data']['directory'] + config_data['train_test_data']['X_train']
    y_train_path = '../' + config_data['train_test_data']['directory'] + config_data['train_test_data']['y_train']

    X_test_path = '../' + config_data['train_test_data']['directory'] + config_data['train_test_data']['X_test']
    y_test_path = '../' + config_data['train_test_data']['directory'] + config_data['train_test_data']['y_test']

    util.pickle_dump(X_train, X_train_path)
    util.pickle_dump(y_train, y_train_path)
    util.pickle_dump(X_test, X_test_path)
    util.pickle_dump(y_test, y_test_path)

    print('data_pipeline good')







