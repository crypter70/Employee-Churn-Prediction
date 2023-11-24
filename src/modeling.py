import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import utils as util
import joblib


# load_data
def load_data(config_data: dict, file_name: str) -> pd.DataFrame:
    try:
        PATH = config_data['train_test_data']['directory'] + file_name
        file_load = util.pickle_load(PATH)
    except:
        PATH = '../' + config_data['train_test_data']['directory'] + file_name
        file_load = util.pickle_load(PATH)
    return file_load


# train model
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    param = config_data['final_model']['parameter']
    dt = DecisionTreeClassifier(**param)
    dt.fit(X_train, y_train)
    return dt


# eval model
def evaluation_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_test_pred = model.predict(X_test)
    report = classification_report(y_true = y_test, y_pred = y_test_pred)
    print(report)


# dump model
def dump_model(to_dump, model_name):
    try:
        joblib.dump(to_dump, config_data['final_model']['model_directory'] + model_name)
    except:
        joblib.dump(to_dump, '../' + config_data['final_model']['model_directory'] + model_name)


if __name__ == "__main__":

    # load configuration file
    config_data = util.load_config()

    X_train = load_data(config_data, 'X_train_feng.pkl')
    y_train = load_data(config_data, 'y_train_feng.pkl')
    X_test = load_data(config_data, 'X_test_feng.pkl')
    y_test = load_data(config_data, 'y_test_feng.pkl')

    X_train = X_train['Oversampling']
    y_train = y_train['Oversampling']

    final_model = train_model(X_train, y_train)
    evaluation_model(final_model, X_test, y_test)
    dump_model(final_model, 'DecisionTreeClassifier.pkl')

    print('modeling good')