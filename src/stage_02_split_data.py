from src.utils.all_utils import read_yaml,create_directory,save_local_df
import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def split_and_save(config_path,params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    # save dataset in the local directory
    # create path to directory : artifacts/raw_local_dir/raw_local_file/data.csv
    artifacts_dir = config['artifacts']['artifacts_dir']
    raw_local_dir = config['artifacts']['raw_local_dir']
    raw_local_file = config['artifacts']['raw_local_file']

    raw_local_dir_path = os.path.join(artifacts_dir,raw_local_dir,raw_local_file)

    df = pd.read_csv(raw_local_dir_path, sep=",")

    # get params from parama.yaml
    split_ratio = params['base']['test_size']
    random_state = params['base']['random_state']
    # splitting of data
    train,test = train_test_split(df,train_size=split_ratio,random_state=random_state)

    # get path of  split data
    split_data_dir = config['artifacts']['split_data_dir']

    # create a dir to save split data
    create_directory([os.path.join(artifacts_dir,split_data_dir)])

    # path for train and test data
    train_data_filename = config['artifacts']['train']
    test_data_filename = config['artifacts']['test']

    # train and tes datapath
    train_data_path = os.path.join(artifacts_dir,split_data_dir,train_data_filename)
    test_data_path = os.path.join(artifacts_dir,split_data_dir,test_data_filename)

    # save dataframe
    for data,data_path in (train,train_data_path),(test, test_data_path):
       save_local_df(data,data_path)
    




if __name__ == "__main__":
    # create a ArgumentParser object
    args = argparse.ArgumentParser()

    # add a argument --config on cli
    args.add_argument("--config",'-c',default='config/config.yaml')
    args.add_argument("--params",'-p',default='params.yaml')

    # call the method
    parsed_args = args.parse_args()

    split_and_save(config_path=parsed_args.config, params_path=parsed_args.params)