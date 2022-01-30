from src.utils.all_utils import read_yaml,create_directory,save_local_df
import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import joblib

def train(config_path,params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    # save dataset in the local directory
    # create path to directory : artifacts/raw_local_dir/raw_local_file/data.csv
    artifacts_dir = config['artifacts']['artifacts_dir']
    split_data_dir = config['artifacts']['split_data_dir']

    
    train_data_filename = config['artifacts']['train']

    # create a path to read the data
    train_data_path = os.path.join(artifacts_dir,split_data_dir,train_data_filename)

    # read the train data
    train_data = pd.read_csv(train_data_path)

    # split train as X and Y
    train_y = train_data['quality']
    train_x = train_data.drop("quality",axis=1)

    # fetch model params from params.yaml
    alpha = params['model_params']['ElasticNet']['alpha']
    l1_ratio = params['model_params']['ElasticNet']['l1_ratio']
    random_state = params['model_params']['ElasticNet']['random_state']

    # define the model
    lr = ElasticNet(alpha=alpha,l1_ratio=l1_ratio,random_state=random_state)
    
    # fit the model
    lr.fit(train_x,train_y)

    # save the model
    model_dir = config['artifacts']['model_dir'] 
    model_filename = config['artifacts']['model_filename']
    model_dir = os.path.join(artifacts_dir,model_dir)
    create_directory([model_dir])
    model_path = os.path.join(model_dir,model_filename)
    joblib.dump(lr, model_path)



if __name__ == "__main__":
    # create a ArgumentParser object
    args = argparse.ArgumentParser()

    # add a argument --config on cli
    args.add_argument("--config",'-c',default='config/config.yaml')
    args.add_argument("--params",'-p',default='params.yaml')

    # call the method
    parsed_args = args.parse_args()

    train(config_path=parsed_args.config, params_path=parsed_args.params)