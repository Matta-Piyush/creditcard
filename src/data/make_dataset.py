from sklearn.model_selection import train_test_split
import pathlib
import pandas as pd 
import yaml

def load_data(file_path):
    return pd.read_csv(file_path)

def split_data(data,split,seed):
    train_data,test_data=train_test_split(data,train_size=split,random_state=seed)
    return train_data,test_data

def save_data(data,loc):
    data.to_csv(loc,index=False)

def main():
    curr_dir=pathlib.Path(__file__)
    home_dir=curr_dir.parent.parent.parent
    params_file_path=home_dir.as_posix()+"/params.yaml"
    params=yaml.safe_load(open(params_file_path))['make_dataset']
    data_path=home_dir.as_posix()+"/data/raw/creditcard.csv"
    output_path=home_dir.as_posix()+"/data/processed"
    data=load_data(data_path)
    train,test=split_data(data,params['train_split'],params['seed'])
    for _ in ['train','test']:
        save_data(data=_ ,loc=(output_path+"/"+_+".csv"))
if __name__=="__main__":
    main()




