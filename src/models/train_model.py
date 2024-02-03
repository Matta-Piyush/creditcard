from sklearn.ensemble import RandomForestClassifier
import pandas as pd  
import yaml
import sys
import pathlib
import joblib

def save_model(model,path):
    joblib.dump(model,path+'/model.joblib')

def train_model(X,y,estimators,max_depth,seed):
    model=RandomForestClassifier(n_estimators=estimators,max_depth=max_depth,random_state=seed)
    model.fit(X,y)
    return model

def main():
    curr_dir=pathlib.Path(__file__)
    home_dir=curr_dir.parent.parent.parent
    params_file_path=home_dir.as_posix()+'/params.yaml'
    params=yaml.safe_load(open(params_file_path))['train']
    input_1=sys.argv[1]
    train_data_path=home_dir.as_posix()+input_1
    train_data=pd.read_csv(train_data_path)
    TARGET='Class'
    X=train_data.drop(TARGET,axis=1)
    y=train_data[TARGET]
    model=train_model(X,y,params['estimators'],params['max_depth'],params['seed'])
    input_2=sys.argv[2]
    model_path=home_dir.as_posix()+input_2
    pathlib.Path(model_path).mkdir(parents=True,exist_ok=True)
    save_model(model,model_path)

if __name__=="__main__":
    main()


    