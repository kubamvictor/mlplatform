import sys
import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import mlflow 
from mlflow.tracking import MlflowClient
import neptune

neptune.init('kubamvictor/sandbox',
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwaV9rZXkiOiJlNDM2MmMxMC1iNDk2LTQ2YWMtYTE2MC1hMGZkYzJhNjZjMGUifQ==')

# Set mlflow tracking uri
# remote_server_uri = "..."
# mlflow.set_tracking_uri(remote_server_uri)

# set experiment
experiment_name = "my-experiment"
mlflow.set_experiment(experiment_name)

def get_experiment_id(experiment_name):
    """
    function to retrive experiment id given experiment name
    """
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    try:
        exp_id = [exp.experiment_id for exp in client.list_experiments() if exp.name == experiment_name][0]
        print("Experiment found!")
        print("Returning experiment id.....")
        print("Experiment_id: {}".format(exp_id))
    except Exception as e:
        print("Experiment not found! {}".format(e))
        pass
    return exp_id

# mlflow client
# client = MlflowClient()
# run = client.create_run(get_experiment_id(experiment_name)) # run Start:

# Get system arguments

with mlflow.start_run():
    train_datafile, test_datafile = sys.argv[1],sys.argv[2] if len(sys.argv) > 2 else print(" No input for train and test files")

    train_df = pd.read_csv(train_datafile,sep=",")
    test_df = pd.read_csv(test_datafile)

    X_train = train_df.iloc[:,0:-1].values
    y_train= np.array(train_df.iloc[:,-1])

    X_test = test_df.iloc[:,0:-1].values
    y_test = np.array(test_df.iloc[:,-1])

    # Parameters
    n_estimators = 5
    random_state = 100
    min_samples_leaf = 5
    regressor = RandomForestRegressor(n_estimators = n_estimators, random_state = random_state, min_samples_leaf=min_samples_leaf)

    regressor.fit(X_train,y_train)

    y_pred = regressor.predict(X_test)

    mae =  metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred) 
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    print(sum([el for el in range(50000000)]))

    # Normal approach
    print('Mean Absolute Error:', mae)  
    print('Mean Squared Error:', mse)  
    print('Root Mean Squared Error:', rmse)

    # log parameters
    mlflow.log_param(key="n_estimators",value=n_estimators)
    mlflow.log_param(key = "random_state", value=random_state)
    mlflow.log_param(key="min_samples_leaf",value=min_samples_leaf)

    # log metrics
    mlflow.log_metric(key="mae",value=mae)
    mlflow.log_metric(key="mse",value=mse)
    mlflow.log_metric(key="rmse",value=rmse)

# client.set_terminated(run.info.run_id) 

#----------------------------------------
# Neptune
#----------------------------------------

PARAMS = {
    "n_estimators":5,
    "random_state":100,
    "min_samples_leaf":5
}

with neptune.create_experiment(name="ml_platforms",params=PARAMS):
    neptune.log_metric("mae",mae)
    neptune.log_metric("mse",mse)
    neptune.log_metric("rmse",rmse)