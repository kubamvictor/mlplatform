import sys
import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import mlflow 
from mlflow.tracking import MlflowClient

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
client = MlflowClient()
run = client.create_run(get_experiment_id(experiment_name)) # run Start:

# Get system arguments
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

# Normal approach
print('Mean Absolute Error:', mae)  
print('Mean Squared Error:', mse)  
print('Root Mean Squared Error:', rmse)

# log parameters
client.log_param(run.info.run_id,"n_estimators",n_estimators)
client.log_param(run.info.run_id,"random_state",random_state)
client.log_param(run.info.run_id,"min_samples_leaf",min_samples_leaf)

# log metrics
client.log_metric(run.info.run_id,"mae",mae)
client.log_metric(run.info.run_id,"mse",mse)
client.log_metric(run.info.run_id,"rmse",rmse)

client.set_terminated(run.info.run_id) # Run end