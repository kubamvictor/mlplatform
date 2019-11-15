import os
import sys
import pandas as pd 
import numpy as np 
import sklearn

def get_cmd_args():
    """
    Function to read command line arguments
    """
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        print("Error! No data file")

    return data_file


def write_data(table, filename):
    if not os.path.exists('./data/splitter'):
        print("creating data/splitter directory")
        os.makedirs('./data/splitter')

    print("Writing to ./data/splitter/{}".format(filename))
    table.to_csv('./data/splitter/' + filename, index=False,sep=",")

def main():
    """
    """
    raw_data_file = get_cmd_args()
    print("Loading raw data file...")
    raw_df = pd.read_csv(raw_data_file,sep=",")
    print("Loading complete")
    msk = np.random.rand(len(raw_df)) < 0.8
    print("Finished train, test splitting")
    train_df = raw_df[msk]
    test_df = raw_df[~msk]
    # Write train and test data to file
    print("Writing train dataset with {} columns and {} rows to file".format(train_df.shape[1],train_df.shape[0]))
    write_data(train_df,'train.csv')
    print("Writing training data complete")
    print("Writing test dataset with {} columns and {} rows to file".format(test_df.shape[1],test_df.shape[0]))
    write_data(test_df,'test.csv')

if __name__ == "__main__":
    main()