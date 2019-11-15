import os
import sys
from shutil import copyfile

import logging

data_file = sys.argv[1] if len(sys.argv) > 1 else "housing.csv"

def download_file(file_name):
    base_path = "./../../../datasets/"
    try:
        file_location = base_path+str(file_name)
    except Exception as e:
        print("Error! {}".format(e))

    # Check if file exists
    if os.path.exists("./data/raw/"+str(file_name)):
        print("File already exists.")
        pass
    else:    
        print("File found!")
        print("Downloading file ....")  

        if not os.path.exists("./data/raw/"):
            print("creating data/raw directory")
            os.makedirs("./data/raw/")

        copyfile(file_location,"./data/raw/"+str(file_name)) 
        print("File download complete.") 

if __name__ == "__main__":
    download_file(data_file)
    