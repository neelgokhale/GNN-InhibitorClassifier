# ../src/constants.py

import os


class Constant:
    # paths and urls
    DATASET_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv"
    ROOT_DATA_PATH = "data"
    RAW_PATH = "data/"
    PROCESSED_PATH = os.path.join(ROOT_DATA_PATH, "processed")
    TRAIN_FILENAME = "HIV_train.csv"
    TRAIN_OSP_FILENAME = "HIV_train_osp.csv"
    TEST_FILENAME = "HIV_test.csv"
    SYSTEM = "" # "apple" not used bc MPS not enabled for 'aten::scatter_reduce.two_out'
    
    # model
    MODEL_NAME = "gnn-inhibitor-detector"
    