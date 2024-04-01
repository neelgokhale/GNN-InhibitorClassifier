# ../src/constants.py

import os


class Constant:
    # paths and urls
    DATASET_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv"
    ROOT_DATA_PATH = "data"
    RAW_PATH = os.path.join(ROOT_DATA_PATH, "raw")
    PROCESSED_PATH = os.path.join(ROOT_DATA_PATH, "processed")
    TRAIN_PATH = os.path.join(RAW_PATH, "HIV_train.csv")
    TRAIN_OSP_PATH = os.path.join(RAW_PATH, "HIV_train_osp.csv")
    TEST_PATH = os.path.join(RAW_PATH, "HIV_test.csv")
    