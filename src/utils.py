# ../src/utils.py

import os
import pandas as pd

from typing import Optional
from sklearn.model_selection import train_test_split

from constants import Constant as c


# TODO: in the future, should move utils functions into other files based on context

def split_csv_file(test_size: float=0.3, random_state: Optional[int]=None, **kwargs) -> None:
    """Split raw csv data into train and test csv using sklearn's `train_test_split` function

    Args:
        `test_size` (`float`, optional): test split percent. Defaults to 0.3.
        `random_state` (`Optional[int]`, optional): random state integer. Defaults to None.
        `**kwargs` (optional): keyword arguments for the `train_test_split` function
    """
    pathname = os.path.join(c.RAW_PATH, "HIV.csv")
    data = pd.read_csv(pathname).reset_index()
    
    # split into train and test sets
    train, test = train_test_split(data, test_size=test_size, random_state=random_state, **kwargs)
    
    # save sets
    train.to_csv(c.TRAIN_PATH) # we will need the index column for oversampling
    test.to_csv(c.TEST_PATH, index=False)

def generate_oversampled_data() -> None:
    """Oversample the negative HIV classes to balance with positive classes
    """
    data = pd.read_csv(c.TRAIN_PATH)
    data.index = data['index'] # use index column to setup index
    start_index = data.iloc[0]['index']
    
    # get multiplier
    neg_vals = data['HIV_active'].value_counts()[0]
    pos_vals = data['HIV_active'].value_counts()[1]
    multiplier = int(neg_vals/pos_vals) - 1
    
    # replicate the positive classes
    rep_pos = [data[data['HIV_active'] == 1]] * multiplier
    
    # append and shuffle replicated data into dataset
    for rep in rep_pos:
        data = pd.concat([data, rep], ignore_index=True)
        
    data = data.sample(frac=1).reset_index(drop=True)
    
    # re-assign index based on saved start value
    index = range(start_index, start_index + data.shape[0])
    data.index = index
    data['index'] = index
    
    # save new dataset
    data.to_csv(c.TRAIN_OSP_PATH)
