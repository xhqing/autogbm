#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py: auto_train func, the entry of autogbm
Created by xuhuaqing.
"""

import pandas as pd
import os
import time

class UnkownError(Exception):
    """UnkownError class"""
class NotsupportError(Exception):
    """NotsupportError class"""

def auto_train(train_set: pd.DataFrame, 
                label_name: str, task: str="binary", metric: str="auc", 
                models_save_path: str="autogbm_models/0", gpu: bool=False):
    """
    param `train_set`: 
        1. should be pandas dataframe;
        2. it's columns including `label_name`;
        3. all columns' name should be string;
        4. `train_set.dtypes` must be int or float
    
    param `label_name`: 
        1. the type of `label_name` must be string; (eq to param `train_set`'s 3.)
        2. it must be in the `train_set`'s columns (eq to param `train_set`'s 2.)

    param `task`:
        1. the type of `task` must be string;
        2. it is in ["binary","regression"];
        
    param `metric`:
        1. the type of `metric` must be string;
        2. the defult value is "auc";
        3. if `task` is "binary", `metric` in ["acc", "auc", "recall", "precission", "f1-score"],
            if `task` is "regression" `metric` in ["mse", "mae"]

    param `models_save_path`:
        1. the type of `models_save_path` must be string;
        2. defult value is "./autogbm_models/0", if it's exit then "./autogbm_models/1", etc.

    param `gpu`:
        1. the type of `gpu` must be bool;
        2. the defult value of `gpu` is False;
    """

    if not isinstance(train_set, pd.DataFrame):
        raise TypeError("\033[01;31;01mThe data type of `train_set` must be pandas dataframe!\033[01;00;00m ")

    if label_name not in train_set:
        raise ValueError("\033[01;31;01mThe `label_name` must be in the columns of `train_set`!\033[01;00;00m ")

    for col in train_set:
        if not isinstance(col, str):
            raise TypeError(f"\033[01;31;01mAll columns name of `train_set` must be string, while {col} is not string type!\033[01;00;00m ")

    PANDAS_DTYPE_MAPPER = {'int8': 'int', 'int16': 'int', 'int32': 'int', 'int64': 'int',
                       'uint8': 'int', 'uint16': 'int', 'uint32': 'int', 'uint64': 'int',
                       'float16': 'float', 'float32': 'float', 'float64': 'float'}
    data_dtypes = train_set.dtypes
    if not all(dtype.name in PANDAS_DTYPE_MAPPER for dtype in data_dtypes):
        bad_fields = [train_set.columns[i] for i, dtype in
                      enumerate(data_dtypes) if dtype.name not in PANDAS_DTYPE_MAPPER]

        msg = """\033[01;31;01m`train_set.dtypes` for data must be int or float.
                Did not expect the data types in fields: """
        raise ValueError(msg + ', '.join(bad_fields) + "\033[01;31;01m")
                    
    if not isinstance(task, str):
        raise TypeError(f"\033[01;31;01mThe type of `task` must be string, while {task} is not string type!\033[01;31;01m")
    if task not in ["binary", "regression"]:
        raise ValueError(f"\033[01;31;01m`task` must be \"binary\" or \"regression\", while `task` is \"{task}\"!\033[01;31;01m")

    if not isinstance(metric, str):
        raise TypeError(f"\033[01;31;01mThe type of `metric` must be string, while {metric} is not string type!\033[01;31;01m")
    if task == "binary":
        support_metrics = ["acc", "auc", "recall", "precission", "f1-score"]
        if metric not in support_metrics:
            msg = """\033[01;31;01mCurrently, for binary task, only support metrics are: """
            msg += ", ".join(support_metrics)
            raise ValueError(msg+", "+f"while you input `metric` is \"{metric}!\"\033[01;31;01m")
    if task == "regression":
        support_metrics = ["mse", "mae"]
        if metric not in support_metrics:
            msg = """\033[01;31;01mCurrently, for regression task, only support metrics are: """
            msg += ", ".join(support_metrics)
            raise ValueError(msg+", "+f"while you input `metric` is {metric}!\033[01;31;01m")

    if not isinstance(models_save_path, str):
        raise TypeError(f"""\033[01;31;01mThe type of `models_save_path` must be string, 
                        while {models_save_path} is not string type!\033[01;31;01m""")
    
    current_abspath = os.path.abspath(".")
    dirs = models_save_path.split("/")
    if dirs[-1] != "0":
        if models_save_path[-1] == "/":
            models_save_path += "0"
        else:
            models_save_path += "/0"

    start_time = time.time()
    while os.path.exists(models_save_path):
        dirs = models_save_path.split("/")
        numstr = dirs[-1]
        models_save_path = "/".join(dirs[:-1])+"/"+str(int(numstr)+1)
        end_time = time.time()
        cost_time = end_time - start_time
        if cost_time > 60:
            raise TimeoutError(f"""\033[01;31;01mTimeout. Make dir(s) `{models_save_path}` failed! Please retry!\033[01;31;01m""")
    
    os.makedirs(models_save_path)
    if models_save_path[0] != "/":
        models_save_abspath = current_abspath + "/" + models_save_path
    else:
        models_save_abspath = models_save_path

    numstr = models_save_abspath.split("/")[-1]
    if os.path.exists(models_save_path):
        print(f"""Make dir(s) {models_save_path} succeed. Location: {models_save_abspath}""")
    else:
        raise UnkownError(f"""\033[01;31;01mMake dir(s) {models_save_path} failed!\033[01;31;01m""")
        
    if not isinstance(gpu, bool):
        raise TypeError("""\033[01;31;01m`gpu` value must be bool type!\033[01;31;01m""")
    if gpu:
        raise NotsupportError("""\033[01;31;01mSorry, not support gpu yet!\033[01;31;01m""")

    

