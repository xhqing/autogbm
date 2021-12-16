#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py: auto_train func, the entry of autogbm
Created by xuhuaqing.
"""

import pandas as pd
import os
import time
from copy import deepcopy
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder

from autogbm.scores import nauc, acc
from autogbm.eda import AutoEDA
from autogbm.util import print_formatted_json

class UnkownError(Exception):
    """UnkownError class"""
class NotsupportError(Exception):
    """NotsupportError class"""
class LabelError(Exception):
    """LabelError class"""

class GBMModel:
    """GBMModel class"""
    def __init__(self, max_epoch):
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.label_name = None
        self.train_loop_num = 0
        self.is_multi_label = None
        self.imp_cols = None
        self.unknow_cols = None
        self.lgb_info = dict()
        self.pre_increament_preds = True
        self.first_preds = False
        self.done_training = False
        self.max_epoch = max_epoch
        self.auto_eda = AutoEDA()

    def fit(self, trainset, label_name, remaining_time_budget=None):
        self.train_loop_num += 1

        if self.train_loop_num == 1:
            sample_num = 500
        elif self.train_loop_num == 2:
            sample_num = 1000
        elif self.train_loop_num == 3:
            sample_num = 2000
        elif self.train_loop_num == 4:
            sample_num = 3000
        else:
            sample_num = 500*2**(self.train_loop_num-3)

        if sample_num <= trainset.shape[0]:
            self.X_train = deepcopy(trainset.drop(label_name, axis=1).loc[:sample_num-1,:])
            self.Y_train = deepcopy(trainset.loc[:sample_num-1, label_name])
            if not self.label_name:
                self.label_name = label_name
        else:
            self.pre_increament_preds = False
            if self.train_loop_num == 1:
                self.first_preds = True
            self.train_loop_num = 1

    def predict(self, x_test: pd.DataFrame, remaining_time_budget=None):
        if self.pre_increament_preds or self.first_preds:
            if self.X_test is None: self.X_test = x_test
            preds = self.simple_lgb(self.X_train, self.Y_train, self.X_test)
            if self.first_preds:
                self.first_preds = False
                self.train_loop_num = 0
        else:
            if self.train_loop_num == 1:
                self.X_test.index = -self.X_test.index - 1
                main_df = pd.concat([self.X_train, self.X_test], axis=0)
                del self.X_train, self.X_test

                eda_info = self.auto_eda.get_info(main_df)
                eda_info['is_multi_label'] = self.is_multi_label
                print_formatted_json(eda_info)
                import pdb
                pdb.set_trace()
                self.data_space = TabularDataSpace(self.metadata_info, eda_info, main_df, self.Y_train, self.lgb_info)
                self.model_space = TabularModelSpace(self.metadata_info, eda_info)
                self.explore = Explore(self.metadata_info, eda_info, self.model_space, self.data_space)
            print('time', remaining_time_budget)
            self.explore.explore_space(train_loop_num=self.train_loop_num, time_remain=remaining_time_budget)
            preds = self.explore.predict()

        return preds

    def simple_lgb(self, X, y, test_x):

        self.params = {
            "boosting_type": "gbdt",
            "objective": "multiclass",
            'num_class': y.nunique(),
            "metric": "multi_logloss",
            "verbosity": -1,
            "seed": 2020,
            "num_threads": 4,
        }

        self.hyperparams = {
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'max_bin': 110,
            'subsample': 1,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'min_child_weight': 0.001,
            'min_split_gain': 0.02,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            "learning_rate": 0.1
        }
        num_boost_round = 10

        print('simple lgb predict train_loop_num:', self.train_loop_num)
        if self.train_loop_num == 1:
            if X.shape[1] > 500:
                pass
                # self.sample_cols = list(set(X.columns))[::2]
                # self.unknow_cols = [col for col in X.columns if col not in self.sample_cols]
                # X = X[self.sample_cols]
                # test_x = test_x[self.sample_cols]
            if self.is_multi_label:
                pass
                # self.params['num_class'] = 2
                # all_preds = []
                # for cls in range(y.shape[1]):
                #     cls_y = y[:, cls]
                #     data = lgb.Dataset(X, cls_y)
                #     self.models[cls] = lgb.train({**self.params, **self.hyperparams}, data)
                #     preds = self.models[cls].predict(test_x)
                #     all_preds.append(preds[:, 1])
                # preds = np.stack(all_preds, axis=1)
            else:
                lgb_train = lgb.Dataset(X, y.values)
                self.model = lgb.train({**self.params, **self.hyperparams}, train_set=lgb_train, num_boost_round=num_boost_round)
                preds = self.model.predict(test_x)
            self.log_feat_importances()
        else:
            num_boost_round += self.train_loop_num * 5
            num_boost_round = min(40, num_boost_round)

            if self.is_multi_label:
                pass
                # models = {}
                # all_preds = []
                # for cls in range(y.shape[1]):
                #     cls_y = y[:, cls]
                #     data = lgb.Dataset(X[self.imp_cols], cls_y)
                #     models[cls] = lgb.train({**self.params, **self.hyperparams}, data)
                #     preds = models[cls].predict(test_x[self.imp_cols])
                #     all_preds.append(preds[:, 1])
                # preds = np.stack(all_preds, axis=1)
            else:
                lgb_train = lgb.Dataset(X[self.imp_cols], y)
                model = lgb.train({**self.params, **self.hyperparams}, train_set=lgb_train, num_boost_round=num_boost_round)
                preds = model.predict(test_x[self.imp_cols])
        return preds

    def log_feat_importances(self):
        if not self.is_multi_label:
            importances = pd.DataFrame({'features': [i for i in self.model.feature_name()],
                                                                    'importances': self.model.feature_importance("gain")})
        else:
            importances = pd.DataFrame({'features': [i for i in self.models[0].feature_name()],
                                                                    'importances': self.models[0].feature_importance("gain")})

        importances.sort_values('importances', ascending=False, inplace=True)

        importances = importances[importances['importances'] > 0]
        size = int(len(importances)*0.8)
        if self.imp_cols is None:
            if self.unknow_cols is not None:
                self.imp_cols = self.unknow_cols + importances['features'].tolist()
            else:
                self.imp_cols = importances['features'].tolist()
        else:
            self.imp_cols = importances['features'].tolist()
        self.lgb_info['imp_cols'] = self.imp_cols


def pandas_dtype_check(dataset: pd.DataFrame):
    bad_fields = None
    PANDAS_DTYPE_MAPPER = {'int8': 'int', 'int16': 'int', 'int32': 'int', 'int64': 'int',
                       'uint8': 'int', 'uint16': 'int', 'uint32': 'int', 'uint64': 'int',
                       'float16': 'float', 'float32': 'float', 'float64': 'float'}
    data_dtypes = dataset.dtypes
    if not all(dtype.name in PANDAS_DTYPE_MAPPER for dtype in data_dtypes):
        bad_fields = [dataset.columns[i] for i, dtype in
                      enumerate(data_dtypes) if dtype.name not in PANDAS_DTYPE_MAPPER]
    return bad_fields

def auto_train(train_set: pd.DataFrame, test_set: pd.DataFrame, label_name: str, task: str="multi", metric: str="nauc", 
                models_save_path: str="autogbm_models/0", gpu: bool=False, time_budget: int=1200, max_epoch: int=50):
    """
    param `train_set` and `test_set`: 
        1. should be pandas dataframe;
        2. it's columns including `label_name`;
        3. all columns' name should be string;
        4. `train_set.dtypes` and `test_set.dtypes` must be int or float
        5. the index must be start from 0,1,2,3,...
    
    param `label_name`:
        1. the type of `label_name` must be string;
        2. it must be in the `train_set`'s and `test_set`'s columns

    param `task`:
        1. the type of `task` must be string;
        2. it is in ["binary", "multi", "regression"];
        
    param `metric`:
        1. the type of `metric` must be string;
        2. the defult value is "auc";
        3. if `task` is "binary", `metric` in ["acc", "auc", "recall", "precission", "f1-score"],
            if `task` is "regression" `metric` in ["mse", "mae"],
            if `task` is "multi" `metric` in ["nauc","acc"]

    param `models_save_path`:
        1. the type of `models_save_path` must be string;
        2. defult value is "./autogbm_models/0", if it's exit then "./autogbm_models/1", etc.

    param `gpu`:
        1. the type of `gpu` must be bool;
        2. the defult value of `gpu` is False;

    param `time_budget`:
        the uplimit training time, the default value is 1200 seconds(20min). 

    param `max_epoch`:
        the max number of training round, default is 50 rounds.
    """

    if not isinstance(train_set, pd.DataFrame):
        raise TypeError("\033[01;31;01mThe data type of `train_set` must be pandas dataframe!\033[01;00;00m ")
    
    train_set.reset_index(drop=True, inplace=True)
    test_set.reset_index(drop=True, inplace=True)

    if label_name not in train_set or label_name not in test_set:
        raise ValueError("\033[01;31;01mThe `label_name` must be in the columns of `train_set` and `test_set`!\033[01;00;00m ")

    for col in train_set:
        if not isinstance(col, str):
            raise TypeError(f"\033[01;31;01mAll columns name of `train_set` must be string, while {col} is not string type!\033[01;00;00m ")

    bad_fields = pandas_dtype_check(train_set)
    if bad_fields:
        msg = """\033[01;31;01m`train_set.dtypes` for data must be int or float.
                    Did not expect the data types in fields: """
        raise ValueError(msg + ', '.join(bad_fields) + "\033[01;31;01m")

    for col in test_set:
        if not isinstance(col, str):
            raise TypeError(f"\033[01;31;01mAll columns name of `test_set` must be string, while {col} is not string type!\033[01;00;00m ")

    bad_fields = pandas_dtype_check(test_set)
    if bad_fields:
        msg = """\033[01;31;01m`test_set.dtypes` for data must be int or float.
                    Did not expect the data types in fields: """
        raise ValueError(msg + ', '.join(bad_fields) + "\033[01;31;01m")
                    
    if not isinstance(task, str):
        raise TypeError(f"\033[01;31;01mThe type of `task` must be string, while {task} is not string type!\033[01;31;01m")
    if task not in ["multi", "regression"]:
        raise ValueError(f"\033[01;31;01m`task` must be \"multi\" or \"regression\", while `task` is \"{task}\"!\033[01;31;01m")

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
    if task == "multi":
        support_metrics = ["nauc","acc"]
        if metric not in support_metrics:
            msg = """\033[01;31;01mCurrently, for multi task, only support metrics are: """
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

    if task == "binary":
        task = "multi"
        # label_nunique = train_set[label_name].nunique()
        # if label_nunique > 2:
        #     raise LabelError(f"""\033[01;31;01mIt's binary task, but label have {label_nunique} diffirent kinds of values!\033[01;31;01m""")
        # if label_nunique < 2:
        #     raise LabelError(f"""\033[01;31;01mIt's binary task, but label have only {label_nunique} kind of value!\033[01;31;01m""")
        # if label_nunique != test_set[label_name].nunique():
        #     raise LabelError(f"""\033[01;31;01mtrain_set label nunique not the same as test_set label nunique!\033[01;31;01m""")
    elif task == "regression":
        raise NotsupportError("""\033[01;31;01mSorry, not support regression task yet!\033[01;31;01m""")
    elif task == "multi":
        label_nunique = train_set[label_name].nunique()
        if label_nunique < 2:
            raise LabelError(f"""\033[01;31;01mIt's multi task, but label have only {label_nunique} kind of value!\033[01;31;01m""")
        if label_nunique != test_set[label_name].nunique():
            raise LabelError(f"""\033[01;31;01mtrain_set label nunique not the same as test_set label nunique!\033[01;31;01m""")
    
        model = GBMModel(max_epoch=max_epoch)
        start_time = int(time.time())
        for i in range(max_epoch):
            remaining_time_budget = start_time + time_budget - int(time.time())
            model.fit(trainset=deepcopy(train_set), label_name=label_name, remaining_time_budget=remaining_time_budget)
            remaining_time_budget = start_time + time_budget - int(time.time())
            y_pred = model.predict(x_test=deepcopy(test_set.drop(label_name, axis=1)), remaining_time_budget=remaining_time_budget)

            ohe_y_test = OneHotEncoder(categories='auto').fit_transform([[ele] for ele in test_set[label_name]]).toarray()
            nauc_score = nauc(y_test=ohe_y_test, prediction=y_pred)
            acc_score = acc(y_test=ohe_y_test, prediction=y_pred)
            print("Epoch={}, evaluation: nauc_score={}, acc_score={}".format(i+1, nauc_score, acc_score))

            # if i+1 == 8: break



