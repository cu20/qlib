# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import copy
import os
import shutil
import time
from pathlib import Path
from pprint import pprint

import numpy as np
import torch


import pandas as pd
import fire
import sys
import pickle

DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.append(str(DIRNAME.parent / "baseline"))
sys.path.append(str(DIRNAME.parent.parent.parent))

import qlib
from qlib.model.meta.task import MetaTask
from qlib.contrib.meta.data_selection.model import MetaModelDS
from qlib.contrib.meta.data_selection.dataset import InternalData, MetaDatasetDS
from qlib.data.dataset.handler import DataHandlerLP
from qlib import auto_init
from qlib.model.trainer import TrainerR
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.tests.data import GetData

from rolling_benchmark import RollingBenchmark  # NOTE: sys.path is changed for import RollingBenchmark


class DDGDA:
    """
    please run `python workflow.py run_all` to run the full workflow of the experiment

    **NOTE**
    before running the example, please clean your previous results with following command
    - `rm -r mlruns`
    """

    def __init__(
        self,
        data_dir="cn_data",
        market="csi300",
        sim_task_model="linear",
        forecast_model="linear",
        alpha="158",
        alpha2='158',
        rank_label=True,
        step=20,
        horizon=1,
    ):
        self.data_dir = data_dir
        self.market = market
        if data_dir == "cn_data":
            GetData().qlib_data(target_dir="~/.qlib/qlib_data/cn_data", exists_skip=True)
            auto_init()
        else:
            qlib.init(
                provider_uri="~/.qlib/qlib_data/" + data_dir, region="us" if self.data_dir == "us_data" else "cn",
            )
        self.step = step
        # NOTE:
        # the horizon must match the meaning in the base task template
        self.horizon = horizon
        self.sim_task_model = sim_task_model  # The model to capture the distribution of data.
        self.forecast_model = forecast_model  # downstream forecasting models' type
        self.alpha = alpha
        self.alpha2 = alpha2
        self.tag = ""
        self.seed = 43
        self.rank_label = rank_label

    @property
    def exp_name(self):
        return f"{self.data_dir}_{self.market}_{self.alpha}_rank{self.rank_label}_s{self.step}_h{self.horizon}"

    @property
    def meta_exp_name(self):
        return f"DDG-DA_{self.exp_name}_{self.alpha2}_{self.seed}"

    def get_feature_importance(self):
        # this must be lightGBM, because it needs to get the feature importance
        rb = RollingBenchmark(
            data_dir=self.data_dir,
            market=self.market,
            model_type="gbdt",
            alpha=self.alpha,
            horizon=self.horizon,
            step=self.step,
            init_data=False,
            rank_label=self.rank_label,
        )
        task = rb.basic_task()

        with R.start(experiment_name="feature_importance"):
            model = init_instance_by_config(task["model"])
            dataset = init_instance_by_config(task["dataset"])
            model.fit(dataset)

        fi = model.get_feature_importance()

        # Because the model use numpy instead of dataframe for training lightgbm
        # So the we must use following extra steps to get the right feature importance
        df = dataset.prepare(segments=slice(None), col_set="feature", data_key=DataHandlerLP.DK_R)
        cols = df.columns
        fi_named = {cols[int(k.split("_")[1])]: imp for k, imp in fi.to_dict().items()}

        return pd.Series(fi_named)

    @property
    def _dataframe_path(self):
        return DIRNAME / (self.exp_name + "_fea_label_df.pkl")

    @property
    def _handler_path(self):
        return DIRNAME / (self.exp_name + "_handler_proxy.pkl")

    def dump_data_for_proxy_model(self):
        """
        Dump data for training src model.
        The src model will be trained upon the proxy forecasting model.
        This dataset is for the proxy forecasting model.
        """
        rb = RollingBenchmark(
            data_dir=self.data_dir,
            market=self.market,
            model_type=self.sim_task_model,
            alpha=self.alpha,
            horizon=self.horizon,
            step=self.step,
            rank_label=self.rank_label,
            init_data=False,
        )
        task = rb.basic_task()
        dataset = init_instance_by_config(task["dataset"])
        prep_ds = dataset.prepare(slice(None), col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)

        topk = 30
        fi = self.get_feature_importance()
        col_selected = fi.nlargest(topk)
        feature_df = prep_ds["feature"]
        label_df = prep_ds["label"]

        feature_selected = feature_df.loc[:, col_selected.index]

        feature_selected = feature_selected.groupby("datetime").apply(lambda df: (df - df.mean()).div(df.std()))
        feature_selected = feature_selected.fillna(0.0)

        df_all = {
            "label": label_df.reindex(feature_selected.index),
            "feature": feature_selected,
        }
        df_all = pd.concat(df_all, axis=1)
        df_all.to_pickle(self._dataframe_path)

        # dump data in handler format for aligning the interface
        handler = DataHandlerLP(
            data_loader={
                "class": "qlib.data.dataset.loader.StaticDataLoader",
                "kwargs": {"config": self._dataframe_path},
            }
        )
        handler.to_pickle(self._handler_path, dump_all=True)

    @property
    def _internal_data_path(self):
        return (
            DIRNAME
            / f"{self.data_dir}_{self.market}_{self.alpha2}_rank{self.rank_label}_s{self.step}_h{self.horizon}internal_data.pkl"
        )

    def dump_meta_ipt(self):
        """
        Dump data for training src model.
        This function will dump the input data for src model
        """
        # According to the experiments, the choice of the model type is very important for achieving good results
        rb = RollingBenchmark(
            data_dir=self.data_dir,
            market=self.market,
            model_type=self.sim_task_model,
            alpha=self.alpha,
            horizon=self.horizon,
            step=self.step,
            rank_label=self.rank_label,
            init_data=False,
        )
        sim_task = rb.basic_task()

        if self.sim_task_model == "gbdt":
            sim_task["model"].setdefault("kwargs", {}).update({"early_stopping_rounds": None, "num_boost_round": 150})

        exp_name_sim = "sim_" + self.exp_name + "_43"
        internal_data = InternalData(sim_task, self.step, exp_name=exp_name_sim)
        internal_data.setup(trainer=TrainerR)

        with self._internal_data_path.open("wb") as f:
            pickle.dump(internal_data, f)

    def train_meta_model(self, seed=43):
        """
        training a src model based on a simplified linear proxy model;
        """

        # 1) leverage the simplified proxy forecasting model to train src model.
        # - Only the dataset part is important, in current version of src model will integrate the
        rb = RollingBenchmark(
            data_dir=self.data_dir,
            market=self.market,
            model_type=self.sim_task_model,
            alpha=self.alpha,
            horizon=self.horizon,
            step=self.step,
            rank_label=self.rank_label,
            init_data=False,
        )
        sim_task = rb.basic_task()
        proxy_forecast_model_task = {
            # "model": "qlib.contrib.model.linear.LinearModel",
            "dataset": {
                "class": "qlib.data.dataset.DatasetH",
                "kwargs": {
                    "handler": f"file://{self._handler_path.absolute()}",
                    "segments": {
                        "train": ("2008-01-01", "2010-12-31"),
                        "test": ("2011-01-01", sim_task["dataset"]["kwargs"]["segments"]["test"][1],),
                    },
                },
            },
            # "record": ["qlib.workflow.record_temp.SignalRecord"]
        }
        # the proxy_forecast_model_task will be used to create src tasks.
        # The test date of first task will be 2011-01-01. Each test segment will be about 20days
        # The tasks include all training tasks and test tasks.

        # 2) preparing src dataset
        kwargs = dict(
            task_tpl=proxy_forecast_model_task,
            step=self.step,
            segments=str(sim_task["dataset"]["kwargs"]["segments"]["valid"][-1]),
            # keep test period consistent with the dataset yaml
            trunc_days=1 + self.horizon,
            hist_step_n=30,
            fill_method="max",
            rolling_ext_days=0,
        )
        # NOTE:
        # the input of src model (internal data) are shared between proxy model and final forecasting model
        # but their task test segment are not aligned! It worked in my previous experiment.
        # So the misalignment will not affect the effectiveness of the method.
        with self._internal_data_path.open("rb") as f:
            internal_data = pickle.load(f)
        md = MetaDatasetDS(exp_name=internal_data, **kwargs)

        # 3) train and logging src model
        with R.start(experiment_name=self.meta_exp_name):
            R.log_params(**kwargs)
            mm = MetaModelDS(
                step=self.step, hist_step_n=kwargs["hist_step_n"], lr=0.001, max_epoch=100, seed=seed
            )
            mm.fit(md)
            R.save_objects(model=mm)

    @property
    def _task_path(self):
        return DIRNAME / (self.exp_name + "tasks.pkl")

    def meta_inference(self):
        """
        Leverage meta-model for inference:
        - Given
            - baseline tasks
            - input for meta model(internal data)
            - meta model (its learnt knowledge on proxy forecasting model is expected to transfer to normal forecasting model)
        """
        # 1) get meta model
        alpha = self.alpha
        self.alpha = 158
        # self.alpha2 = 158
        exp = R.get_exp(experiment_name=self.meta_exp_name)
        self.alpha = alpha
        rec = exp.list_recorders(rtype=exp.RT_L)[0]
        meta_model: MetaModelDS = rec.load_object("model")

        # 2)
        # we are transfer to knowledge of meta model to final forecasting tasks.
        # Create MetaTaskDataset for the final forecasting tasks
        # Aligning the setting of it to the MetaTaskDataset when training Meta model is necessary

        # 2.1) get previous config
        param = rec.list_params()
        trunc_days = 1 + self.horizon
        step = int(param["step"])
        hist_step_n = int(param["hist_step_n"])
        fill_method = param.get("fill_method", "max")

        rb = RollingBenchmark(
            data_dir=self.data_dir,
            market=self.market,
            model_type=self.forecast_model,
            alpha=self.alpha,
            horizon=self.horizon,
            step=self.step,
            rank_label=self.rank_label,
            init_data=False,
        )
        task_l = rb.create_rolling_tasks()

        # 2.2) create meta dataset for final dataset
        kwargs = dict(
            task_tpl=task_l,
            step=step,
            segments=0.0,  # all the tasks are for testing
            trunc_days=trunc_days,
            hist_step_n=hist_step_n,
            fill_method=fill_method,
            task_mode=MetaTask.PROC_MODE_TRANSFER,
        )

        with self._internal_data_path.open("rb") as f:
            internal_data = pickle.load(f)
        mds = MetaDatasetDS(exp_name=internal_data, **kwargs)

        # 3) meta model make inference and get new qlib task
        new_tasks = meta_model.inference(mds)
        with self._task_path.open("wb") as f:
            pickle.dump(new_tasks, f)

    def train_and_eval_tasks(self):
        """
        Training the tasks generated by meta model
        Then evaluate it
        """
        with self._task_path.open("rb") as f:
            tasks = copy.deepcopy(pickle.load(f))
        rb = RollingBenchmark(
            data_dir=self.data_dir,
            market=self.market,
            model_type=self.forecast_model,
            alpha=self.alpha,
            horizon=self.horizon,
            step=self.step,
            rank_label=self.rank_label,
            init_data=False,
        )
        rb.tag = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime()) + "_DDG-DA"
        rec = rb.run_all(tasks)
        return rec, rb.COMB_EXP

    def run_all(self):
        # 1) file: handler_proxy.pkl
        self.dump_data_for_proxy_model()
        # 2)
        # file: internal_data_s20.pkl
        # mlflow: data_sim_s20, models for calculating meta_ipt
        self.dump_meta_ipt()
        # 3) meta model will be stored in `DDG-DA`
        self.train_meta_model()
        # 4) new_tasks are saved in "tasks_s20.pkl" (reweighter is added)
        self.meta_inference()
        # 5) load the saved tasks and train model
        self.train_and_eval_tasks()

    def run_offline(self):
        train_time = []
        start_time = time.time()
        self.dump_data_for_proxy_model()
        self.dump_meta_ipt()
        data_time = time.time() - start_time
        print(time.time() - start_time)
        for i in range(0, 10):
            self.seed = 43 + i
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            start_time = time.time()
            self.train_meta_model(seed=self.seed)
            train_time.append(time.time() - start_time)

        train_time = np.array(train_time)
        print(f"Time cost: {train_time.mean() + data_time}")

    def run_online(self):
        all_metrics = {
            k: []
            for k in [
                'mse', 'mae',
                "IC",
                "ICIR",
                "Rank IC",
                "Rank ICIR",
                # '1day.excess_return_without_cost.annualized_return',
                # '1day.excess_return_without_cost.information_ratio',
                # '1day.excess_return_without_cost.max_drawdown',
                "1day.excess_return_with_cost.annualized_return",
                "1day.excess_return_with_cost.information_ratio",
                "1day.excess_return_with_cost.max_drawdown",
            ]
        }

        test_time = []
        for i in range(0, 10):
            self.seed = 43 + i
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            start_time = time.time()
            self.meta_inference()
            self.tag = str(start_time)
            rec, experiment_name = self.train_and_eval_tasks()
            test_time.append(time.time() - start_time)
            exp = R.get_exp(experiment_name=experiment_name)
            rec = exp.list_recorders(rtype=exp.RT_L)[0]
            metrics = rec.list_metrics()
            for k in all_metrics.keys():
                all_metrics[k].append(metrics[k])
            pprint(all_metrics)

        with R.start(experiment_name="final_" + self.exp_name):
            R.save_objects(all_metrics=all_metrics)
            test_time = np.array(test_time)
            print(f"Time cost: {test_time.mean()}")
            R.log_metrics(test_time=test_time.mean())
            res = {}
            for k in all_metrics.keys():
                v = np.array(all_metrics[k])
                res[k] = [v.mean(), v.std()]
                R.log_metrics(**{"final_" + k: res[k][0]})
                R.log_metrics(**{"final_" + k + "_std": res[k][1]})
            pprint(res)


if __name__ == "__main__":
    # GetData().qlib_data(exists_skip=True)
    # auto_init()
    fire.Fire(DDGDA)
