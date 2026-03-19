# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import Optional
import time
from pprint import pprint

import numpy as np
import pandas as pd
import torch

from pathlib import Path
import sys
DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.append(str(DIRNAME.parent.parent.parent))

import qlib
from qlib.data.dataset import Dataset, DataHandlerLP, TSDataSampler
from qlib.model.ens.ensemble import RollingEnsemble
from qlib.utils import init_instance_by_config
import fire
import yaml
import pandas as pd
from qlib import auto_init
from tqdm.auto import tqdm
from qlib.model.trainer import TrainerR
from qlib.log import get_module_logger
from qlib.utils.data import update_config
from qlib.workflow import R, Experiment
from qlib.tests.data import GetData

from qlib.workflow.task.gen import task_generator, RollingGen
from qlib.workflow.task.collect import RecorderCollector
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord


class RollingBenchmark:
    def __init__(
        self,
        data_dir="cn_data",
        market="csi300",
        init_data=True,
        model_type="linear",
        step=20,
        alpha="158",
        horizon=1,
        rank_label=True,
        h_path: Optional[str] = None,
        train_start: Optional[str] = None,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None,
        task_ext_conf: Optional[dict] = None,
        reload_tag: Optional[list] = None,
    ) -> None:
        self.reload_tags = reload_tag
        self.data_dir = data_dir
        self.market = market
        if init_data:
            if data_dir == "cn_data":
                GetData().qlib_data(target_dir="~/.qlib/qlib_data/cn_data", exists_skip=True)
                auto_init()
            else:
                qlib.init(
                    provider_uri="~/.qlib/qlib_data/" + data_dir, region="us" if self.data_dir == "us_data" else "cn",
                )
        self.step = step
        self.horizon = horizon
        # self.rolling_exp = rolling_exp
        self.model_type = model_type
        self.h_path = h_path
        self.train_start = train_start
        self.test_start = test_start
        self.test_end = test_end
        self.logger = get_module_logger("RollingBenchmark")
        self.task_ext_conf = task_ext_conf
        self.rank_label = rank_label
        self.alpha = alpha
        self.tag = ""
        if self.data_dir == "us_data":
            self.benchmark = "^gspc"
        elif self.market == "csi500":
            self.benchmark = "SH000905"
        elif self.market == "csi100":
            self.benchmark = "SH000903"
        else:
            self.benchmark = "SH000300"

    @property
    def rolling_exp(self):
        return f"rolling_{self.data_dir}_{self.market}_{self.model_type}_alpha{self.alpha}_h{self.horizon}_step{self.step}_rank{self.rank_label}_{self.tag}"

    @property
    def COMB_EXP(self):
        return "final_" + self.rolling_exp

    def basic_task(self):
        """For fast training rolling"""
        if self.model_type == "gbdt":
            conf_path = (
                DIRNAME.parent.parent
                / "benchmarks"
                / "LightGBM"
                / "workflow_config_lightgbm_Alpha{}.yaml".format(self.alpha)
            )
            # dump the processed data on to disk for later loading to speed up the processing
            filename = "lightgbm_alpha{}_handler_horizon{}.pkl".format(self.alpha, self.horizon)
        elif self.model_type == "linear":
            conf_path = (
                DIRNAME.parent.parent
                / "benchmarks"
                / "Linear"
                / "workflow_config_linear_Alpha{}.yaml".format(self.alpha)
            )
            # dump the processed data on to disk for later loading to speed up the processing
            filename = "linear_alpha{}_handler_horizon{}.pkl".format(self.alpha, self.horizon)
        elif self.model_type == "MLP":
            conf_path = (
                DIRNAME.parent.parent / "benchmarks" / "MLP" / "workflow_config_mlp_Alpha{}.yaml".format(self.alpha)
            )
            # dump the processed data on to disk for later loading to speed up the processing
            filename = "MLP_alpha{}_handler_horizon{}.pkl".format(self.alpha, self.horizon)
        else:
            conf_path = (
                DIRNAME.parent.parent
                / "benchmarks"
                / self.model_type
                / "workflow_config_{}_Alpha{}.yaml".format(self.model_type.lower(), self.alpha)
            )
            filename = "alpha{}_handler_horizon{}.pkl".format(self.alpha, self.horizon)
            # raise AssertionError("Model type is not supported!")

        filename = f"{self.data_dir}_{self.market}_rank{self.rank_label}_{filename}"
        h_path = DIRNAME.parent / "baseline" / filename

        if self.h_path is not None:
            h_path = Path(self.h_path)

        with conf_path.open("r") as f:
            conf = yaml.safe_load(f)

        # modify dataset horizon
        conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["label"] = [
            "Ref($close, -{}) / Ref($close, -1) - 1".format(self.horizon + 1)
        ]

        if self.market != "csi300":
            conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["instruments"] = self.market
            if self.data_dir == "us_data":
                conf["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["label"] = [
                    "Ref($close, -{}) / $close - 1".format(self.horizon)
                ]

        batch_size = 5000
        if self.market == "csi100":
            batch_size = 2000
        elif self.market == "csi500":
            batch_size = 8000

        for k, v in {"early_stop": 8, "batch_size": batch_size, "lr": 0.001, "seed": None,}.items():
            if k in conf["task"]["model"]["kwargs"]:
                conf["task"]["model"]["kwargs"][k] = v
        if conf["task"]["model"]["class"] == "TransformerModel":
            conf["task"]["model"]["kwargs"]["dim_feedforward"] = 32
            conf["task"]["model"]["kwargs"]["reg"] = 0

        task = conf["task"]

        if self.task_ext_conf is not None:
            task = update_config(task, self.task_ext_conf)

        h_conf = task["dataset"]["kwargs"]["handler"]
        if not (self.model_type == "gbdt" and self.alpha == 158):
            expect_label_processor = "CSRankNorm" if self.rank_label else "CSZScoreNorm"
            delete_label_processor = "CSZScoreNorm" if self.rank_label else "CSRankNorm"
            proc = h_conf["kwargs"]["learn_processors"][-1]
            if (
                isinstance(proc, str) and self.rank_label and proc == delete_label_processor
                or
                isinstance(proc, dict) and proc["class"] == delete_label_processor
            ):
                h_conf["kwargs"]["learn_processors"] = h_conf["kwargs"]["learn_processors"][:-1]
                print("Remove", delete_label_processor)
                h_conf["kwargs"]["learn_processors"].append(
                    {"class": expect_label_processor, "kwargs": {"fields_group": "label"}}
                )
        print(h_conf)

        if not h_path.exists():
            h = init_instance_by_config(h_conf)
            h.to_pickle(h_path, dump_all=True)
            print('Save handler file to', h_path)

        # if not self.rank_label:
        #     task['model']['kwargs']['loss'] = 'ic'
        task["dataset"]["kwargs"]["handler"] = f"file://{h_path}"
        task["record"] = ["qlib.workflow.record_temp.SignalRecord"]

        if self.train_start is not None:
            seg = task["dataset"]["kwargs"]["segments"]["train"]
            task["dataset"]["kwargs"]["segments"]["train"] = pd.Timestamp(self.train_start), seg[1]

        if self.test_start is not None:
            seg = task["dataset"]["kwargs"]["segments"]["train"]
            task["dataset"]["kwargs"]["segments"]["test"] = pd.Timestamp(self.test_start), seg[1]

        if self.test_end is not None:
            seg = task["dataset"]["kwargs"]["segments"]["test"]
            task["dataset"]["kwargs"]["segments"]["test"] = seg[0], pd.Timestamp(self.test_end)
        self.logger.info(task)
        return task

    def create_rolling_tasks(self):
        task = self.basic_task()
        task_l = task_generator(
            task, RollingGen(step=self.step, trunc_days=self.horizon + 1)
        )  # the last two days should be truncated to avoid information leakage
        return task_l

    def train_rolling_tasks(self, task_l=None):
        if task_l is None:
            task_l = self.create_rolling_tasks()
        trainer = TrainerR(experiment_name=self.rolling_exp)
        trainer(task_l)

    def ens_rolling(self, exp_name):
        rc = RecorderCollector(
            experiment=exp_name,
            artifacts_key=["pred", "label"],
            process_list=[RollingEnsemble()],
            # rec_key_func=lambda rec: (self.COMB_EXP, rec.info["id"]),
            artifacts_path={"pred": "pred.pkl", "label": "label.pkl"},
        )
        res = rc()

        with R.start(experiment_name=self.COMB_EXP):
            R.log_params(exp_name=self.rolling_exp)
            R.save_objects(**{"pred.pkl": res["pred"], "label.pkl": res["label"]})
        return res["pred"]

    def update_rolling_rec(self, preds=None):
        """
        Evaluate the combined rolling results
        """
        backtest_config = {
            "strategy": {
                "class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy",
                "kwargs": {"signal": "<PRED>", "topk": 50, "n_drop": 5},
            },
            "backtest": {
                "start_time": None,
                "end_time": None,
                "account": 100000000,
                "benchmark": self.benchmark,
                "exchange_kwargs": {
                    "limit_threshold": None if self.data_dir == "us_data" else 0.095,
                    "deal_price": "close",
                    "open_cost": 0.0005,
                    "close_cost": 0.0015,
                    "min_cost": 5,
                },
            },
        }
        rec = R.get_exp(experiment_name=self.COMB_EXP).list_recorders(rtype=Experiment.RT_L)[0]
        SigAnaRecord(recorder=rec, skip_existing=True).generate()
        PortAnaRecord(recorder=rec, config=backtest_config, skip_existing=True).generate()
        label = init_instance_by_config(self.basic_task()["dataset"], accept_types=Dataset).\
            prepare(segments="test", col_set="label", data_key=DataHandlerLP.DK_L)
        if isinstance(label, TSDataSampler):
            label = pd.DataFrame({'label': label.data_arr[:-1][:, 0]}, index=label.data_index)
        else:
            label.columns = ['label']
        label['pred'] = preds.loc[label.index]
        # rmse = np.sqrt(((label['pred'].to_numpy() - label['label'].to_numpy()) ** 2).mean())
        mse = ((label['pred'].to_numpy() - label['label'].to_numpy()) ** 2).mean()
        mae = np.abs(label['pred'].to_numpy() - label['label'].to_numpy()).mean()
        rec.log_metrics(mse=mse, mae=mae)
        print(f"Your evaluation results can be found in the experiment named `{self.COMB_EXP}`.")
        return rec

    def run_all(self, task_l=None, reload_tag=None):
        # the results will be  save in mlruns.
        if reload_tag is None:
            # 1) each rolling task is saved in rolling_models
            self.train_rolling_tasks(task_l)
        else:
            self.tag = reload_tag
        # 2) combined rolling tasks and evaluation results are saved in rolling
        preds = self.ens_rolling(self.rolling_exp)
        rec = self.update_rolling_rec(preds)
        return rec

    def run_exp(self):
        all_metrics = {
            k: []
            for k in [
                'mse', 'mae',
                "IC",
                "ICIR",
                "Rank IC",
                "Rank ICIR",
                "1day.excess_return_with_cost.annualized_return",
                "1day.excess_return_with_cost.information_ratio",
                "1day.excess_return_with_cost.max_drawdown",
            ]
        }
        test_time = []
        R.set_uri((DIRNAME / 'mlruns').as_uri())
        for i in range(5):
            np.random.seed(i + 43)
            torch.manual_seed(i + 43)
            torch.cuda.manual_seed(i + 43)
            start_time = time.time()
            self.tag = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
            reload_tag = None
            if self.reload_tags and len(self.reload_tags) > i:
                self.tag = self.reload_tags[i]
                try:
                    print(self.tag)
                    R.get_exp(experiment_name=self.rolling_exp, create=False)
                    reload_tag = self.reload_tags[i]
                except Exception as e:
                    self.logger.info(e)
                    self.logger.info('Start retraining from scratch...')
                    reload_tag = None
            rec = self.run_all(reload_tag=reload_tag)
            test_time.append(time.time() - start_time)
            # exp = R.get_exp(experiment_name=self.COMB_EXP)
            # rec = exp.list_recorders(rtype=exp.RT_L)[0]
            metrics = rec.list_metrics()
            for k in all_metrics.keys():
                all_metrics[k].append(metrics[k])
            pprint(all_metrics)

        with R.start(
            experiment_name=f"final_{self.data_dir}_{self.market}_{self.alpha}_{self.horizon}_{self.model_type}"
        ):
            R.save_objects(all_metrics=all_metrics)
            test_time = np.array(test_time)
            R.log_metrics(test_time=test_time)
            print(f"Time cost: {test_time.mean()}")
            res = {}
            for k in all_metrics.keys():
                v = np.array(all_metrics[k])
                res[k] = [v.mean(), v.std()]
                R.log_metrics(**{"final_" + k: res[k]})
            pprint(res)
        test_time = np.array(test_time)
        print(f"Time cost: {test_time.mean()}")


if __name__ == "__main__":
    # GetData().qlib_data(exists_skip=True)
    # auto_init()
    fire.Fire(RollingBenchmark)
