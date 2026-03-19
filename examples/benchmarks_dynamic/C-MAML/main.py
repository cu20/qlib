# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import copy
from pathlib import Path
import sys

from qlib.utils.data import deepcopy_basic_type

DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.append(str(DIRNAME))
sys.path.append(str(DIRNAME.parent.parent.parent))

import torch

from qlib.data.dataset import Dataset, TSDataSampler

from qlib.workflow.task.gen import RollingGen
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP

import fire
from examples.benchmarks_dynamic.incremental.main import Incremental
from qlib.contrib.meta.incremental.utils import preprocess
from qlib.contrib.meta.incremental.dataset import MetaDatasetInc
from qlib.contrib.meta.incremental.model import CMAML


class OKASA(Incremental):

    @property
    def meta_exp_name(self):
        return f"CMAML_{self.market}_{self.forecast_model}_alpha{self.alpha}_horizon{self.horizon}_step{self.step}_rank{self.rank_label}_{self.tag}"

    def dump_data(self):
        segments = self.basic_task["dataset"]["kwargs"]["segments"]

        t = copy.deepcopy(self.basic_task)
        t["dataset"]["kwargs"]["segments"]["train"] = (
            segments["train"][0],
            segments["test"][1],
        )
        ds = init_instance_by_config(t["dataset"], accept_types=Dataset)
        data = ds.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if t["dataset"]["class"] == "TSDatasetH":
            data.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
            # self.L = rolling_task['dataset']['kwargs']['step_len']
        # else:
        #     data = None

        rolling_task = deepcopy_basic_type(self.basic_task)
        if "pt_model_kwargs" in rolling_task["model"]["kwargs"] and rolling_task["model"]["class"] != "DNNModelPytorch":
            self.factor_num = rolling_task["model"]["kwargs"]["pt_model_kwargs"]["input_dim"]
        elif "d_feat" in rolling_task["model"]["kwargs"]:
            self.factor_num = rolling_task["model"]["kwargs"]["d_feat"]
        else:
            self.factor_num = 6 if self.alpha == 360 else 20

        trunc_days = self.horizon if self.data_dir == "us_data" else (self.horizon + 1)
        gen = RollingGen(step=self.step, rtype=RollingGen.ROLL_SD)
        segments = rolling_task["dataset"]["kwargs"]["segments"]
        train_begin = segments["train"][0]
        train_end = gen.ta.get(gen.ta.align_idx(train_begin) + gen.step - 1)
        test_begin = gen.ta.get(gen.ta.align_idx(train_begin) + gen.step - 1 + trunc_days)
        test_end = segments["valid"][1]
        test_end = gen.ta.get(gen.ta.align_idx(test_end) - gen.step)
        # extra_begin = gen.ta.get(gen.ta.align_idx(train_end) + 1)
        # extra_end = gen.ta.get(gen.ta.align_idx(test_begin) - 1)
        seperate_point = str(rolling_task["dataset"]["kwargs"]["segments"]["valid"][0])
        rolling_task["dataset"]["kwargs"]["segments"] = {
            "train": (segments["train"][0], train_end),
            # "extra": (extra_begin, extra_end),
            "test": (test_begin, test_end),
        }

        kwargs = dict(
            task_tpl=rolling_task,
            step=self.step,
            # segments=0.7,  # keep test period consistent with the dataset yaml
            segments=seperate_point,
            # trunc_days=self.horizon+1,
            task_mode="train",
        )
        if self.forecast_model == "MLP" and self.alpha == 158:
            kwargs.update(task_mode="test")
        md_offline = MetaDatasetInc(data=data, **kwargs)
        md_offline.meta_task_l = preprocess(
            md_offline.meta_task_l,
            factor_num=self.factor_num,
            is_mlp=self.forecast_model == "MLP",
            alpha=self.alpha,
            step=self.step,
            H=self.horizon if self.data_dir == "us_data" else (1 + self.horizon),
        )
        L = md_offline.meta_task_l[0].get_meta_input()["X_test"].shape[1]
        if self.not_sequence:
            self.x_dim = L
            self.factor_num = self.x_dim
        else:
            self.x_dim = self.factor_num * L

        test_begin = segments["valid"][0]
        # train_end = gen.ta.get(gen.ta.align_idx(train_begin) + gen.step - 1)
        # test_begin = gen.ta.get(gen.ta.align_idx(train_begin) + gen.step - 1 + trunc_days)
        rolling_task["dataset"]["kwargs"]["segments"] = {
            "test": (test_begin, segments["test"][1]),
            # "extra": (extra_begin, extra_end),
            # "test": (test_begin, segments['test'][1]),
        }
        # if trunc_days > 1:
        #     extra_end = gen.ta.get(gen.ta.align_idx(test_begin) - 1)
        #     extra_begin = gen.ta.get(gen.ta.align_idx(test_begin) - trunc_days + 1)
        #     rolling_task["dataset"]["kwargs"]["segments"]["train"] = (
        #         extra_begin,
        #         extra_end,
        #     )

        kwargs.update(task_tpl=rolling_task, segments=0.0)
        if self.forecast_model == "MLP" and self.alpha == 158:
            kwargs.update(task_mode="test")
            data_I = ds.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        else:
            data_I = None
        md_online = MetaDatasetInc(data=data, data_I=data_I, **kwargs)
        md_online.meta_task_l = preprocess(
            md_online.meta_task_l,
            factor_num=self.factor_num,
            is_mlp=self.forecast_model == "MLP",
            alpha=self.alpha,
            step=self.step,
            H=self.horizon if self.data_dir == "us_data" else (1 + self.horizon),
        )
        return md_offline, md_online

    def offline_training(self, seed=43):
        """
        training a meta model based on a simplified linear proxy model;
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # with R.start(experiment_name=self.meta_exp_name):
        mm = CMAML(
            self.basic_task,
            sample_num=8000 if self.market == "csi500" else 5000,
            x_dim=self.x_dim,
            is_rnn=self.is_rnn,
            alpha=self.alpha,
            first_order=self.first_order,
            pretrained_model=None,
        )
        mm.fit(self.meta_dataset_offline)
        # R.save_objects(model=mm)
        return mm


if __name__ == "__main__":
    fire.Fire(OKASA)
