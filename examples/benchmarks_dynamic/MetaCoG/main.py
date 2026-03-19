# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pathlib import Path
import sys
DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.append(str(DIRNAME))
sys.path.append(str(DIRNAME.parent.parent.parent))

import torch
from qlib.utils import init_instance_by_config
from qlib.workflow import R, Experiment
from qlib.contrib.meta.incremental.model import MetaCoG
import fire

from examples.benchmarks_dynamic.incremental.main import Incremental
from examples.benchmarks.benchmark import Benchmark


class CML(Incremental):

    def offline_training(self, seed=43):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        benchmark = Benchmark(
            data_dir=self.data_dir,
            model_type=self.forecast_model,
            alpha=self.alpha,
            market=self.market,
            rank_label=self.rank_label,
            init_data=False
        )
        R.set_uri("../../benchmarks/mlruns/")
        model = benchmark.get_fitted_model(f"_{seed}")
        R.set_uri("./mlruns/")

        # with R.start(experiment_name=self.meta_exp_name):
        mm = MetaCoG(
            self.basic_task,
            is_rnn=self.is_rnn,
            alpha=self.alpha,
            lr_model=0.001,
            first_order=self.first_order,
            pretrained_model=model,
        )
        mm.fit(self.meta_dataset_offline)

        if model is None:
            with R.start(experiment_name=benchmark.exp_name + f"_{seed}"):
                model = init_instance_by_config(benchmark.basic_task()["model"])
                model.model = mm.framework.model
                R.save_objects(**{"params.pkl": model})
            # R.save_objects(model=mm)
        return mm

    @property
    def meta_exp_name(self):
        return f"MetaCoG_{self.market}_{self.forecast_model}_alpha{self.alpha}_horizon{self.horizon}_step{self.step}_rank{self.rank_label}_{self.tag}"


if __name__ == "__main__":
    fire.Fire(CML)
