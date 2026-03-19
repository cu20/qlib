import torch as torch
import numpy as np
import random
import os
from qlib import get_module_logger
import logging
import argparse


def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)


class Logger(object):
    def __init__(self, info="", toy=False):
        self.info = info
        self.toy = toy
        self.folder_path, self.log_path = self.get_log_path()
        self.logger = get_module_logger("run")
        fh = logging.FileHandler(self.log_path)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def get_log_path(self):
        path = "/home/zhangzexi/zhangzexi/pycharmProjects/quant_relational_model_new_framework/logs"
        if self.toy:
            path = os.path.join(path, "toy_experiment")
        today, now = self.get_acc_time()
        if not os.path.exists(os.path.join(path, today)):
            os.mkdir(os.path.join(path, today))
        folder_path = os.path.join(path, today, f"{now}_{self.info}")
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        log_path = os.path.join(folder_path, f"{now}_{self.info}.log")
        return folder_path, log_path

    def get_folder_path(self):
        return self.folder_path

    def get_acc_time(self):
        from datetime import timezone, timedelta, datetime

        beijing = timezone(timedelta(hours=8))
        utc_time = datetime.utcnow().replace(tzinfo=timezone.utc)
        time_beijing = utc_time.astimezone(beijing)
        today = str(time_beijing.date())
        now = str(time_beijing.time())[:8]
        return today, now


def get_parser():
    parser = argparse.ArgumentParser(description="commind parameter")
    parser.add_argument("-epochs", dest="epochs", type=int, help="epochs", default=1)
    parser.add_argument(
        "-early_stop", dest="early_stop", type=int, help="early_stop", default=30
    )
    parser.add_argument(
        "-graph_type",
        dest="graph_type",
        type=str,
        help="graph_type",
        default="original",
    )
    parser.add_argument(
        "-model",
        dest="model",
        type=str,
        help="model",
        default="GAT",
    )  # choose from GAT, GSLGraphModel
    parser.add_argument(
        "-GSL_conv_type",
        dest="GSL_conv_type",
        type=str,
        help="GSL_conv_type",
        default="GCN",
    )  # choose from GAT, GCN
    parser.add_argument("-residual", action="store_true", help="residual")
    args = parser.parse_args()
    return args
