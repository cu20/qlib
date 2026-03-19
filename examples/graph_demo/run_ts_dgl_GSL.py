from pathlib import Path
import sys

DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.append(str(DIRNAME))
sys.path.append(str(DIRNAME.parent.parent.parent))


import qlib
from qlib.constant import REG_CN, REG_US
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_BENCH, CSI300_DATASET_CONFIG
from qlib.contrib.graph import stock_concept_data, stock_stock_data
from qlib.data import D
from qlib import get_module_logger


from GSL_utils import setup_seed, Logger, get_parser
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import pprint


def load_graph(market, instrument, start_date, end_date, relation_source="industry"):
    universe = D.list_instruments(
        D.instruments(instrument), start_time=start_date, end_time=end_date
    )
    stocks_sorted_list = sorted(list(universe.keys()))
    stocks_index_dict = {}
    for i, stock in enumerate(stocks_sorted_list):
        stocks_index_dict[stock] = i

    if relation_source == "stock-stock":
        return (
            stock_stock_data.get_all_matrix(
                market,
                stocks_index_dict,
                data_path="/home/zhangzexi/.qlib/qlib_data/graph_data/",
            ),
            stocks_sorted_list,
        )
    elif relation_source == "industry":
        industry_dict = stock_concept_data.read_graph_dict(
            market,
            relation_name="SW_belongs_to",
            data_path="/home/zhangzexi/.qlib/qlib_data/graph_data/",
        )
        return (
            stock_concept_data.get_full_connection_matrix(
                industry_dict, stocks_index_dict
            ),
            stocks_sorted_list,
        )
    else:
        raise ValueError("unknown graph name `%s`" % relation_source)


def load_correlation_graph(
    market, stocks_list, dates, time_window, dataset_path, force_refresh=False
):
    # 从/data/zhangzexi中读取已经处理好的每天的相关性矩阵, 作为graph learning的真值
    # 经过实验,从csv转化为最终的npy文件,还是比较慢的,因此还是要暂存-->集成到Price_data中
    file_path = os.path.join(
        dataset_path, f"dynamicA_{dates[0]}_{dates[-1]}_{time_window}.npy"
    )
    if force_refresh or not os.path.exists(file_path):
        dfs_folder_name = (
            f"A_share_qlib_csi300_{time_window}"
            if market == "A_share"
            else f"{market}_all"
        )
        all_A = []
        path = os.path.join("/data/zhangzexi/correlation_A", dfs_folder_name)
        stock_num = len(stocks_list)
        for date in tqdm(dates):
            large_df = pd.DataFrame(
                np.zeros((stock_num, stock_num)),
                columns=stocks_list,
                index=stocks_list,
                dtype=np.float16,
            )  # 会存在数据缺失的问题 设置large_df
            if os.path.exists(f"{path}/{date}.csv"):
                df = pd.read_csv(
                    f"{path}/{date}.csv",
                    index_col="instrument",
                    dtype={stock: np.float16 for stock in stocks_list},
                )
                large_df.loc[df.columns, df.columns] = df.values
            else:
                print(f"{path}/{date}.csv 无文件, 跳过")
            all_A.append(large_df.values)
        all_A = np.stack(all_A, axis=0)
        np.save(file_path, all_A)
    all_A = np.load(file_path)
    return all_A


def make_config(
    n_epochs,
    early_stop,
    logger=None,
    toy=False,
    graph_type="original",
    model="GAT",
    generate_weight=False,
):
    if toy:
        train_start_date = "2008-01-01"
        train_end_date = "2009-12-31"
        valid_start_date = "2010-01-01"
        valid_end_date = "2010-12-31"
        test_start_date = "2011-01-01"
        test_end_date = "2011-08-01"
    else:
        train_start_date = "2008-01-01"
        train_end_date = "2014-12-31"
        valid_start_date = "2015-01-01"
        valid_end_date = "2016-12-31"
        test_start_date = "2017-01-01"
        test_end_date = "2020-08-01"

    market = "A_share"
    instrument = "csi300"

    dataset_path = os.path.join(
        "/home/zhangzexi/zhangzexi/pycharmProjects/quant_relational_model_new_framework/data",
        f"{market}_{instrument}_{train_start_date}_{train_end_date}_{valid_start_date}_{valid_end_date}_{test_start_date}_{test_end_date}",
    )

    rel_encoding, stock_name_list = load_graph(
        market, instrument, train_start_date, test_end_date, "stock-stock"
    )

    if graph_type == "original":
        use_corr_en = False
        corr_en = None
    elif graph_type == "self_loop":
        # 使用只有自环
        use_corr_en = False
        corr_en = None
        rel_encoding = np.eye(len(stock_name_list)).astype(np.float64)
    elif graph_type == "all_1":
        # 使用全连接图
        use_corr_en = False
        corr_en = None
        rel_encoding = np.ones((len(stock_name_list), len(stock_name_list))).astype(
            np.float64
        )
    elif graph_type == "finetune_by_relation":
        use_corr_en = True
        train_corr_en = load_correlation_graph(
            market=market,
            stocks_list=stock_name_list,
            dates=D.calendar(
                start_time=train_start_date, end_time=train_end_date, freq="day"
            ),
            time_window=20,
            dataset_path=dataset_path,
            force_refresh=False,
        )
        valid_corr_en = load_correlation_graph(
            market=market,
            stocks_list=stock_name_list,
            dates=D.calendar(
                start_time=valid_start_date, end_time=valid_end_date, freq="day"
            ),
            time_window=20,
            dataset_path=dataset_path,
            force_refresh=False,
        )
        test_corr_en = load_correlation_graph(
            market=market,
            stocks_list=stock_name_list,
            dates=D.calendar(
                start_time=test_start_date, end_time=test_end_date, freq="day"
            ),
            time_window=20,
            dataset_path=dataset_path,
            force_refresh=False,
        )
        corr_en = {"train": train_corr_en, "valid": valid_corr_en, "test": test_corr_en}

    config = {}
    config["model"] = {
        "class": "Graphs",
        "module_path": "pytorch_ts_GSL",
        "kwargs": {
            "graph_model": model,
            "d_feat": 6,
            "hidden_size": 64,
            "num_layers": 1,
            "loss": "mse",
            "dropout": 0.7,
            "n_epochs": n_epochs,  # 这里原来是200
            "metric": "loss",
            "base_model": "LSTM",
            "GPU": 0,
            "lr": 1e-4,
            "early_stop": early_stop,
            "rel_encoding": rel_encoding,
            "use_corr_en": use_corr_en,
            "corr_en": corr_en,
            "stock_name_list": stock_name_list,
            "loss": "mse",  # another choise: rank_mse
            "logger": logger,
            "generate_weight": generate_weight,
        },
    }
    config["log"] = {
        "class": "Graphs",
        "module_path": "pytorch_ts_GSL",
        "kwargs": {
            # "graph_model": "GAT",  # or 'simpleHGN'
            "graph_model": "GSLGraphModel",
            "d_feat": 6,
            "hidden_size": 64,
            "num_layers": 1,
            "loss": "mse",
            "dropout": 0.7,
            "n_epochs": n_epochs,
            "metric": "loss",
            "base_model": "LSTM",
            "GPU": 3,
            "lr": 1e-4,
            "early_stop": early_stop,
        },
    }

    dh_config = {
        "start_time": train_start_date,
        "end_time": test_end_date,
        "fit_start_time": train_start_date,
        "fit_end_time": train_end_date,
        "infer_processors": [
            {
                "class": "RobustZScoreNorm",
                "kwargs": {"clip_outlier": True, "fields_group": "feature"},
            },
            {"class": "Fillna", "kwargs": {"fill_value": 0, "fields_group": "feature"}},
        ],
        "instruments": instrument,
        "label": ["Ref($close, -2) / Ref($close, -1) - 1"],
        "learn_processors": [
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ],
    }

    handler = {
        "class": "Alpha360",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": dh_config,
    }

    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler,
            "segments": {
                "train": (train_start_date, train_end_date),
                "valid": (valid_start_date, valid_end_date),
                "test": (test_start_date, test_end_date),
            },
        },
    }
    config["dataset"] = dataset_config

    return config, dataset_path


def make_port_config(model, dataset):
    return {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "start_time": "2017-01-01",
            "end_time": "2020-08-01",
            "account": 100000000,
            "benchmark": CSI300_BENCH,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }


def load_dataset(dataset_config, dataset_path, force_refresh=False):
    if not os.path.exists(dataset_path) or force_refresh:
        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)
        dataset = init_instance_by_config(dataset_config)
        with open(os.path.join(dataset_path, "dataset.pkl"), "wb") as f:
            pickle.dump(dataset, f)
        with open(os.path.join(dataset_path, "dataset_learn.pkl"), "wb") as f:
            pickle.dump(dataset.handler._learn, f)
        with open(os.path.join(dataset_path, "dataset_data.pkl"), "wb") as f:
            pickle.dump(dataset.handler._data, f)
        with open(os.path.join(dataset_path, "dataset_infer.pkl"), "wb") as f:
            pickle.dump(dataset.handler._infer, f)
    with open(os.path.join(dataset_path, "dataset.pkl"), "rb") as f:
        dataset = pickle.load(f)
    with open(os.path.join(dataset_path, "dataset_learn.pkl"), "rb") as f:
        dataset_learn = pickle.load(f)
        dataset.handler._learn = dataset_learn
    with open(os.path.join(dataset_path, "dataset_data.pkl"), "rb") as f:
        dataset_data = pickle.load(f)
        dataset.handler._data = dataset_data
    with open(os.path.join(dataset_path, "dataset_infer.pkl"), "rb") as f:
        dataset_infer = pickle.load(f)
        dataset.handler._infer = dataset_infer
    return dataset


if __name__ == "__main__":
    setup_seed(2023)
    # use default data
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    args = get_parser()
    My_Logger = Logger(info="", toy=False)
    logger = My_Logger.logger
    logger.info(args.graph_type)
    logger.info(f"using model:{args.model}")
    logger.info(f"generate weight:{args.generate_weight}")
    config, dataset_path = make_config(
        n_epochs=args.epochs,
        early_stop=args.early_stop,
        logger=logger,
        toy=False,
        graph_type=args.graph_type,
        model=args.model,
        generate_weight=args.generate_weight,
    )
    model = init_instance_by_config(config["model"])
    dataset = load_dataset(config["dataset"], dataset_path, force_refresh=False)

    port_analysis_config = make_port_config(model, dataset)

    # start exp
    with R.start(experiment_name="gats_dgl"):
        R.log_params(**flatten_dict(config["log"]))
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        sar = SigAnaRecord(recorder)
        metrics = sar.generate()
        IC, ICIR, RankIC, RankICIR = (
            metrics["IC"],
            metrics["ICIR"],
            metrics["Rank IC"],
            metrics["Rank ICIR"],
        )
        test_result = {
            "IC": IC,
            "ICIR": ICIR,
            "RankIC": RankIC,
            "RankICIR": RankICIR,
        }
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        (
            bench_result,
            excess_return_without_cost,
            excess_return_with_cost,
            analysis_df,
        ) = par.generate()
        benchmark_return = bench_result.at["annualized_return", "risk"]
        ann_excess_return, information_ratio = (
            excess_return_with_cost.at["annualized_return", "risk"],
            excess_return_with_cost.at["information_ratio", "risk"],
        )
        test_result["benchmark_return"] = benchmark_return
        test_result["ann_excess_return"] = ann_excess_return
        test_result["information_ratio"] = information_ratio
        logger.info(pprint.pformat(test_result))
