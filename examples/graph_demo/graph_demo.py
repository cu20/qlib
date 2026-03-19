import qlib
from qlib.data import D
from qlib.constant import REG_CN, REG_US
from qlib.contrib.graph import stock_concept_data, stock_stock_data
import numpy as np

provider_uri = "~/.qlib/qlib_data/cn_data"
region = REG_CN
qlib.init(provider_uri=provider_uri, region=region)
market = "A_share"
instrument = "csi300"
train_start_date = "2008-01-01"
train_end_date = "2019-12-31"
valid_start_date = "2020-01-01"
valid_end_date = "2020-12-31"
test_start_date = "2021-01-01"
test_end_date = "2022-08-31"

universe = D.list_instruments(
    D.instruments(instrument), start_time=train_start_date, end_time=test_end_date
)
stocks_sorted_list = sorted(list(universe.keys()))
stocks_index_dict = {}
for i, stock in enumerate(stocks_sorted_list):
    stocks_index_dict[stock] = i

# 获取行业数据
industry_dict = stock_concept_data.read_graph_dict(
    market,
    relation_name="SW_belongs_to",
    data_path="/home/zhangzexi/.qlib/qlib_data/graph_data/",
)
industry_matrix = stock_concept_data.get_full_connection_matrix(
    industry_dict, stocks_index_dict
)
print(industry_matrix.shape)
print(np.sum(industry_matrix))

# 获取stock-stock数据
stock_stock_matrix = stock_stock_data.get_all_matrix(
    market, stocks_index_dict, data_path="/home/zhangzexi/.qlib/qlib_data/graph_data/"
)
print(stock_stock_matrix.shape)
print(np.sum(stock_stock_matrix))
