
import sys
from pathlib import Path
DIRNAME = Path(__file__).absolute().resolve().parent

sys.path.append(str(DIRNAME))
sys.path.append(str(DIRNAME.parent.parent))

import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
import warnings
import pickle
import qlib.contrib.data.handler as handler

warnings.filterwarnings('ignore')


class Raw(handler.Alpha158):

    def get_feature_config(self):
        fields = ['$open', '$high', '$low', '$close', '$volume']
        names = ['open', 'high', 'low', 'close', 'volume']

        return fields, names

    def get_label_config(self):
        return (["Ref($close, -2)/Ref($close, -1) - 1"], ["label"])


qlib.contrib.data.handler.Raw = Raw

# use default data
# NOTE: need to download data from remote: python scripts/get_data.py qlib_data_cn --target_dir ~/.qlib/qlib_data/cn_data
provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)

root_path = '/data/zhaolifan/project/qlib/examples/ensemble/'

with open(root_path + 'pkl_data/dataset.pkl', 'rb') as f:
    data = pickle.load(f)
dataset = data[0]
dataset.handler.__dict__.update(data[1])

import pickle
import numpy as np

with open(root_path + 'pkl_data/base_models.pkl', 'rb') as f: feature_data = pickle.load(f)
with open(root_path + 'pkl_data/label.pkl', 'rb') as f: label = pickle.load(f)

with open(root_path + 'pkl_data/model_weight.pkl', 'rb') as f: model_weight = pickle.load(f).values
if len(model_weight) < len(label): model_weight = np.vstack((
    np.ones((len(label) - len(model_weight), 6)),
    model_weight,
))

label_norm_df = feature_data['label']
label_norm = label_norm_df.loc[label.index].values.squeeze()

dates = feature_data.index.get_level_values(level='datetime')
backtest_start_index = (dates == label.index[0][0]).argmax()

train_data = feature_data[:backtest_start_index]
pred_data = feature_data[backtest_start_index:]
train_X = train_data.values[:, 1:]
train_y = train_data.values[:, 0]
X = pred_data.values[:, 1:]

w_lr = np.abs(train_X).mean(axis=0, keepdims=True)
w_lr /= w_lr.sum()

methods = {
    'LSTM': X[:, 4],
    'Average': X.mean(axis=1),
    'Linear': (w_lr * X).sum(axis=1),
    'REnsemble': (model_weight * X).sum(axis=1),
}

back_test_data = methods['Linear']
back_test_data_df = pd.DataFrame(back_test_data, index=label.index)

import pandas as pd
from qlib.model.base import Model as qlibModel
from qlib.contrib.eva.alpha import calc_ic


class Model(qlibModel):

    def __init__(self, back_test_data_df): self.pred = back_test_data_df

    def fit(self, dataset): pass

    def predict(self, dataset): return self.pred


def bt(back_test_data):
    back_test_data
    back_test_data_df = pd.DataFrame(back_test_data, index=label.index)
    model = Model(back_test_data_df)

    with R.start(experiment_name="test"):
        recorder = R.get_recorder()
        ba_rid = recorder.id
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        sar = SigAnaRecord(recorder)
        pred = sar.load("pred.pkl")
        lb = sar.load("label.pkl")
        ic, ric = calc_ic(pred.iloc[:, 0], lb.iloc[:, sar.label_col])
        # error = back_test_data - label_norm
        # error[np.isinf(error)] = np.nan
        metrics = {
            "IC": ic.mean(),
            "ICIR": ic.mean() / ic.std(),
            # "Rank IC": ric.mean(),
            # "Rank ICIR": ric.mean() / ric.std(),
            # 'MSE': np.nanmean((error)**2)*0.5,
            # 'MAE': np.nanmean(np.abs(error))
        }
        # sar.generate()
        return metrics


df_dic = dict()
for k, v in methods.items():
    df_dic[k] = bt(v)

eval_df = pd.DataFrame(df_dic).T
# eval_df.to_excel('result.xlsx')
print(eval_df)
