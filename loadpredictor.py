import pprint
from functools import partial

import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.distribution.piecewise_linear import PiecewiseLinearOutput
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.seq2seq import MQCNNEstimator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.trainer import Trainer
from pathlib import Path
from gluonts.model.predictor import Predictor

predictor = Predictor.deserialize(Path("./tmp/"))

df = pd.read_csv("./k_data.csv", header=0, index_col=0)
print(Path("./tmp/"))
test_data = ListDataset(
    [{"start": df.index[0], "target": df.value,"cat":[df.code],"dynamic_feat":[df.volume,df.turn]}],
    freq = "1d"
)


forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_data,  # test dataset
    predictor=predictor,  # predictor
    num_samples=20,  # number of sample paths we want for evaluation
)

forecasts = list(forecast_it)
tss = list(ts_it)
print(forecasts)
print(tss)
result1 = pd.DataFrame(forecasts)
result1.to_csv("./forecasts.csv", encoding="gbk", index=False)
result = pd.DataFrame(tss)
result.to_csv("./tss.csv", encoding="gbk", index=False)