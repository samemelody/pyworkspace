

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


prediction_length = 16
df = pd.read_csv("./traindata.csv", header=0, index_col=0)
print(Path("./tmp/"))
training_data = ListDataset(
    [{"start": df.index[0], "target": df.value,"cat":[df.code],"dynamic_feat":[df.volume,df.turn]}],
    freq = "1d"
)
print(training_data)

df = pd.read_csv("./k_data.csv", header=0, index_col=0)
print(Path("./tmp/"))
test_data = ListDataset(
    [{"start": df.index[0], "target": df.value,"cat":[df.code],"dynamic_feat":[df.volume,df.turn]}],
    freq = "1d"
)
print(training_data)

estimator = DeepAREstimator(
    prediction_length=prediction_length,
    context_length=100,
    freq="1d",
    trainer=Trainer(ctx="cpu", 
                    epochs=5, 
                    learning_rate=1e-3, 
                    num_batches_per_epoch=100
                   )
)

train_output = estimator.train(training_data)

train_output.serialize(Path("./tmp/"))


forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_data,  # test dataset
    predictor=train_output,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation
)



forecasts = list(forecast_it)
tss = list(ts_it)

print(forecasts)
print(tss)