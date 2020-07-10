#!/usr/bin/env python
# coding: utf-8

# 

# 

# In[23]:



import pprint
from functools import partial
import baostock as bs
import pandas as pd

from gluonts.dataset.common import ListDataset
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.distribution.piecewise_linear import PiecewiseLinearOutput
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from pathlib import Path
from gluonts.model.predictor import Predictor
from gluonts.distribution.neg_binomial import NegativeBinomialOutput
import matplotlib.pyplot as plt
import json
#tqdm.autonotebook.tqdm
from tqdm.autonotebook import tqdm


# In[24]:


def mygetstockdata(code):
    
   # print('login respond error_code:'+lg.error_code)
   # print('login respond  error_msg:'+lg.error_msg)
    rs = bs.query_history_k_data_plus(code,
        "date,close,volume,turn",
        start_date='2019-01-01', 
        frequency="d", adjustflag="2") #frequency="d"取日k线，adjustflag="3"默认不复权
   
    return rs.get_data()


# In[25]:


liststock =['sz.300807','sz.300789','sz.300771','sz.300546','sz.300479','sz.300462','sz.300455','sz.300449','sz.300386','sz.300368']
prediction_length = 30
#liststock = ['sz.300462','sz.300789']
listtrandic = []
listtestdic = []
lg = bs.login()
for ite in liststock:
    dd = mygetstockdata(ite)
    trandic = {"start":dd.date[0],"target":list(dd.close),"cat":int(liststock[0].split('.')[1]),"feat_dynamic":[list(dd.volume),list(dd.turn)]}
    testdic = {"start":dd.date[0],
                   "target":(dd.close)[:-prediction_length],
                   "cat":int(liststock[0].split('.')[1]),
                   "feat_dynamic":[(dd.volume)[:-prediction_length],(dd.turn)[:-prediction_length]]}
    #strjon = json.dumps(dic)
    listtrandic.append(trandic)
    listtestdic.append(testdic)
bs.logout()

traindata = ListDataset(
    listtrandic,
    freq = "1d"
)

testdata = ListDataset(
    listtestdic,
    freq = "1d"
)


# In[ ]:





# In[38]:


#prediction_length = 30
estimator = DeepAREstimator(
    prediction_length=prediction_length,
    freq="1d",
    distr_output = NegativeBinomialOutput(),
    trainer=Trainer(ctx="cpu",
                    epochs=100,#30
                    learning_rate=1e-5,
                    num_batches_per_epoch=100, #100
                     batch_size=64
                   )
)
predictor = estimator.train(traindata)

#predictor.serialize(Path("./tmp/"))


# In[35]:


from gluonts.evaluation.backtest import make_evaluation_predictions
from tqdm.autonotebook import tqdm
forecast_it, ts_it = make_evaluation_predictions(
    dataset=testdata,
    predictor=predictor,
    num_samples=100
)

print("Obtaining time series conditioning values ...")
tss = list(tqdm(ts_it, total=len(testdata)))
print("Obtaining time series predictions ...")
forecasts = list(tqdm(forecast_it, total=len(testdata)))


# In[36]:


from gluonts.evaluation import Evaluator
import numpy as np

class CustomEvaluator(Evaluator):

    def get_metrics_per_ts(self, time_series, forecast):
        successive_diff = np.diff(time_series.values.reshape(len(time_series)))
        successive_diff = successive_diff ** 2
        successive_diff = successive_diff[:-prediction_length]
        denom = np.mean(successive_diff)
        pred_values = forecast.samples.mean(axis=0)
        true_values = time_series.values.reshape(len(time_series))[-prediction_length:]
        num = np.mean((pred_values - true_values) ** 2)
        rmsse = num / denom
        metrics = super().get_metrics_per_ts(time_series, forecast)
        metrics["RMSSE"] = rmsse
        return metrics

    def get_aggregate_metrics(self, metric_per_ts):
        wrmsse = metric_per_ts["RMSSE"].mean()
        agg_metric, _ = super().get_aggregate_metrics(metric_per_ts)
        agg_metric["MRMSSE"] = wrmsse
        return agg_metric, metric_per_ts


evaluator = CustomEvaluator(quantiles=[0.5, 0.67, 0.95, 0.99])
agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(testdata))
print(json.dumps(agg_metrics, indent=4))


# In[37]:


import os

plot_log_path = "./plots/"
directory = os.path.dirname(plot_log_path)
if not os.path.exists(directory):
    os.makedirs(directory)
    

def plot_prob_forecasts(ts_entry, forecast_entry, path, sample_id, inline=True):
    plot_length = 150
    prediction_intervals = (50, 90)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    _, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    ax.axvline(ts_entry.index[-prediction_length], color='r')
    plt.legend(legend, loc="upper left")
    if inline:
        plt.show()
        plt.clf()
    else:
        plt.savefig('{}forecast_{}.pdf'.format(path, sample_id))
        plt.close()

print("Plotting time series predictions ...")
for i in tqdm(range(10)):
    ts_entry = tss[i]
    forecast_entry = forecasts[i]
    plot_prob_forecasts(ts_entry, forecast_entry, plot_log_path, i)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




